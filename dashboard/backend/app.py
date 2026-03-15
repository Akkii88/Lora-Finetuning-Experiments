from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import os

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_CHECKPOINT = 'distilbert-base-uncased'
LORA_ADAPTER_PATH = '../../distilbert-lora-imdb-final/checkpoint-250'
PROMPT_TUNING_ADAPTER_PATH = '../../prompt-tuned-imdb'
IA3_ADAPTER_PATH = '../../ia3-tuned-imdb'
ID2LABEL = {0: "Negative", 1: "Positive"}

# Global variables for models
models = {}
tokenizer = None

def load_models():
    global tokenizer
    print("Loading models and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    
    # 1. Base Model
    print("Loading Base Model...")
    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=2)
    models['base'] = base_model
    
    # 2. LoRA Model
    print("Loading LoRA Model...")
    # Need a fresh base for LoRA to avoid conflicts
    lora_base = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=2)
    models['lora'] = PeftModel.from_pretrained(lora_base, LORA_ADAPTER_PATH)
    
    # 3. Prompt-Tuning Model
    print("Loading Prompt-Tuning Model...")
    pt_base = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=2)
    models['prompt'] = PeftModel.from_pretrained(pt_base, PROMPT_TUNING_ADAPTER_PATH)
    
    # 4. IA3 Model
    print("Loading IA3 Model...")
    try:
        ia3_base = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT, num_labels=2)
        models['ia3'] = PeftModel.from_pretrained(ia3_base, IA3_ADAPTER_PATH)
    except Exception as e:
        print(f"Skipping IA3: {e}")

    for name in models:
        models[name].eval()
        
    print("Models loaded successfully.")

@app.route('/metrics', methods=['GET'])
def get_metrics():
    # These are the hardcoded metrics from our research results
    metrics = {
        "base": {
            "name": "Base Model (Untrained)",
            "accuracy": 0.50,
            "trainable_params": 0,
            "total_params": 66955010,
            "description": "Pre-trained DistilBERT without fine-tuning."
        },
        "lora": {
            "name": "LoRA (PEFT)",
            "accuracy": 0.887,
            "trainable_params": 739586,
            "total_params": 67694596,
            "description": "Fine-tuned using Low-Rank Adaptation."
        },
        "prompt": {
            "name": "Prompt-Tuning (PEFT)",
            "accuracy": 0.825,
            "trainable_params": 15360,
            "total_params": 66970370,
            "description": "Fine-tuned by learning 20 virtual tokens."
        },
        "ia3": {
            "name": "IA3 (PEFT)",
            "accuracy": 0.862,
            "trainable_params": 605954,
            "total_params": 67560964,
            "description": "Fine-tuned via scaling internal activations."
        }
    }
    
    advanced_stats = {
        "hyperparam": [
            {"name": "Rank 4", "accuracy": 85.30},
            {"name": "Rank 8", "accuracy": 88.70},
            {"name": "Rank 16", "accuracy": 88.90}
        ],
        "data_efficiency": [
            {"samples": "100", "accuracy": 70.00},
            {"samples": "500", "accuracy": 87.60},
            {"samples": "1000", "accuracy": 88.70}
        ],
        "cross_domain": [
            {"model": "Base", "accuracy": 49.20},
            {"model": "Prompt", "accuracy": 84.40},
            {"model": "LoRA", "accuracy": 86.80},
            {"model": "IA3", "accuracy": 89.60}
        ]
    }
    
    return jsonify({"models": metrics, "advanced_stats": advanced_stats})

@app.route('/predict', methods=['post'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
        
    results = {}
    
    # Base and LoRA can use standard max_length
    # Prompt-Tuning needs adjusted max_length due to virtual tokens
    
    for name, model in models.items():
        max_len = 492 if name == 'prompt' else 512
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).flatten().tolist()
            prediction = torch.argmax(logits, dim=1).item()
            
        results[name] = {
            "prediction": ID2LABEL[prediction],
            "confidence": probs[prediction],
            "probs": probs
        }
        
    return jsonify(results)

if __name__ == '__main__':
    load_models()
    app.run(port=5001, debug=True)
