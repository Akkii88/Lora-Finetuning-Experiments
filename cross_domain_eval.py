import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from datasets import load_dataset
import numpy as np
import warnings

# Suppress warnings to keep output clean
warnings.filterwarnings('ignore')

device = "cpu"
model_checkpoint = 'distilbert-base-uncased'

print("Loading Yelp Polarity dataset for Cross-Domain Generalization...")
# yelp_polarity uses 0: Negative, 1: Positive, identical to our IMDB mapping
yelp = load_dataset("yelp_polarity")

# We evaluate on a sample of 250 reviews to ensure fast CPU testing
np.random.seed(42)
test_idx = np.random.randint(len(yelp['test']), size=250)
eval_dataset = yelp['test'].select(test_idx)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

def evaluate_model(model_name, model, is_prompt_tuning=False):
    max_len = 492 if is_prompt_tuning else 512
    
    predictions = []
    references = eval_dataset['label']
    
    print(f"Evaluating {model_name} on Yelp Distribution (Zero-Shot)...", end="", flush=True)
    model.eval()
    with torch.no_grad():
        for i in range(len(eval_dataset)):
            if i % 50 == 0:
                print(".", end="", flush=True)
            text = eval_dataset[i]['text']
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=1).item()
            predictions.append(pred)
            
    acc = np.mean(np.array(predictions) == np.array(references))
    print(f"\n{model_name} Accuracy on Yelp: {acc*100:.2f}%\n")
    return acc

print("Loading Base Model...")
base_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2).to(device)
evaluate_model("Base Model (Untrained)", base_model)

print("Loading LoRA Model...")
lora_model = PeftModel.from_pretrained(base_model, './distilbert-lora-imdb-final/checkpoint-250').to(device)
evaluate_model("LoRA (IMDB trained)", lora_model)

print("Loading Prompt-Tuning Model...")
pt_base = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2).to(device)
pt_model = PeftModel.from_pretrained(pt_base, './prompt-tuned-imdb').to(device)
evaluate_model("Prompt-Tuning (IMDB trained)", pt_model, is_prompt_tuning=True)

try:
    print("Loading IA3 Model...")
    ia3_base = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2).to(device)
    ia3_model = PeftModel.from_pretrained(ia3_base, './ia3-tuned-imdb').to(device)
    evaluate_model("IA3 (IMDB trained)", ia3_model)
except Exception as e:
    print(f"IA3 model unavailable or incomplete: {e}")

print("Cross-Domain Evaluation Complete.")
