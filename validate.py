import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from datasets import load_dataset
import os

# Set device to CPU for stability during inference
device = "cpu"

model_checkpoint = 'distilbert-base-uncased'
adapter_path = './distilbert-lora-imdb-final/checkpoint-250' # Use the final checkpoint

print("Loading dataset for evaluation...")
imdb_dataset = load_dataset("imdb")

# Same indices as recorded in the finetune.py turn if possible, 
# but we can just pick a few interesting ones from the test set
test_indices = [0, 10, 20, 50, 100]

id2label = {0: "Negative", 1: "Positive"}

print(f"Loading base model and LoRA adapter from {adapter_path}...")
base_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

print("\n--- Final Evaluation (Fine-tuned Model) ---")
for idx in test_indices:
    text = imdb_dataset['test'][idx]['text']
    label = imdb_dataset['test'][idx]['label']
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    prediction = torch.argmax(logits, dim=1).item()
    
    print(f"\nReview Subset: {text[:150]}...")
    print(f"Ground Truth: {id2label[label]}")
    print(f"Prediction  : {id2label[prediction]}")
    if label == prediction:
        print("Result: ✅ CORRECT")
    else:
        print("Result: ❌ INCORRECT")
