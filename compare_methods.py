import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from datasets import load_dataset
import numpy as np

# Use CPU for consistent comparison and memory stability
device = "cpu"
model_checkpoint = 'distilbert-base-uncased'

# Paths to the best checkpoints/models
lora_adapter_path = './distilbert-lora-imdb-final/checkpoint-250'
prompt_tuning_adapter_path = './prompt-tuned-imdb'

print("Loading dataset for final comparison...")
imdb_dataset = load_dataset("imdb")
test_indices = [5, 15, 25, 45, 65] # Different from before to ensure fresh results

id2label = {0: "Negative", 1: "Positive"}
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def get_prediction(model, text, is_prompt_tuning=False):
    # Adjust max_length for prompt tuning tokens
    max_len = 492 if is_prompt_tuning else 512
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.argmax(logits, dim=1).item()

# 1. Base Model (Untrained)
print("\nEvaluating Base Model...")
base_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2).to(device)
base_preds = [get_prediction(base_model, imdb_dataset['test'][idx]['text']) for idx in test_indices]

# 2. LoRA Model
print("Evaluating LoRA Model...")
lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path).to(device)
lora_preds = [get_prediction(lora_model, imdb_dataset['test'][idx]['text']) for idx in test_indices]

# 3. Prompt-Tuning Model
# Need a fresh base model instance to avoid adapter conflicts in some PEFT versions
print("Evaluating Prompt-Tuning Model...")
base_model_pt = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2).to(device)
pt_model = PeftModel.from_pretrained(base_model_pt, prompt_tuning_adapter_path).to(device)
pt_preds = [get_prediction(pt_model, imdb_dataset['test'][idx]['text'], is_prompt_tuning=True) for idx in test_indices]

print("\n" + "="*80)
print(f"{'Review Snippet':<50} | {'Truth':<10} | {'Base':<5} | {'LoRA':<5} | {'Prompt':<5}")
print("-" * 80)

for i, idx in enumerate(test_indices):
    text = imdb_dataset['test'][idx]['text'][:47] + "..."
    truth = id2label[imdb_dataset['test'][idx]['label']]
    p_base = "✅" if base_preds[i] == imdb_dataset['test'][idx]['label'] else "❌"
    p_lora = "✅" if lora_preds[i] == imdb_dataset['test'][idx]['label'] else "❌"
    p_pt = "✅" if pt_preds[i] == imdb_dataset['test'][idx]['label'] else "❌"
    
    print(f"{text:<50} | {truth:<10} | {p_base:<5} | {p_lora:<5} | {p_pt:<5}")

print("="*80)
