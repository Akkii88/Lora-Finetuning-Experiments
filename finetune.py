from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np
import os

# 1. Load & Process Dataset
print("Loading IMDB dataset...")
imdb_dataset = load_dataset("imdb")

# Using a subset for faster training but keeping it IMDB
N = 1000 
print(f"Selecting {N} samples from IMDB...")
rand_idx_train = np.random.randint(len(imdb_dataset.get("train")) - 1, size=N)
rand_idx_test = np.random.randint(len(imdb_dataset.get("test")) - 1, size=N)

dataset = DatasetDict({
    'train': Dataset.from_dict({
        'label': [imdb_dataset['train'][int(i)]['label'] for i in rand_idx_train],
        'text': [imdb_dataset['train'][int(i)]['text'] for i in rand_idx_train]
    }),
    'validation': Dataset.from_dict({
        'label': [imdb_dataset['test'][int(i)]['label'] for i in rand_idx_test],
        'text': [imdb_dataset['test'][int(i)]['text'] for i in rand_idx_test]
    })
})

# 2. Load model
model_checkpoint = 'distilbert-base-uncased'
id2label = {0: "Negative", 1: "Positive"}
label2id = {"Negative": 0, "Positive": 1}

print(f"Loading base model: {model_checkpoint}")
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, 
    num_labels=2, 
    id2label=id2label, 
    label2id=label2id
)

# 3. Process data
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):
    tokenizer.truncation_side = "left"
    return tokenizer(examples["text"], truncation=True, max_length=512)

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 4. BEFORE Evaluation
print("\n--- BEFORE Fine-tuning ---")
model.eval()
accuracy_metric = evaluate.load("accuracy")

def run_sample_eval(model, tokenizer, text, label):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    prediction = torch.argmax(logits, dim=1).item()
    print(f"Text: {text[:100]}...")
    print(f"Truth: {id2label[label]} | Predicted: {id2label[prediction]}")

# Sample test on first few items
for i in range(3):
    run_sample_eval(model, tokenizer, dataset['validation'][i]['text'], dataset['validation'][i]['label'])

# 5. Training config with LoRA
print("\n--- Setting up LoRA ---")
peft_config = LoraConfig(
    task_type="SEQ_CLS",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=['q_lin', 'v_lin']
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 6. Training Pipeline
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./distilbert-lora-imdb-final",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting Fine-tuning...")
trainer.train()

# 7. AFTER Evaluation
print("\n--- AFTER Fine-tuning ---")
model.eval()
for i in range(3):
    run_sample_eval(model, tokenizer, dataset['validation'][i]['text'], dataset['validation'][i]['label'])

# Save
model.save_pretrained("./lora-finetuned-imdb")
print("Saved to ./lora-finetuned-imdb")
