from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, PromptTuningConfig, PromptTuningInit, TaskType
import evaluate
import torch
import numpy as np
import os

# 1. Load & Process Dataset (Same subset as LoRA for fair comparison)
print("Loading IMDB dataset for Prompt-Tuning...")
imdb_dataset = load_dataset("imdb")

N = 1000 
print(f"Selecting {N} samples from IMDB...")
# Setting a seed for consistency in comparison
np.random.seed(42)
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
    return tokenizer(examples["text"], truncation=True, max_length=492) # 512 - 20 virtual tokens

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 4. Training config with Prompt-Tuning
print("\n--- Setting up Prompt-Tuning ---")
peft_config = PromptTuningConfig(
    task_type=TaskType.SEQ_CLS,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=20,
    prompt_tuning_init_text="Classify if this movie review is positive or negative:",
    tokenizer_name_or_path=model_checkpoint,
    num_layers=6,
    token_dim=768,
    num_attention_heads=12,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 5. Training Pipeline
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./distilbert-prompt-tuning-imdb",
    learning_rate=1e-2, # Prompt tuning often requires higher learning rate than LoRA
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

print("Starting Prompt-Tuning...")
trainer.train()

# 6. Save Model
model.save_pretrained("./prompt-tuned-imdb")
print("Saved to ./prompt-tuned-imdb")
