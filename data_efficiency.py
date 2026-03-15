import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import warnings
import os

warnings.filterwarnings('ignore')

model_checkpoint = 'distilbert-base-uncased'
id2label = {0: "Negative", 1: "Positive"}
label2id = {"Negative": 0, "Positive": 1}

print("Loading IMDB dataset for efficiency experiments...")
imdb_dataset = load_dataset("imdb")

# Pre-process evaluation dataset (N=1000) for consistent comparison with existing results
np.random.seed(42)
rand_idx_test = np.random.randint(len(imdb_dataset.get("test")) - 1, size=1000)

val_dataset = Dataset.from_dict({
    'label': [imdb_dataset['test'][int(i)]['label'] for i in rand_idx_test],
    'text': [imdb_dataset['test'][int(i)]['text'] for i in rand_idx_test]
})

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

print("Tokenizing evaluation dataset...")
tokenized_val = val_dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(p):
    predictions, labels = p
    return accuracy_metric.compute(predictions=np.argmax(predictions, axis=1), references=labels)

def train_lora_efficiency(num_samples):
    print(f"\n--- Training LoRA on {num_samples} samples ---")
    np.random.seed(num_samples) # different seed to pick random subset
    rand_idx_train = np.random.randint(len(imdb_dataset.get("train")) - 1, size=num_samples)
    
    train_dataset = Dataset.from_dict({
        'label': [imdb_dataset['train'][int(i)]['label'] for i in rand_idx_train],
        'text': [imdb_dataset['train'][int(i)]['text'] for i in rand_idx_train]
    })
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id
    )
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["q_lin", "k_lin", "v_lin"]
    )
    model = get_peft_model(model, peft_config)
    
    output_dir = f"./distilbert-lora-imdb-{num_samples}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="no",
        save_strategy="no", # Save time for simple experiments
        logging_steps=10
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Accuracy for N={num_samples}: {eval_results['eval_accuracy']*100:.2f}%")
    return eval_results['eval_accuracy']

if __name__ == "__main__":
    acc_100 = train_lora_efficiency(100)
    acc_500 = train_lora_efficiency(500)
    
    print("\n" + "="*50)
    print("--- Data Efficiency Results (LoRA) ---")
    print(f"N=100:  {acc_100*100:.2f}%")
    print(f"N=500:  {acc_500*100:.2f}%")
    print(f"N=1000: 88.70% (From baseline run)")
    print("="*50)
