from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, PrefixTuningConfig, TaskType, IA3Config
import evaluate
import torch
import numpy as np

print("Loading IMDB dataset for 3rd PEFT Method...")
imdb_dataset = load_dataset("imdb")

N = 1000 
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

model_checkpoint = 'distilbert-base-uncased'
id2label = {0: "Negative", 1: "Positive"}
label2id = {"Negative": 0, "Positive": 1}

print("Loading base model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id
)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Let's try IA3 as our third method. It scales inner activations and is extremely fast.
# PrefixTuning on sequence classification for encoder-only models can be fragile in PEFT.
print("\n--- Setting up IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations) ---")
peft_config = IA3Config(
    task_type=TaskType.SEQ_CLS,
    target_modules=["k_lin", "v_lin", "ffn.lin1"], # standard for DistilBERT
    feedforward_modules=["ffn.lin1"],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="./distilbert-ia3-imdb",
    learning_rate=3e-3, # IA3 needs a relatively higher learning rate
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

print("Starting IA3 Fine-Tuning...")
trainer.train()

model.save_pretrained("./ia3-tuned-imdb")
print("Saved to ./ia3-tuned-imdb")
