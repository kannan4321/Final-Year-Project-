from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load your dataset
dataset = load_dataset("csv", data_files={"train": "cleaned_judgments.csv"}, split="train")

# Preprocess
def preprocess(example):
    return tokenizer(example["pet"], truncation=True, padding="max_length", max_length=512)

tokenizer = BertTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
dataset = dataset.map(preprocess)

# Label encoding
label_map = {"J": 0, "judgment": 1}
dataset = dataset.map(lambda x: {"labels": label_map.get(x["Judgement_type"], 0)})

# Model
model = BertForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased", num_labels=2)

# Training args
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=100,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# Train
trainer.train()

# ✅ Save final model and tokenizer
model.save_pretrained("fine_tuned_legalbert")
tokenizer.save_pretrained("fine_tuned_legalbert")
