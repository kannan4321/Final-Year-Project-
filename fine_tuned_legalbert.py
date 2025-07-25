import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load your cleaned dataset (assuming text + label columns)
df = pd.read_csv("cleaned_judgments.csv")

# Use only rows with valid labels
valid_labels = ["J", "judgment"]
df = df[df['Judgement_type'].isin(valid_labels)].copy()

# Encode labels as 0 and 1
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Judgement_type'])  # J -> 0, judgment -> 1

# Combine some features as input text (adjust as needed)
df['text'] = df['pet'].fillna('') + " vs " + df['res'].fillna('') + ". " + df['case_no'].fillna('')

# Split dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.1, random_state=42
)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

# Tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Load model
model = BertForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased", num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
)

# Metrics
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1": f1_score(p.label_ids, preds)
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Save model and tokenizer
model.save_pretrained("fine_tuned_legalbert")
tokenizer.save_pretrained("fine_tuned_legalbert")

print("✅ Model training and saving complete.")
