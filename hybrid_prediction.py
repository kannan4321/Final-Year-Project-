import torch
from transformers import BertTokenizer, BertForSequenceClassification
import xgboost as xgb
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import sys
from utils import extract_structured_features_from_text

# Load BERT model and tokenizer
print("Loading BERT model...")
bert_model = BertForSequenceClassification.from_pretrained("fine_tuned_legalbert")
tokenizer = BertTokenizer.from_pretrained("fine_tuned_legalbert")

# Load XGBoost model and encoders
print("Loading XGBoost model and encoders...")
xgb_model = joblib.load("xgb_model.pkl")
column_transformer = joblib.load("column_transformer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

label_map = {
    0: "Judgment in favor of Petitioner",
    1: "Judgment in favor of Respondent"
}

def predict_from_text(text):
    # BERT Prediction
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        bert_outputs = bert_model(**inputs)
    logits = bert_outputs.logits
    bert_pred = torch.argmax(logits, dim=1).item()
    bert_confidence = torch.softmax(logits, dim=1)[0][bert_pred].item()

    print(f"BERT raw logits: {logits}")
    print(f"BERT predicted class: {bert_pred}")
    print(f"BERT confidence: {bert_confidence:.2f}")
    print(f"BERT interpreted outcome: {label_map[bert_pred]}")

    # Extract structured data from text
    structured_data = extract_structured_features_from_text(text)
    structured_df = pd.DataFrame([structured_data])

    print("Structured input before encoding:")
    print(structured_df)

    # Encode structured data
    structured_encoded = column_transformer.transform(structured_df)
    print(f"Structured encoded shape: {structured_encoded.shape}")

    # XGBoost Prediction
    xgb_pred = xgb_model.predict(structured_encoded)[0]
    xgb_proba = xgb_model.predict_proba(structured_encoded)
    xgb_confidence = np.max(xgb_proba)

    print(f"XGBoost predicted class: {xgb_pred}")
    print(f"XGBoost confidence: {xgb_confidence:.2f}")
    print(f"XGBoost interpreted outcome: {label_map[xgb_pred]}")

    # Final Outcome (can be fusion logic here)
    final_prediction = bert_pred  # Modify for hybrid fusion if needed

    outcome = label_map[final_prediction]
    recommendation = f"Based on the prediction '{outcome}', consider reviewing similar past cases."
    return outcome, recommendation

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hybrid_prediction.py <text_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    with open(input_file, 'r', encoding='utf-8') as f:
        case_text = f.read()

    outcome, recommendation = predict_from_text(case_text)
    print("\nPrediction Outcome:", outcome)
    print("Recommendation:", recommendation)
