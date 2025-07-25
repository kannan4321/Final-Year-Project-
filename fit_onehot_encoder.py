import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load your dataset
df = pd.read_csv("cleaned_judgments.csv")

# Define the exact columns to use, in order
features = ['diary_no', 'Judgement_type', 'case_no', 'pet', 'res',
            'pet_adv', 'res_adv', 'bench', 'judgement_by', 'judgment_dates']

df = df[features].dropna()

# Initialize and fit OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(df)

# Save encoder
joblib.dump(encoder, "onehot_encoder.pkl")
print("OneHotEncoder saved successfully.")
