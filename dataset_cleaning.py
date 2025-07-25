import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv("judgments.csv")

# Step 1: Drop Irrelevant Columns
columns_to_drop = ["temp_link", "language"]
data = data.drop(columns=columns_to_drop)

# Step 2: Handle Missing Values
data = data[~(data['pet'].isna() & data['res'].isna())]

# Fill missing advocate names with 'Unknown'
data.loc[:, 'pet_adv'] = data['pet_adv'].fillna('Unknown')
data.loc[:, 'res_adv'] = data['res_adv'].fillna('Unknown')

# Replace missing bench and judgement_by with 'Not Specified'
data.loc[:, 'bench'] = data['bench'].fillna('Not Specified')
data.loc[:, 'judgement_by'] = data['judgement_by'].fillna('Not Specified')

# Step 3: Convert Dates to Datetime Format
data['judgment_dates'] = pd.to_datetime(data['judgment_dates'], errors='coerce', dayfirst=True)

# Step 4: Text Normalization
def clean_text(text):
    if pd.isna(text):
        return "Unknown"
    text = str(text).strip().upper()
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = " ".join(text.split())  # Remove extra whitespace
    return text

text_columns = ['pet', 'res', 'pet_adv', 'res_adv', 'bench', 'judgement_by']
for col in text_columns:
    data.loc[:, col] = data[col].apply(clean_text)

# Step 5: Remove Duplicate Records
data = data.drop_duplicates()

# Step 6: Save Cleaned Data
cleaned_file_path = "cleaned_judgments.csv"
data.to_csv(cleaned_file_path, index=False)

print(f"Data cleaned and saved to: {cleaned_file_path}")
