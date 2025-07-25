import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv('cleaned_judgments.csv')

# Features and target
X = df.drop(columns=['Judgement_type'])
y = df['Judgement_type']

# Categorical columns to encode
categorical_columns = ['case_no', 'pet', 'res', 'pet_adv', 'res_adv', 'bench', 'judgement_by', 'judgment_dates']

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Column transformer for encoding categorical columns
column_transformer = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
], remainder='drop')

# Build pipeline with column transformer and XGBoost classifier
pipeline = Pipeline([
    ('transform', column_transformer),
    ('xgb', XGBClassifier(eval_metric='mlogloss'))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Fit model
pipeline.fit(X_train, y_train)

# Save the pipeline and label encoder
joblib.dump(pipeline, 'xgb_model_pipeline.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("\u2705 XGBoost model trained and saved successfully.")
