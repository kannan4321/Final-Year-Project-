import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Step 1: Load Data
print("Loading data...")
data = pd.read_csv('cleaned_judgments.csv')

# Step 2: Define Features and Target
features = ['bench', 'judgement_by']  # reduced high-cardinality columns
target = 'Judgement_type'

# Step 3: Encode Target Labels
print("Encoding features...")
label_encoder = LabelEncoder()
data[target] = label_encoder.fit_transform(data[target])

# Step 4: Train-Test Split
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: OneHotEncoder with memory optimization
try:
    onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
except TypeError:
    onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=True)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', onehot_encoder, features)
    ]
)

# Step 6: Build Pipeline
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', xgb_model)])

# Step 7: Train the Model
print("Training new XGBoost model...")
pipeline.fit(X_train, y_train)

# Step 8: Save Model and Encoders
joblib.dump(pipeline, 'xgboost_model.joblib')
joblib.dump(onehot_encoder, 'onehot_encoder.pkl') 
joblib.dump(label_encoder, 'label_encoder.pkl')
print("Model and encoders saved successfully!")

# Step 9: Evaluate Model
print("Evaluating model...")
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on structured data: {accuracy:.4f}")
