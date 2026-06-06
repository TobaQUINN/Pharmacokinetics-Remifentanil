import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import os

# Load data
data = pd.read_csv("data/Remifentanil_preprocessed.csv")

print("="*80)
print("XGBOOST BASELINE MODEL FOR REMIFENTANIL PK PREDICTION")
print("="*80)

# Feature set (must match pipeline definition)
features = [
    'Time', 'TimeDelta',
    'Rate', 'Amt', 'CumulativeDose', 'InfusionActive',
    'Age', 'Sex_Encoded', 'Ht', 'Wt', 'BSA', 'LBM', 'BMI'
]

target = "Log_conc"

# Patient-level split
patient_ids = data["ID"].unique()

np.random.seed(42)
np.random.shuffle(patient_ids)

train_size = int(0.8 * len(patient_ids))
train_ids = patient_ids[:train_size]
test_ids = patient_ids[train_size:]

train_df = data[data["ID"].isin(train_ids)]
test_df = data[data["ID"].isin(test_ids)]

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

# Model
model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# Save model (cleaner path)
os.makedirs("models", exist_ok=True)
model.save_model("models/xgboost_model.json")

print("Model saved to models/xgboost_model.json")
