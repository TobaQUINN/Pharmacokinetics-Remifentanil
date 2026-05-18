import pandas as pd
import numpy as np

# XGBoost regression model
from xgboost import XGBRegressor

# Evaluation metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Plotting
import matplotlib.pyplot as plt


# Load preprocessed data
data = pd.read_csv('data/Remifentanil_preprocessed.csv')

print("="*80)
print("XGBOOST BASELINE MODEL FOR REMIFENTANIL PK PREDICTION")
print("="*80)

# Define features and target

features = [
    # Time features
    'Time',
    'TimeDelta',

    # Dosing features
    'Rate',
    'Amt',
    'CumulativeDose',
    'InfusionActive',

    # Patient covariates
    'Age',
    'Sex_Encoded',
    'Ht',
    'Wt',
    'BSA',
    'LBM',
    'BMI',
]

target = 'Log_conc'


# Splitting data into training and testing sets based on unique patient IDs

patient_ids = data['ID'].unique()

# Shuffle patients
np.random.seed(42)
np.random.shuffle(patient_ids)

# 80% train patients
train_size = int(0.8 * len(patient_ids))

train_ids = patient_ids[:train_size]
test_ids = patient_ids[train_size:]

# Create train and test dataframes
train_df = data[data['ID'].isin(train_ids)]
test_df = data[data['ID'].isin(test_ids)]

# Extract feature matrices
X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

# Train XGBoost regressor
xgboost_model = XGBRegressor(

    # Number of boosting trees
    n_estimators=300,

    # Tree depth
    #
    # Higher depth:
    # more complex learning
    # but higher overfitting risk
    max_depth=6,

    # Step size for learning
    #
    # Smaller learning rate:
    # slower but more stable learning
    learning_rate=0.05,

    # Fraction of rows sampled per tree
    #
    # Helps reduce overfitting
    subsample=0.8,

    # Fraction of features sampled per tree
    colsample_bytree=0.8,

    # Objective function
    #
    # Since this is regression:
    # predict continuous concentrations
    objective='reg:squarederror',

    # Random seed for reproducibility
    random_state=42
)

# Train Model

xgboost_model.fit(X_train, y_train)

y_pred = xgboost_model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"XGBoost Model MAE: {mae:.4f}")
print(f"XGBoost Model MSE: {mse:.4f}")
print(f"XGBoost Model R²: {r2:.4f}")


# Checking Feature Importance

importance = xgboost_model.feature_importances_

# Create dataframe
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
})

# Sort descending
importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========\n")
print(importance_df)


# ============================================
# PLOT FEATURE IMPORTANCE
# ============================================

plt.figure(figsize=(10, 8))

plt.barh(
    importance_df['Feature'],
    importance_df['Importance']
)

plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.title(
    "XGBoost Feature Importance\n"
    "Remifentanil PK Prediction"
)

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
plt.savefig('plots/xgboost_feature_importance.png')
