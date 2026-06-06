import pandas as pd
import numpy as np

# DATA CLEANING AND VALIDATION


def clean_remifentanil_data(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure ID consistency
    if not (df["ID"] == df["Subject"]).all():
        raise ValueError("ID and Subject columns do not match")

    df = df.drop(columns=["Subject"])

    # Drop missing target values
    df = df.dropna(subset=["conc"])

    # Basic validity filters
    df = df[
        (df["conc"] >= 0) &
        (df["Wt"] > 0) &
        (df["Age"] > 0) &
        (df["Amt"] >= 0) &
        (df["Rate"] >= 0) &
        (df["Time"] >= 0)
    ]

    df.to_csv("data/Remifentanil_cleaned.csv", index=False)
    return df


# FEATURE ENGINEERING
def feature_engineering(input_path, output_path):
    # Load cleaned data
    df = pd.read_csv(input_path)

    print("Feature Engineering Started")
    print("Initial shape:", df.shape)

    # Sorting before engineering cumulative value
    df = df.sort_values(["ID", "Time"]).reset_index(drop=True)

    # Cumulative Dose
    df["CumulativeDose"] = df.groupby("ID")["Amt"].cumsum()

    # Time delta per patient
    df["TimeDelta"] = df.groupby("ID")["Time"].diff().fillna(0)

    # Infusion status
    df["InfusionActive"] = (df["Rate"] > 0).astype(int)

    # BMI
    df["BMI"] = df["Wt"] / (df["Ht"] / 100) ** 2

    print("Feature engineering complete")
    print("Final shape:", df.shape)

    # Save engineered data
    df.to_csv(output_path, index=False)

    print(f"Saved engineered data to: {output_path}")

    return df

# FEATURE ENCODING AND TARGET TRANSFORMATION


def transform_target_and_encode(input_path, output_path):
    df = pd.read_csv(input_path)

    # Target distribution
    print("Raw concentration distribution:")
    print(df['conc'].describe())
    print(f"Skewness: {df['conc'].skew():.2f}\n")

    # Safe log transform (critical fix)
    df['Log_conc'] = np.log1p(df['conc'])  # log(1 + x)

    print("Log-transformed concentration distribution:")
    print(df['Log_conc'].describe())
    print(f"Skewness: {df['Log_conc'].skew():.2f}\n")

    print("=" * 40)
    print("CATEGORICAL ENCODING")
    print("=" * 40)

    # Sex encoding
    print('Sex Distribution:')
    print(df.groupby('ID')['Sex'].first().value_counts())

    df['Sex_Encoded'] = df['Sex'].map({'Male': 1, 'Female': 0})

    print('Encoded Sex Distribution: Male=1, Female=0')

    # Drop unnecessary columns BEFORE saving final dataset
    cols_to_drop = [c for c in ['Sex', 'rownames'] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    df.to_csv(output_path, index=False)

    print(f"\nSaved preprocessed data to: {output_path}")

    return df


# FEATURE SELECTION
def select_features(input_path):
    df = pd.read_csv(input_path)

    print("=" * 40)
    print("FEATURE SELECTION")
    print("=" * 40)

    # Drop target + leakage columns
    cols_to_drop = [c for c in ['conc', 'Sex',
                                'rownames', 'Subject'] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Define feature set (truth source for training)
    feature_columns = [
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

    # Keep only available columns (safety guard)
    feature_columns = [c for c in feature_columns if c in df.columns]

    X = df[feature_columns]

    print(f"Selected {len(feature_columns)} features")
    print("Features:", feature_columns)

    return X, df


# Reason: Explicit feature selection prevents accidentally using leakage variables
# (like the original target 'conc' ).
