import pandas as pd
import numpy as np

# DATA CLEANING
print("=" * 40)
print("DATA CLEANING")
print("=" * 40)

# Importing data into a dataframe
df = pd.read_csv('data/Remifentanil.csv')

# Checking data columns
print("Columns in Remifentanil dataset: ",
      df.columns)

# Checking data types of each column
print("Data types of each column: ",
      df.dtypes)

# Checking first few rows of the dataset
print("First 5 rows of the dataset: ",
      df.head())
# Verify ID and Subject are Identical
assert (df['ID'] == df['Subject']).all(
), "ID and Subject columns do not match!"
df = df.drop(columns=['Subject'])  # Drop redundant 'Subject' column
print("Dropped redundant 'Subject' column (identical to 'ID')]\n")

# Checking for missing values
print("Missing values in each column: ",
      df.isnull().sum())
# Checking null concentration pattern more carefully
null_conc = df[df['conc'].isnull()]
print("Null Concentration time values: ",
      null_conc['Time'].value_counts())
# Handling missing values
# Droping missing values
df = df.dropna()
# Reason: We have 115 missing Concentration values, but they fall into 3 clinical patterns:
# 1. Baseline(t=0): Drug hasn't been administered yet, so undefined concentration
# 2. Post-infusion: Monitoring stopped, so no more concentration measurements
# 3. Mid Infusion: True missingness, sample error
# We can safely drop rows with missing concentration values without losing critical information, as they do not represent random missingness but rather specific clinical scenarios.

# Checking for duplicates
print("Number of duplicate rows: ",
      df.duplicated().sum())

# Rows after dropping missing values
print(f"Rows after dropping missing target:, {df.shape[0]}")
print(f"Percentage of data retained: {100 * df.shape[0] / 2107:.1f}%")
print(f"Missing values after:  {df.isnull().sum()}")

# Checking for negative or Invalid values in key columns
print("DATA VALIDITY CHECKS")
print("Negative concentration values present: ",
      (df['conc'] < 0).values.any())
print("Invalid weight values present: ",
      (df['Wt'] <= 0).values.any())
print("Invalid age values present: ",
      (df['Age'] <= 0).values.any())
print("Invalid Amt values present: ",
      (df['Amt'] < 0).values.any())
print("Invalid Rate values present: ",
      (df['Rate'] < 0).values.any())
print("Invalid Time values present: ",
      (df['Time'] < 0).values.any())
# Catching data quality issuesearly to prevent mysterious model failures

# Saving cleaned data
df.to_csv('data/Remifentanil_cleaned.csv', index=False)


# FEATURE ENGINEERING
print("=" * 40)
print("FEATURE ENGINEERING")
print("=" * 40)
# Importing cleaned data
cleaned_df = pd.read_csv('data/Remifentanil_cleaned.csv')

# Sorting data by patient ID and time
sorted_df = cleaned_df.sort_values(['ID', 'Time']).reset_index(drop=True)
print("Data sorted by ID and Time.")

# Cumulative Dose calculation
cleaned_df['CumulativeDose'] = cleaned_df.groupby('ID').apply(
    lambda group: np.cumsum(
        group['Rate'].values *
        np.diff(np.concatenate([[0], group['Time'].values]))
    )
).explode().values

print(
    f"Cumulative dose range: {cleaned_df['CumulativeDose'].min():.2f} to {cleaned_df['CumulativeDose'].max():.2f}\n")

# Reason: Total drug exposure matters more than instantaneous rate for steady-state
# concentration. This captures the pharmacokinetic principle that concentration
# is determined by both dose rate and time (AUC = Area Under Curve).

# Time since Infusion Start
print("Creating time-based features...")

# Time since start of Infusion, renaming the 'Time' column to 'TimeSinceStart' for clarity
df.rename(columns={'Time': 'TimeSinceStart'}, inplace=True)

# Time delta between consecutive observations (per patient)
cleaned_df['TimeDelta'] = cleaned_df.groupby('ID')['Time'].diff().fillna(0)

print(
    f"Time delta range: {cleaned_df['TimeDelta'].min():.2f} to {cleaned_df['TimeDelta'].max():.2f} minutes\n")

# Reason: PK models are fundamentally time-dependent. Distribution and elimination
# phases occur over different time scales. Time delta helps the model understand
# measurement frequency changes (sparse early, dense during infusion).

# Infusion Status (binary feature indicating if infusion is active)
cleaned_df['InfusionActive'] = (cleaned_df['Rate'] > 0).astype(int)

print(f"Infusion active: {cleaned_df['InfusionActive'].sum()} observations")
print(
    f"Infusion stopped: {(cleaned_df['InfusionActive'] == 0).sum()} observations\n")

# Reason: Concentration dynamics differ fundamentally during infusion (input + elimination)
# vs post-infusion (elimination only). This binary feature helps the model
# distinguish these pharmacokinetic phases

# Body Composition Features
print("Body composition features already present:")
print("  - BSA (Body Surface Area): DuBois formula")
print("  - LBM (Lean Body Mass): Affects volume of distribution")
print()

# Reason: BSA and LBM are standard PK covariates. Remifentanil dosing is often
# weight-adjusted because volume of distribution scales with lean body mass.
# These are already correctly calculated in the dataset.

# Body Mass Index (BMI)
cleaned_df['BMI'] = cleaned_df['Wt'] / (cleaned_df['Ht'] / 100) ** 2

print(
    f"BMI created - range: {cleaned_df['BMI'].min():.1f} to {cleaned_df['BMI'].max():.1f}\n")

# Reason: BMI captures the relationship between height and weight differently than
# BSA or LBM. High BMI patients may have altered clearance due to lipid accumulation.

# Saving engineered features
cleaned_df.to_csv('data/Remifentanil_engineered.csv', index=False)

# Target Transformation
print("="*40)
print("TARGET TRANSFORMATION")
print("="*40)

df_eng = pd.read_csv('data/Remifentanil_engineered.csv')

# Checking distribution of target variable
print("Raw concentration distribution:")
print(df_eng['conc'].describe())
print(f"Skewness: {df_eng['conc'].skew():.2f}\n")

# Applying log transformation to target variable
df_eng['Log_conc'] = np.log(df_eng['conc'])
print("Log-transformed concentration distribution:")
print(df_eng['Log_conc'].describe())
print(f"Skewness: {df_eng['Log_conc'].skew():.2f}\n")
# Reason: Concentration values are often right-skewed due to the nature of drug kinetics.
# Log transformation helps stabilize variance and make the distribution more normal,
# which can improve model performance and convergence. Adding 1 prevents issues with log(0).

# Categorical Encoding
print("=" * 40)
print("CATEGORICAL ENCODING")
print("=" * 40)

# Encoding 'Sex', the only categorical variable
print('Sex Distribution:')
print(df_eng.groupby('ID')['Sex'].first().value_counts())

# Encoding: Basic Mapping; Male= 1, Female = 0
df_eng['Sex_Encoded'] = df_eng['Sex'].map({'Male': 1, 'Female': 0})
print('Encoded Sex Distribution: Male =1, Female = 0')

# Saving Encoded data
df_eng.to_csv('data/Remifentanil_encoded.csv', index=False)

# Removind original un encoded 'Sex' column
df_encoded = pd.read_csv('data/Remifentanil_encoded.csv')
df_encoded = df_encoded.drop(columns=['Sex'])

df_encoded.to_csv('data/Remifentanil_preprocessed.csv', index=False)


# Feature Selection
print("=" * 40)
print("FEATURE SELECTION")
print("=" * 40)

# Dropping columns that would not be used as features

columns_to_drop = [
    'rownames',        # Just an index
    'Subject',         # Already dropped (duplicate of ID)
    'conc',            # Original target (log_conc instead)
    'Sex',             # Replaced with Sex_Encoded
]

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

target_column = 'log_conc'
group_column = 'ID'

print(f"Features ({len(feature_columns)}):")
for feat in feature_columns:
    print(f"  - {feat}")
print(f"\nTarget: {target_column}")
print(f"Grouping variable: {group_column}\n")

# Reason: Explicit feature selection prevents accidentally using leakage variables
# (like the original target 'conc' or redundant identifiers).
