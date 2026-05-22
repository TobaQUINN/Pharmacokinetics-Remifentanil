import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data/Remifentanil_preprocessed.csv')

# ===============================================
# Visual EDA for Patient Covariates Distributions
# ===============================================

# Age, Weight and Height Distributions

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
sns.histplot(data['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.subplot(1, 3, 2)
sns.histplot(data['Wt'], bins=20, kde=True)
plt.title('Weight Distribution')
plt.subplot(1, 3, 3)
sns.histplot(data['Ht'], bins=20, kde=True)
plt.title('Height Distribution')
plt.tight_layout()
plt.savefig('plots/age_weight_height_distribution.png')
plt.show()


# BMI and LBM Distributions

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(data['BMI'], bins=20, kde=True)
plt.title('BMI Distribution')
plt.subplot(1, 2, 2)
sns.histplot(data['LBM'], bins=20, kde=True)
plt.title('LBM Distribution')
plt.tight_layout()
plt.savefig('plots/bmi_lbm_distribution.png')
plt.show()


# ==========================================================
# PK Temporal Plots
# ==========================================================

# Log Concentration vs TimeDelta


# Infusion rate vs Time & Cumulative Dose vs Time
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.lineplot(x='Time', y='Rate', data=data, ci=None)
plt.title('Infusion Rate vs Time')
plt.subplot(1, 2, 2)
sns.lineplot(x='Time', y='CumulativeDose', data=data, ci=None)
plt.title('Cumulative Dose vs Time')
plt.tight_layout()
plt.savefig('plots/infusion_rate_cumulative_dose.png')
plt.show()


# ===========================================================
# Feature Engineering Validation Plots
# ===========================================================

# Cumulative Dose vs Concentration
plt.figure(figsize=(6, 4))
sns.scatterplot(x='CumulativeDose', y='log_conc', data=data, alpha=0.5)
plt.title('Log Concentration vs Cumulative Dose')
plt.tight_layout()
plt.savefig('plots/log_conc_vs_cumulative_dose.png')
plt.show()
