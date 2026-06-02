# Remifentanil Pharmacokinetic Prediction System

## Predicting Remifentanil Plasma Concentration Using Machine Learning, Pharmacokinetic Feature Engineering, and Stateful Inference

### Project Overview

This project explores the use of Machine Learning to model and predict plasma concentrations of Remifentanil, an ultra-short-acting synthetic opioid commonly used during anesthesia and surgical procedures.

The goal was not simply to train a predictive model, but to understand how pharmacokinetic principles can be integrated into a machine learning workflow to create a system capable of making clinically meaningful concentration predictions over time.

The project combines:

* Pharmacokinetics (PK)
* Pharmacodynamics (PD)
* Feature Engineering
* Machine Learning
* Stateful Inference
* Interactive Deployment
* Containerization

---

# Why Remifentanil?

Remifentanil is a potent μ-opioid receptor agonist used for analgesia and anesthesia.

Unlike opioids such as Morphine and Fentanyl, Remifentanil possesses a unique characteristic:

**It is rapidly metabolized by non-specific blood and tissue esterases.**

This results in:

* Extremely rapid onset
* Extremely rapid offset
* Short context-sensitive half-time
* Predictable recovery profile

### Comparison with Other Opioids

| Drug         | Main Metabolism          | Duration     |
| ------------ | ------------------------ | ------------ |
| Morphine     | Hepatic metabolism       | Long         |
| Fentanyl     | Hepatic metabolism       | Intermediate |
| Remifentanil | Blood & tissue esterases | Ultra-short  |

Because of this unique pharmacokinetic profile, Remifentanil is frequently used in:

* General anesthesia
* Intensive care sedation
* Target Controlled Infusion (TCI) systems
* Procedures requiring rapid titration of analgesic effect

Its fast-changing concentration profile also makes it an interesting machine learning challenge.

---

# Dataset

The dataset used in this project was obtained from Kaggle:

https://www.kaggle.com/datasets/andrewsas26/pharmacokinetics-of-remifentanil

The dataset contains longitudinal pharmacokinetic observations from multiple patients receiving Remifentanil infusions.

Each patient contains multiple sequential observations over time, creating a patient-specific time series rather than independent observations.

### Dataset Characteristics

* Longitudinal data
* Repeated measurements per patient
* Multiple infusion phases
* Plasma concentration measurements
* Patient demographic covariates

Examples of covariates include:

* Age
* Sex
* Weight
* Height
* Lean Body Mass (LBM)
* Body Surface Area(BSA)
* Dose amount

---

# Understanding the Data

Before any modeling was performed, significant effort was spent understanding:

* What Remifentanil is
* How it behaves pharmacokinetically
* How infusion rate affects concentration
* How elimination occurs
* How concentration changes over time

One of the earliest observations was that the dataset was not a traditional tabular dataset.

Instead, it represented a collection of patient trajectories.

Each patient contained a sequence of concentration measurements that evolved over time.

This immediately raised an important question:

> Can a model learn concentration dynamics using only the raw features provided?

---

# Inspiration from Target Controlled Infusion (TCI)

While studying Remifentanil pharmacokinetics, I discovered Target Controlled Infusion (TCI) systems.

TCI systems continuously estimate drug concentrations and adjust infusion rates to achieve desired targets.

Although this project does not attempt to replace a clinical TCI system, it inspired a similar design philosophy:

* Track patient state over time
* Account for previous dosing history
* Use temporal information
* Predict concentration dynamically

This significantly influenced the engineering decisions made throughout the project.

---

# Feature Engineering

One of the most important stages of the project involved introducing pharmacokinetic knowledge into the machine learning pipeline.

Rather than relying solely on the raw dataset features, several engineered features were introduced.

## 1. Cumulative Dose

Represents the total amount of Remifentanil administered up to a given point.

Purpose:

* Captures drug accumulation
* Represents total exposure
* Provides historical context

---

## 2. Time Delta

Represents the time elapsed since the previous observation.

Purpose:

* Encodes temporal spacing
* Helps model elimination dynamics
* Improves understanding of concentration changes

---

## 3. Infusion Active

Binary indicator:

* 1 = infusion running
* 0 = infusion stopped

Purpose:

* Distinguishes input phase from elimination phase
* Allows model to learn different pharmacokinetic regimes

---

## 4. BMI

Calculated from:

BMI = Weight / Height²

Purpose:

* Captures body composition effects
* Provides additional patient-specific information

---

# Why Not Lag Features?

Lag features were considered during development.

Examples:

* Previous concentration
* Previous dose

However, they were ultimately excluded.

Reasons included:

* Risk of information leakage
* Reduced generalization
* Increased deployment complexity
* Dependence on concentration values unavailable in real-world inference scenarios

Instead, the project focused on features that could be derived from available patient information and infusion history.

---

# Data Preprocessing

The preprocessing pipeline included:

* Missing value handling
* Patient-wise sorting
* Temporal ordering
* Feature engineering
* Covariate encoding
* Feature validation

Particular attention was given to preserving patient-specific trajectories.

---

# Model Selection

Several factors influenced the decision to use XGBoost.

## Why XGBoost?

XGBoost provides:

* Strong performance on tabular datasets
* Ability to capture nonlinear relationships
* Robust handling of mixed feature types
* Built-in regularization
* Feature importance analysis

Most importantly, it performs exceptionally well when domain knowledge is incorporated through engineered features.

This made it a strong candidate for pharmacokinetic modeling.

---

# Model Performance

The final model achieved:

* MAE ≈ 0.23
* MSE ≈ 0.09
* R² ≈ 0.97

Additional validation included:

* Predicted vs Actual analysis
* Residual analysis
* Patient trajectory visualization
* Out-of-distribution testing

---

# Deployment

The project was deployed using Streamlit.

The application provides:

* Interactive patient inputs
* Stateful patient tracking
* Dynamic feature engineering
* Real-time concentration prediction
* Concentration trajectory visualization

Users can simulate patient infusion sessions and observe how concentration predictions evolve over time.

---

# Containerization

To ensure reproducibility and portability, the application was containerized using Docker.

Benefits:

* Environment consistency
* Dependency isolation
* Simplified deployment
* Reproducible execution


---

# Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Streamlit:

```bash
streamlit run streamlit_app.py
```

---

# Running with Docker

Build image:

```bash
docker build -t remifentanil-pk .
```

Run container:

```bash
docker run -p 8501:8501 remifentanil-pk
```

Open:

```text
http://localhost:8501
```

---

# Lessons Learned

This project evolved far beyond model training.

It became an exploration of:

* Pharmacokinetics
* Biomedical data science
* Feature engineering
* Machine learning systems
* Deployment
* Containerization
* Software engineering

One of the biggest lessons learned was that building a model is only the beginning.

Transforming that model into a usable system requires a completely different set of skills involving architecture, deployment, debugging, reproducibility, and user experience.

---

# Future Work

This is my first deployed non-generic ML application, potential future improvements will be included when i come up with ideas to improve this project further.


---

# Disclaimer

This project is intended for educational and research purposes only.

It is not a medical device and should not be used for clinical decision-making.
