import streamlit as st
import pandas as pd
import numpy as np

from app.feature_engineering import engineer_input_features, compute_bmi
from app.state_manager import (
    initialize_patient,
    get_patient_state,
    update_patient_state,
    reset_patient
)


import xgboost as xgb

model = xgb.Booster()
model.load_model("app/models/xgboost_model.json")


# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(page_title="ML in Pharmacokinetics", layout="wide")

st.title("🧬 Machine Learning in Pharmacokinetics: Using XGBoost Algorithm to predict "
         "patient Remifentanil Plasma Concentration with Real Longitudinal Time Series Clinical Data")


# -----------------------------
# SESSION INIT
# -----------------------------
if "initialized" not in st.session_state:
    st.session_state.initialized = False

if "patient_id" not in st.session_state:
    st.session_state.patient_id = None


# -----------------------------
# PATIENT INITIALIZATION PANEL
# -----------------------------
st.sidebar.header("Patient Initialization")

with st.sidebar.form("init_form"):
    patient_id = st.text_input("Patient ID")
    time = st.number_input("Initial Time", value=0.0)

    age = st.number_input("Age", value=30)
    sex = st.selectbox("Sex", ["Male", "Female"])

    wt = st.number_input("Weight (kg)", value=70.0)
    ht = st.number_input("Height (cm)", value=170.0)

    bsa = st.number_input("Body Surface Area", value=1.8)
    lbm = st.number_input("Lean Body Mass", value=50.0)

    submit = st.form_submit_button("Initialize Patient")


if submit:
    st.session_state.patient_id = patient_id
    st.session_state.initialized = True

    initialize_patient(patient_id)

    st.session_state.raw_state = {
        "Time": time,
        "Age": age,
        "Wt": wt,
        "Ht": ht,
        "BSA": bsa,
        "LBM": lbm,
        "Sex": sex
    }

    st.success(f"Patient {patient_id} initialized")


# -----------------------------
# LIVE SIMULATION PANEL
# -----------------------------
if st.session_state.initialized:

    st.subheader(f"Live Simulation: Patient {st.session_state.patient_id}")

    col1, col2 = st.columns(2)

    with col1:
        time = st.number_input(
            "Time (current step)", value=0.0, key="time_live")
        rate = st.number_input("Rate", value=0.0, key="rate_live")
        amt = st.number_input("Amt", value=0.0, key="amt_live")

    with col2:
        run_step = st.button("Run Prediction Step")
        reset = st.button("Reset Patient")

    if reset:
        reset_patient(st.session_state.patient_id)
        st.session_state.initialized = False
        st.session_state.patient_id = None
        st.session_state.raw_state = {}
        st.rerun()

    if run_step:

        patient_id = st.session_state.patient_id
        state = get_patient_state(patient_id)

        raw_input = {
            "Time": time,
            "Rate": rate,
            "Amt": amt,
            "Age": st.session_state.raw_state["Age"],
            "Sex": st.session_state.raw_state["Sex"],
            "Wt": st.session_state.raw_state["Wt"],
            "Ht": st.session_state.raw_state["Ht"],
            "BSA": st.session_state.raw_state["BSA"],
            "LBM": st.session_state.raw_state["LBM"]
        }

        features = engineer_input_features(state, raw_input)

        dmatrix = xgb.DMatrix(features)
        log_conc_pred = model.predict(dmatrix)[0]
        conc_pred = np.exp(log_conc_pred)

        update_patient_state(
            patient_id,
            time=time,
            amt=amt,
            prediction=log_conc_pred,
            concentration=conc_pred
        )

        st.metric("Predicted Log Concentration", f"{log_conc_pred:.4f}")
        st.metric("Plasma Remifentanil Concentration (ng/mL)",
                  f"{conc_pred:.4f}")

        st.subheader("Feature Vector")
        st.dataframe(features)

        st.subheader("Patient History")
        st.dataframe(pd.DataFrame(state["history"]))
