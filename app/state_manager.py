# ============================================
# state_manager.py
# ============================================

"""
Handles temporal patient state.

This is critical because:
PK systems are sequential.

The model depends on:
- infusion history
- cumulative exposure
- temporal evolution

For now:
state is stored in memory.

Production systems would likely use:
- Redis
- PostgreSQL
- MongoDB
"""

from collections import defaultdict

# Stores patient states
patient_states = defaultdict(dict)


def initialize_patient(patient_id):

    if patient_id not in patient_states:

        patient_states[patient_id] = {
            "last_time": 0.0,
            "CumulativeDose": 0.0,
            "history": []
        }


def get_patient_state(patient_id):

    initialize_patient(patient_id)

    return patient_states[patient_id]


def update_patient_state(
    patient_id,
    time,
    amt,
    prediction,
    concentration
):

    state = get_patient_state(patient_id)

    TimeDelta = max(0, time - get_patient_state(patient_id)["last_time"])

    state["last_time"] = time

    state["history"].append({
        "Time(s)": time,
        "Dose Amount(ng)": amt,
        "Prediction(Log Concentration)": prediction,
        "Plasma Concentration(ng/mL)": concentration,
    })


def reset_patient(patient_id):

    if patient_id in patient_states:
        del patient_states[patient_id]
