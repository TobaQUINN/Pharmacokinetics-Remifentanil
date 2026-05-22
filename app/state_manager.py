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

# ============================================
# IN-MEMORY STATE STORE
# ============================================

patient_states = {}


# ============================================
# INITIALIZE PATIENT
# ============================================

def initialize_patient(patient_id):
    """
    Creates default patient state
    if patient does not exist.
    """

    if patient_id not in patient_states:

        patient_states[patient_id] = {

            "last_time": 0.0,

            "cum_dose": 0.0,

            "last_rate": 0.0
        }


# ============================================
# GET PATIENT STATE
# ============================================

def get_patient_state(patient_id):
    """
    Retrieves patient state.
    """

    initialize_patient(patient_id)

    return patient_states[patient_id]


# ============================================
# UPDATE PATIENT STATE
# ============================================

def update_patient_state(
    patient_id,
    last_time,
    cum_dose,
    last_rate
):
    """
    Updates stored patient state.
    """

    patient_states[patient_id] = {

        "last_time": last_time,

        "cum_dose": cum_dose,

        "last_rate": last_rate
    }


# ============================================
# RESET PATIENT
# ============================================

def reset_patient(patient_id):
    """
    Clears patient state.
    """

    if patient_id in patient_states:

        del patient_states[patient_id]

        return True

    return False
