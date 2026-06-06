import pandas as pd


def compute_bmi(weight, height_cm):

    height_m = height_cm / 100

    return weight / (height_m ** 2)


def engineer_input_features(
    patient_state,
    raw_input
):

    time = raw_input["Time"]

    rate = raw_input["Rate"]

    amt = raw_input["Amt"]

    age = raw_input["Age"]

    sex = raw_input["Sex"]

    wt = raw_input["Wt"]

    ht = raw_input["Ht"]

    bsa = raw_input["BSA"]

    lbm = raw_input["LBM"]

    # Time delta
    TimeDelta = time - patient_state["last_time"]

    # Cumulative dose
    cumulative_dose = patient_state["CumulativeDose"]
    cumulative_dose += amt

    # Infusion status
    infusion_active = 1 if rate > 0 else 0

    # BMI
    bmi = compute_bmi(wt, ht)

    # Encoded sex
    sex_encoded = 1 if sex == "Male" else 0

    features = pd.DataFrame([{
        "Time": time,
        "TimeDelta": TimeDelta,
        "Rate": rate,
        "Amt": amt,
        "CumulativeDose": cumulative_dose,
        "InfusionActive": infusion_active,
        "Age": age,
        "Sex_Encoded": sex_encoded,
        "Ht": ht,
        "Wt": wt,
        "BSA": bsa,
        "LBM": lbm,
        "BMI": bmi
    }])

    return features
