import pandas as pd


def engineer_features(data):
    # Time features
    data['delta_time'] = data.groupby('ID')['Time'].diff().fillna(0)

    # Cumulative dose
    data['cum_dose'] = data.groupby('ID')['Amt'].cumsum()

    # Infusion active flag
    data['InfusionActive'] = (data['Rate'] > 0).astype(int)

    return data
