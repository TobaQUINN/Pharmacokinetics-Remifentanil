import pandas as pd


def engineer_features(data):
    # Time features
    data['TimeDelta'] = data.groupby('ID')['Time'].diff().fillna(0)

    # Cumulative dose
    data['CumulativeDose'] = data.groupby('ID')['Amt'].cumsum()

    # Infusion active flag
    data['InfusionActive'] = (data['Rate'] > 0).astype(int)

    return data
