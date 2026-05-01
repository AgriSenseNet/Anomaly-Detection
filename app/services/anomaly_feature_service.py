import pandas as pd

FEATURE_COLUMNS = [
    "value",
    "delta_from_last",
    "rolling_mean_6",
    "rolling_std_6",
    "hours_of_day",
    "day_of_week",
]

def build_anomaly_features(df:pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw sensor readings into ML-ready features.

    Input columns expected:
    - id
    - field_id
    - device_id
    - timestamp
    - parameter
    - value

    Output:
    Same rows, but with extra feature columns.
    """

    if df.empty:
        return df
    
    df = df.copy()#make shure original values don't change

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values( by = ["device_id", "parameter", "timestamp"])

    df["previous_value"] = df.groupby(["device_id", "parameter"])["value"].shift(1)

    df["delta_from_last"] = df["value"] - df["previous_value"]

    # Difference from previous reading
    df["delta_from_last"] = df["value"] - df["previous_value"]

    # Rolling mean of recent 6 readings for same device and same sensor type
    df["rolling_mean_6"] = df.groupby(
        ["device_id", "parameter"]
    )["value"].transform(
        lambda values: values.rolling(window=6, min_periods=1).mean()
    )

    # Rolling standard deviation of recent 6 readings
    df["rolling_std_6"] = df.groupby(
        ["device_id", "parameter"]
    )["value"].transform(
        lambda values: values.rolling(window=6, min_periods=1).std()
    )

    # Time-based features

    df["hours_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # First row has no previous value, so delta becomes empty.
    # First rolling std can also become empty.
    # Fill them with 0.

    df["delta_from_last"] = df["delta_from_last"].fillna(0)
    df["rolling_std_6"] = df["rolling_std_6"].fillna(0)

    return df

def get_feature_columns():
    return FEATURE_COLUMNS
