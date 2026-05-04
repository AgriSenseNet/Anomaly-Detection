import pandas as pd


FEATURE_COLUMNS = [
    "value",
    "delta_from_last",
    "absolute_delta",
    "rolling_mean_6",
    "rolling_std_6",
    "rolling_range_6",
    "same_value_count_6",
    "rolling_z_score_6",
    "hour_of_day",
    "day_of_week",
]


def count_same_values(values):
    """
    Count how many times the latest value appears in the recent rolling window.

    Example:
    [31.11, 31.11, 31.11, 31.11] -> 4
    [30.10, 31.20, 29.90, 31.11] -> 1
    """

    latest_value = values.iloc[-1]

    same_count = (values == latest_value).sum()

    return same_count


def build_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
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
    Same rows, but with extra feature columns for anomaly detection.
    """

    if df.empty:
        return df

    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values(
        by=["device_id", "parameter", "timestamp"]
    )

    group_columns = ["device_id", "parameter"]

    # Previous reading for same device and same sensor type
    df["previous_value"] = df.groupby(group_columns)["value"].shift(1)

    # Change from previous reading
    df["delta_from_last"] = df["value"] - df["previous_value"]

    # Absolute change from previous reading
    # This is useful because both sudden increase and sudden decrease are suspicious.
    df["absolute_delta"] = df["delta_from_last"].abs()

    # Rolling mean of recent 6 readings
    df["rolling_mean_6"] = df.groupby(group_columns)["value"].transform(
        lambda values: values.rolling(window=6, min_periods=1).mean()
    )

    # Rolling standard deviation of recent 6 readings
    df["rolling_std_6"] = df.groupby(group_columns)["value"].transform(
        lambda values: values.rolling(window=6, min_periods=1).std()
    )

    # Rolling minimum of recent 6 readings
    df["rolling_min_6"] = df.groupby(group_columns)["value"].transform(
        lambda values: values.rolling(window=6, min_periods=1).min()
    )

    # Rolling maximum of recent 6 readings
    df["rolling_max_6"] = df.groupby(group_columns)["value"].transform(
        lambda values: values.rolling(window=6, min_periods=1).max()
    )

    # Rolling range = max - min.
    # For stuck sensors, this becomes very small or 0.
    df["rolling_range_6"] = df["rolling_max_6"] - df["rolling_min_6"]

    # Count how many times the current value appears in the recent 6 readings.
    # This helps detect repeated stuck values.
    df["same_value_count_6"] = df.groupby(group_columns)["value"].transform(
        lambda values: values.rolling(window=6, min_periods=1).apply(
            count_same_values,
            raw=False,
        )
    )

    # Z-score compared to recent rolling behavior.
    # This helps detect spikes.
    df["rolling_z_score_6"] = (
        df["value"] - df["rolling_mean_6"]
    ) / df["rolling_std_6"]

    # Time-based features
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Fill empty values created by first rows / zero std
    df["delta_from_last"] = df["delta_from_last"].fillna(0)
    df["absolute_delta"] = df["absolute_delta"].fillna(0)
    df["rolling_std_6"] = df["rolling_std_6"].fillna(0)
    df["rolling_range_6"] = df["rolling_range_6"].fillna(0)
    df["same_value_count_6"] = df["same_value_count_6"].fillna(1)

    # If rolling_std_6 is 0, z-score becomes infinity/NaN.
    # Replace those with 0 for safe model training.
    df["rolling_z_score_6"] = (
        df["rolling_z_score_6"]
        .replace([float("inf"), float("-inf")], 0)
        .fillna(0)
    )

    return df


def get_feature_columns():
    return FEATURE_COLUMNS