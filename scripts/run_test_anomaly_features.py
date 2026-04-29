import pandas as pd

from app.config.db import get_db_connection
from app.services.anomaly_feature_service import build_anomaly_features

def main():

    conn = get_db_connection()
    query = """
        SELECT
            id,
            field_id,
            device_id,
            timestamp,
            parameter,
            value
        FROM sensor_readings
        ORDER BY timestamp DESC;
    """

    df = pd.read_sql(query, conn)
    conn.close()

    features_df = build_anomaly_features(df)

    print("Anomaly Feature Engineering Test")
    print("--------------------------------")
    print(features_df.head(20).to_string())

    print("\nCreated feature columns:")
    print("- delta_from_last")
    print("- rolling_mean_6")
    print("- rolling_std_6")
    print("- hour_of_day")
    print("- day_of_week")


if __name__ == "__main__":
    main()     