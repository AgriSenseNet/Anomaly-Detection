import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.config.db import get_db_connection
from app.services.anomaly_feature_service import build_anomaly_features, get_feature_columns

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FOLDER = PROJECT_ROOT / "models" / "anomaly_detection"

VALID_RANGES = {
    "soil_moisture": (0, 100),
    "soil_temp": (0, 60),
    "ambient_temp": (0, 60),
    "humidity": (0, 100),
    "pressure": (850, 1100),
    "solar_radiation": (0, 120000),
}

def load_clean_sensor_data():
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
        WHERE timestamp < NOW() - INTERVAL '7 days'
        ORDER BY timestamp ASC;
    """

    df = pd.read_sql(query, conn)
    conn.close()
    return df

def train_model_for_parameter(parameter_df, parameter):
    feature_df = build_anomaly_features(parameter_df)

    feature_columns = get_feature_columns()

    x_train = feature_df[feature_columns]

    model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    model.fit(x_train)

    MODEL_FOLDER.mkdir(parents = True, exist_ok = True)

    model_path = MODEL_FOLDER / f"isolation_forest_{parameter}.joblib"

    joblib.dump(model, model_path)

    print(f"Trained model for {parameter}")
    print(f"Saved to: {model_path}")

def main():
    df = load_clean_sensor_data()

    if df.empty:
        print("No valid sensor data found for training.")
        sys.exit(1)

    print("Isolation Forest Training")
    print("-------------------------")
    print(f"Rows loaded: {len(df)}")

    parameters = sorted(df["parameter"].unique())

    for parameter in parameters:
        parameter_df = df[df["parameter"] == parameter].copy()

        #enough_data_df = parameter_df.groupby(["device_id"]).filter(lambda x: len(x) >= 10)

        #if enough_data_df.empty:
        #   print(f"Not enough data for parameter {parameter} to train a model. Skipping.")
        #   continue

        if len(parameter_df) < 20:
            print(f"Skipping {parameter}: not enough data")
            continue  

        train_model_for_parameter(parameter_df, parameter)

    print("\nTraining completed.")

if __name__ == "__main__":
    main()        
        
