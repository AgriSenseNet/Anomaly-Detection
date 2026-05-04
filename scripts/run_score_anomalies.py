import sys
import mlflow
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FOLDER = PROJECT_ROOT / "models" / "anomaly_detection"

from app.config.db import get_db_connection
from app.services.anomaly_feature_service import build_anomaly_features, get_feature_columns
from app.services.mlflow_logging_service import setup_mlflow

VALID_RANGES = {
    "soil_moisture": (0, 100),
    "soil_temp": (0, 60),
    "ambient_temp": (0, 60),
    "humidity": (0, 100),
    "pressure": (850, 1100),
    "solar_radiation": (0, 120000),
    }

def load_sensor_data():
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
        WHERE timestamp >= NOW() - INTERVAL '7 days'
        ORDER BY timestamp ASC;
    """

    df = pd.read_sql(query, conn)
    conn.close()

    return df

def clear_old_anomalies():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM anomaly_events where model_name = 'isolation_forest';")
    conn.commit()
    cursor.close()
    conn.close()


def get_rule_based_anomaly_type(parameter, value):
    min_value, max_value = VALID_RANGES[parameter]

    if value < min_value or value > max_value:
        return "physically_invalid_value"

    return "isolation_forest_outlier"

def insert_anomaly(row, anomaly_score, anomaly_type):
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
        insert into anomaly_events (
            field_id,
            device_id,
            timestamp,
            sensor_type,
            value,
            anomaly_score,
            anomaly_type,
            model_name
        )
        values (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(
        query,
        (
            row["field_id"],
            row["device_id"],
            row["timestamp"],
            row["parameter"],
            float(row["value"]),
            float(anomaly_score),
            anomaly_type,
            "isolation_forest",
        ),
    )
    conn.commit()
    cursor.close()
    conn.close()


def score_parameter(parameter_df, parameter):
    model_path = MODEL_FOLDER / f"isolation_forest_{parameter}.joblib"

    if not model_path.exists():
        print(f"Model not found for {parameter}: {model_path}")
        return 0    

    model = joblib.load(model_path)

    feature_df = build_anomaly_features(parameter_df)

    feature_columns = get_feature_columns()

    x_score = feature_df[feature_columns]

    predictions = model.predict(x_score)

    # decision_function gives higher values for more normal points and lower values for more anomalous points. We want the opposite, so we multiply by -1.
    # After multiply by -1 so higher score means more anomalous. 

    scores = -model.decision_function(x_score)

    anomaly_count = 0

    for index, prediction in enumerate(predictions):
        if prediction == -1:
            row = parameter_df.iloc[index]
            anomaly_score = scores[index]

            anomaly_type = get_rule_based_anomaly_type(row["parameter"], row["value"])

            insert_anomaly(row, anomaly_score, anomaly_type)
            anomaly_count += 1

    print(f"Anomalies found for {parameter}: {anomaly_count}")

    return anomaly_count


def main():
    setup_mlflow()
    print("Scoring sensor anomalies")
    print("------------------------")

    df= load_sensor_data()

    if df.empty:
        print("No sensor data found for scoring.")
        sys.exit(0)


    total_anomalies = 0

    parameters = sorted(df["parameter"].unique())

    with mlflow.start_run(run_name="isolation_forest_scoring"):
        mlflow.log_param("model_name", "Sensor Anomaly Detection")
        mlflow.log_param("model_type", "Isolation Forest")
        mlflow.log_param("scoring_window", "last 7 days")
        mlflow.log_param("output_table", "anomaly_events")

        mlflow.log_metric("scoring_rows", len(df))    

        for parameter in parameters:
            if parameter not in VALID_RANGES:
                print(f"Skipping unknown parameter: {parameter}")
                continue

            parameter_df = df[df["parameter"] == parameter].copy()

            anomaly_count = score_parameter(parameter_df, parameter)
            total_anomalies += anomaly_count

            mlflow.log_metric(
                f"{parameter}_scored_rows",
                len(parameter_df),
            )

            mlflow.log_metric(
                f"{parameter}_anomalies_saved",
                anomaly_count,
            )

        mlflow.log_metric("total_anomalies_saved", total_anomalies)        
        
    print("------------------------")
    print(f"Total anomalies saved: {total_anomalies}")
    print("MLflow scoring logging completed.")

if __name__ == "__main__":
    main()
