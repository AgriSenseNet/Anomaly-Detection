import sys
from pathlib import Path

import mlflow
import pandas as pd

from app.config.db import get_db_connection
from app.services.anomaly_feature_service import build_anomaly_features, get_feature_columns
from app.services.lstm_autoencoder_service import (
    train_lstm_autoencoder,
    SEQUENCE_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    HIDDEN_SIZE,
)
from app.services.mlflow_logging_service import setup_mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FOLDER = PROJECT_ROOT / "models" / "anomaly_detection"

VALID_RANGES = {
    "soil_moisture": (0, 100),
    "soil_temp": (0, 60),
    "ambient_temp": (0, 60),
    "humidity": (0, 100),
    "pressure": (850, 1100),
    "lux": (0, 120000),
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
        ORDER BY device_id, parameter, timestamp ASC;
    """

    df = pd.read_sql(query, conn)
    conn.close()

    return df


def keep_physically_valid_rows(df):
    clean_parts = []

    for parameter, (minimum_value, maximum_value) in VALID_RANGES.items():
        parameter_df = df[df["parameter"] == parameter].copy()

        parameter_df = parameter_df[
            (parameter_df["value"] >= minimum_value)
            & (parameter_df["value"] <= maximum_value)
        ]

        clean_parts.append(parameter_df)

    if not clean_parts:
        return pd.DataFrame()

    return pd.concat(clean_parts, ignore_index=True)


def main():
    setup_mlflow()

    print("Training LSTM Autoencoder")
    print("-------------------------")
    print("Sequence mode: per device_id + parameter")

    df = load_clean_sensor_data()

    if df.empty:
        print("No sensor data found for autoencoder training.")
        sys.exit(1)

    df = keep_physically_valid_rows(df)

    if df.empty:
        print("No physically valid sensor data found for autoencoder training.")
        sys.exit(1)

    feature_df = build_anomaly_features(df)

    feature_columns = get_feature_columns()

    with mlflow.start_run(run_name="lstm_autoencoder_training"):
        mlflow.log_param("model_name", "Sensor Anomaly Detection")
        mlflow.log_param("model_type", "LSTM Autoencoder")
        mlflow.log_param("sequence_mode", "per_device_and_parameter")
        mlflow.log_param("sequence_length", SEQUENCE_LENGTH)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("hidden_size", HIDDEN_SIZE)
        mlflow.log_param("training_filter", "physically valid rows only")

        mlflow.log_text(
            "\n".join(feature_columns),
            "autoencoder_feature_columns.txt",
        )

        result = train_lstm_autoencoder(
            feature_df=feature_df,
            feature_columns=feature_columns,
            model_folder=MODEL_FOLDER,
        )

        mlflow.log_metric("training_rows", result["training_rows"])
        mlflow.log_metric("training_sequences", result["training_sequences"])
        mlflow.log_metric("final_training_loss", result["final_training_loss"])
        mlflow.log_metric("reconstruction_threshold", result["reconstruction_threshold"])
        mlflow.log_metric("feature_count", result["input_size"])

        mlflow.log_artifact(
            str(result["model_file"]),
            artifact_path="models",
        )

        mlflow.log_artifact(
            str(result["scaler_file"]),
            artifact_path="models",
        )

        mlflow.log_artifact(
            str(result["metadata_file"]),
            artifact_path="sequence_metadata",
        )

    print("-------------------------")
    print("LSTM Autoencoder training completed.")
    print(f"Rows used: {result['training_rows']}")
    print(f"Sequences created: {result['training_sequences']}")
    print(f"Model saved to: {result['model_file']}")
    print(f"Scaler saved to: {result['scaler_file']}")
    print(f"Reconstruction threshold: {result['reconstruction_threshold']:.6f}")
    print("MLflow autoencoder logging completed.")


if __name__ == "__main__":
    main()