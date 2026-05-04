import sys
from pathlib import Path
import json

import joblib
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

from app.config.db import get_db_connection
from app.services.anomaly_feature_service import build_anomaly_features, get_feature_columns
from app.services.mlflow_logging_service import setup_mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FOLDER = PROJECT_ROOT / "models" / "anomaly_detection"

CONTAMINATION = 0.02

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


def save_score_distribution_plot(scores, parameter):
    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

    plot_path = MODEL_FOLDER / f"{parameter}_score_distribution.png"

    plt.figure()
    plt.hist(scores, bins=30)
    plt.title(f"Anomaly Score Distribution - {parameter}")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def train_model_for_parameter(parameter_df, parameter):
    feature_df = build_anomaly_features(parameter_df)

    feature_columns = get_feature_columns()

    x_train = feature_df[feature_columns]

    model = IsolationForest(
        n_estimators=100,
        contamination=CONTAMINATION,
        random_state=42,
    )

    model.fit(x_train)

    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_FOLDER / f"isolation_forest_{parameter}.joblib"

    joblib.dump(model, model_path)

    # decision_function gives higher values for normal rows.
    # We multiply by -1 so higher score means more anomalous.
    scores = -model.decision_function(x_train)

    plot_path = save_score_distribution_plot(scores, parameter)

    anomaly_predictions = model.predict(x_train)
    anomaly_count = int((anomaly_predictions == -1).sum())
    anomaly_rate = anomaly_count / len(x_train)

    print(f"Trained model for {parameter}")
    print(f"Saved to: {model_path}")
    print(f"Training rows: {len(x_train)}")
    print(f"Training anomaly rate: {anomaly_rate:.4f}")

    return {
        "parameter": parameter,
        "model_path": model_path,
        "plot_path": plot_path,
        "training_rows": len(x_train),
        "anomaly_count": anomaly_count,
        "anomaly_rate": anomaly_rate,
    }


def main():
    setup_mlflow()

    df = load_clean_sensor_data()

    if df.empty:
        print("No sensor data found for training.")
        sys.exit(1)

    df = keep_physically_valid_rows(df)

    if df.empty:
        print("No physically valid sensor data found for training.")
        sys.exit(1)

    print("Isolation Forest Training")
    print("-------------------------")
    print(f"Rows loaded for training: {len(df)}")

    parameters = sorted(df["parameter"].unique())

    feature_columns = get_feature_columns()

    with mlflow.start_run(run_name="isolation_forest_training"):
        mlflow.log_param("model_name", "Sensor Anomaly Detection")
        mlflow.log_param("model_type", "Isolation Forest")
        mlflow.log_param("algorithm", "IsolationForest")
        mlflow.log_param("contamination", CONTAMINATION)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("feature_set_version", "v2")
        mlflow.log_param("training_filter", "timestamp older than 7 days + physically valid rows")

        mlflow.log_text(
            json.dumps(feature_columns, indent=4),
            "feature_columns.json",
        )

        mlflow.log_text(
            json.dumps(VALID_RANGES, indent=4),
            "valid_sensor_ranges.json",
        )

        trained_model_count = 0
        total_training_rows = 0

        for parameter in parameters:
            parameter_df = df[df["parameter"] == parameter].copy()

            if len(parameter_df) < 20:
                print(f"Skipping {parameter}: not enough data")
                continue

            result = train_model_for_parameter(parameter_df, parameter)

            trained_model_count += 1
            total_training_rows += result["training_rows"]

            mlflow.log_metric(
                f"{parameter}_training_rows",
                result["training_rows"],
            )

            mlflow.log_metric(
                f"{parameter}_training_anomaly_count",
                result["anomaly_count"],
            )

            mlflow.log_metric(
                f"{parameter}_training_anomaly_rate",
                result["anomaly_rate"],
            )

            mlflow.log_artifact(
                str(result["model_path"]),
                artifact_path="models",
            )

            mlflow.log_artifact(
                str(result["plot_path"]),
                artifact_path="score_distribution_plots",
            )

        mlflow.log_metric("trained_model_count", trained_model_count)
        mlflow.log_metric("total_training_rows", total_training_rows)

    print("\nTraining completed.")
    print("MLflow logging completed.")


if __name__ == "__main__":
    main()