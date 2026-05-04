import sys
from pathlib import Path
import mlflow

import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FOLDER = PROJECT_ROOT / "models" / "anomaly_detection"

LABEL_FILE = PROJECT_ROOT / "docs" / "evidence" / "sprint2" / "generated_anomaly_labels.csv"

REPORT_FILE = PROJECT_ROOT / "docs" / "evidence" / "sprint2" / "anomaly_detection_test_report.csv"

PREDICTIONS_FILE = PROJECT_ROOT / "docs" / "evidence" / "sprint2" / "anomaly_predictions.csv"

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

def is_physically_invalid(parameter, value):
    min_value, max_value = VALID_RANGES[parameter]
    if value < min_value or value > max_value:
        return True
    return False

def load_labelled_data():
    df = pd.read_csv(LABEL_FILE)

    # CSV uses sensor_type, but our feature code uses parameter
    df["parameter"] = df["sensor_type"]

    # Add columns expected by feature engineering
    df["field_id"] = "field_001"

    return df


def predict_for_parameter(parameter_df, parameter):
    model_path = MODEL_FOLDER / f"isolation_forest_{parameter}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path )

    feature_df = build_anomaly_features(parameter_df)

    feature_columns = get_feature_columns()
    x_test = feature_df[feature_columns]

    model_predictions = model.predict(x_test)

    scores = -model.decision_function(x_test)

    predicted_labels = []
    anomaly_scores = []
    prediction_reasons = []

    

    for index, prediction in enumerate(model_predictions):
        row = parameter_df.iloc[index]
        parameter = row["parameter"]
        value = row["value"]

        physically_invalid = is_physically_invalid(parameter, value)

        # 1 = anomaly
        # 0 = normal 

        if physically_invalid:
            predicted_labels.append(1)
            prediction_reasons.append("physically_invalid_value")

        elif prediction == -1:
            predicted_labels.append(1)
            prediction_reasons.append("isolation_forest_outlier")

        else:
            predicted_labels.append(0)
            prediction_reasons.append("normal_value")

        anomaly_scores.append(float(scores[index]))

    feature_df["predicted_label"] = predicted_labels
    feature_df["anomaly_score"] = anomaly_scores
    feature_df["prediction_reason"] = prediction_reasons

    return feature_df   


def main():
    setup_mlflow()
    print("Evaluating anomaly detection model")
    print("----------------------------------")

    df = load_labelled_data()

    all_predictions = []

    parameters = sorted(df["parameter"].unique())

    for parameter in parameters:
        parameter_df = df[df["parameter"] == parameter].copy()

        predicted_df = predict_for_parameter(parameter_df, parameter)
        all_predictions.append(predicted_df)

    result_df = pd.concat(all_predictions)

    true_labels = result_df["true_label"]
    predicted_labels = result_df["predicted_label"]

    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()


    report_df = pd.DataFrame(
        [
            {
                "model_name": "isolation_forest_hybrid_rules",
                "total_test_rows": len(result_df),
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
            }
        ]
    )

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    report_df.to_csv(REPORT_FILE, index=False)

    result_df[
        [
            "id",
            "device_id",
            "timestamp",
            "sensor_type",
            "value",
            "true_label",
            "predicted_label",
            "anomaly_score",
            "anomaly_reason",
            "prediction_reason",
        ]
    ].to_csv(PREDICTIONS_FILE, index=False)

    with mlflow.start_run(run_name="isolation_forest_evaluation"):

        mlflow.log_param("model_name", "Sensor Anomaly Detection")
        mlflow.log_param("model_type", "Isolation Forest")
        mlflow.log_param("evaluation_dataset", str(LABEL_FILE.name))

        mlflow.log_metric("total_test_rows", len(result_df))
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.log_metric("true_positive", int(tp))
        mlflow.log_metric("false_positive", int(fp))
        mlflow.log_metric("true_negative", int(tn))
        mlflow.log_metric("false_negative", int(fn))

        mlflow.log_artifact(
            str(REPORT_FILE),
            artifact_path="evaluation_reports",
        )

        mlflow.log_artifact(
            str(PREDICTIONS_FILE),
            artifact_path="evaluation_reports",
        )

    print("Evaluation completed.")
    print(f"Total rows: {len(result_df)}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print()
    print(f"Saved report to: {REPORT_FILE}")
    print(f"Saved predictions to: {PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()