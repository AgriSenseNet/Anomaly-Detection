"""
Smoke-test for anomaly detection inference.
Downloads Anomaly_Isolation_Forest models from MLflow registry, builds fake
sensor readings, applies feature engineering, and scores each parameter.
"""
import sys
import os
import tempfile
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from app.services.anomaly_feature_service import build_anomaly_features, get_feature_columns
from app.services.mlflow_logging_service import setup_mlflow

PARAMETERS = ["soil_moisture", "soil_temp", "ambient_temp", "humidity", "pressure", "lux"]

FAKE_RANGES = {
    "soil_moisture": (20.0,  60.0),
    "soil_temp":     (18.0,  35.0),
    "ambient_temp":  (22.0,  32.0),
    "humidity":      (55.0,  90.0),
    "pressure":      (995.0, 1015.0),
    "lux":           (0.0,   80000.0),
}

# One reading clearly out of normal range to trigger anomaly
SPIKE = {
    "soil_moisture": 95.0,
    "soil_temp":     58.0,
    "ambient_temp":  55.0,
    "humidity":      99.0,
    "pressure":      800.0,
    "lux":           119000.0,
}


def _make_fake_df(parameter: str, n: int = 50) -> pd.DataFrame:
    """50 normal readings + 1 spike at the end."""
    rng = np.random.default_rng(42)
    lo, hi = FAKE_RANGES[parameter]
    values = rng.uniform(lo, hi, n).tolist() + [SPIKE[parameter]]
    import datetime
    base = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    timestamps = [base + datetime.timedelta(minutes=5 * i) for i in range(n + 1)]
    return pd.DataFrame({
        "id":         range(n + 1),
        "field_id":   "test_field_001",
        "device_id":  "device_001",
        "timestamp":  timestamps,
        "parameter":  parameter,
        "value":      values,
    })


def _download_models() -> dict[str, object]:
    """Download all per-parameter IF models from the MLflow registry."""
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()

    versions = client.search_model_versions("name='Anomaly_Isolation_Forest'")
    staging  = [v for v in versions if v.tags.get("stage") == "Staging"]
    if not staging:
        raise RuntimeError("No Anomaly_Isolation_Forest model tagged stage=Staging in MLflow.")

    mv      = max(staging, key=lambda v: int(v.version))
    run_id  = mv.run_id
    print(f"  Downloading from run {run_id[:8]} …")

    models = {}
    for param in PARAMETERS:
        artifact_path = f"models/isolation_forest_{param}.joblib"
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=artifact_path
        )
        models[param] = joblib.load(local_path)
        print(f"  Loaded isolation_forest_{param}")
    return models


if __name__ == "__main__":
    print("Downloading Isolation Forest models from MLflow …")
    models = _download_models()

    feature_columns = get_feature_columns()
    print(f"\nRunning inference on fake sensor data (50 normal + 1 spike each) …")

    for param in PARAMETERS:
        df_raw    = _make_fake_df(param)
        df_feat   = build_anomaly_features(df_raw)
        X         = df_feat[feature_columns].values

        model     = models[param]
        preds     = model.predict(X)            # -1 = anomaly, +1 = normal
        scores    = -model.decision_function(X) # higher = more anomalous

        n_anomaly = int((preds == -1).sum())
        spike_pred = "ANOMALY" if preds[-1] == -1 else "normal"
        spike_score = scores[-1]

        print(f"\n  {param:15s}  anomalies={n_anomaly}/51  "
              f"spike → {spike_pred} (score={spike_score:.4f})")

    print("\nAnomaly Isolation Forest inference — OK")
