"""
Production anomaly inference — runs every 10 minutes via Kubernetes CronJob.

Fetches last 60 minutes of sensor_readings from PostgreSQL for feature context,
flags only readings in the last 10 minutes as new anomalies,
scores with Isolation Forest + LSTM Autoencoder,
and writes detections to anomaly_events.
"""

import datetime
import os
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import torch

sys.path.insert(0, str(Path(__file__).parent))
from app.services.anomaly_feature_service import build_anomaly_features, get_feature_columns
from app.services.lstm_autoencoder_service import LSTMAutoencoder
from app.services.mlflow_logging_service import setup_mlflow

PARAMETERS = ["soil_moisture", "soil_temp", "ambient_temp", "humidity", "pressure", "lux"]
CONTEXT_WINDOW_MINUTES = 60   # history needed to compute rolling features
NEW_WINDOW_MINUTES     = 10   # only flag readings newer than this as new anomalies

DB_CONFIG = {
    "host":     os.environ["POSTGRES_HOST"],
    "port":     int(os.environ.get("POSTGRES_PORT", 5432)),
    "dbname":   os.environ.get("POSTGRES_DB", "smart_agri_c2"),
    "user":     os.environ.get("POSTGRES_USER", "postgres"),
    "password": os.environ["POSTGRES_PASSWORD"],
}


def get_db_conn():
    return psycopg2.connect(**DB_CONFIG)


def fetch_sensor_data(conn) -> pd.DataFrame:
    query = """
        SELECT id, field_id, device_id, timestamp, parameter, value
        FROM sensor_readings
        WHERE timestamp > NOW() - INTERVAL '%s minutes'
          AND parameter = ANY(%s)
        ORDER BY field_id, device_id, parameter, timestamp
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, (CONTEXT_WINDOW_MINUTES, list(PARAMETERS)))
        rows = cur.fetchall()
    return pd.DataFrame(rows)


def load_if_models() -> dict:
    setup_mlflow()
    client  = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions("name='Anomaly_Isolation_Forest'")
    staging  = [v for v in versions if v.tags.get("stage") == "Staging"]
    mv       = max(staging, key=lambda v: int(v.version))
    print(f"  IF  — run {mv.run_id[:8]}, v{mv.version}")
    models = {}
    for param in PARAMETERS:
        path = mlflow.artifacts.download_artifacts(
            run_id=mv.run_id, artifact_path=f"models/isolation_forest_{param}.joblib"
        )
        models[param] = joblib.load(path)
    return models


def load_autoencoder():
    """Returns (model, scaler, threshold, seq_len, feature_columns)."""
    client   = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions("name='Anomaly_LSTM_Autoencoder'")
    staging  = [v for v in versions if v.tags.get("stage") == "Staging"]
    mv       = max(staging, key=lambda v: int(v.version))
    print(f"  AE  — run {mv.run_id[:8]}, v{mv.version}")

    model_path  = mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path="models/lstm_autoencoder.pt"
    )
    scaler_path = mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path="models/lstm_autoencoder_scaler.joblib"
    )

    ckpt  = torch.load(model_path, map_location="cpu", weights_only=False)
    model = LSTMAutoencoder(input_size=ckpt["input_size"], hidden_size=ckpt["hidden_size"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return (
        model, joblib.load(scaler_path),
        ckpt["reconstruction_threshold"],
        ckpt["sequence_length"],
        ckpt["feature_columns"],
    )


def new_cutoff() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=NEW_WINDOW_MINUTES)


def score_isolation_forest(if_models, df_feat, feature_columns) -> list[dict]:
    cutoff = new_cutoff()
    events = []
    for (field_id, device_id, param), group in df_feat.groupby(
        ["field_id", "device_id", "parameter"]
    ):
        if param not in if_models:
            continue
        model  = if_models[param]
        X      = group[feature_columns].values
        preds  = model.predict(X)
        scores = -model.decision_function(X)

        for i, (pred, score) in enumerate(zip(preds, scores)):
            row = group.iloc[i]
            if pred == -1 and row["timestamp"] >= cutoff:
                events.append({
                    "field_id":      field_id,
                    "device_id":     device_id,
                    "timestamp":     row["timestamp"],
                    "sensor_type":   param,
                    "anomaly_score": float(score),
                    "anomaly_type":  "isolation_forest",
                    "value":         float(row["value"]),
                    "model_name":    "Anomaly_Isolation_Forest",
                })
    return events


def score_autoencoder(ae_model, ae_scaler, ae_threshold, ae_seq_len, ae_features, df_feat) -> list[dict]:
    cutoff = new_cutoff()
    events = []
    for (field_id, device_id, param), group in df_feat.groupby(
        ["field_id", "device_id", "parameter"]
    ):
        group    = group.sort_values("timestamp").reset_index(drop=True)
        X        = group[ae_features].values.astype(np.float32)
        X_scaled = ae_scaler.transform(X)

        n = len(X_scaled)
        if n < ae_seq_len:
            continue

        seqs = np.stack([X_scaled[i:i + ae_seq_len] for i in range(n - ae_seq_len + 1)])
        t    = torch.tensor(seqs, dtype=torch.float32)
        with torch.no_grad():
            recon  = ae_model(t)
            errors = torch.mean((recon - t) ** 2, dim=(1, 2)).numpy()

        for seq_i, err in enumerate(errors):
            last_row_i = seq_i + ae_seq_len - 1
            row = group.iloc[last_row_i]
            if err > ae_threshold and row["timestamp"] >= cutoff:
                events.append({
                    "field_id":      field_id,
                    "device_id":     device_id,
                    "timestamp":     row["timestamp"],
                    "sensor_type":   param,
                    "anomaly_score": float(err),
                    "anomaly_type":  "lstm_autoencoder",
                    "value":         float(row["value"]),
                    "model_name":    "Anomaly_LSTM_Autoencoder",
                })
    return events


def insert_anomaly_events(conn, events: list[dict]):
    if not events:
        return
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, """
            INSERT INTO anomaly_events
                (field_id, device_id, timestamp, sensor_type,
                 anomaly_score, anomaly_type, value, model_name)
            VALUES
                (%(field_id)s, %(device_id)s, %(timestamp)s, %(sensor_type)s,
                 %(anomaly_score)s, %(anomaly_type)s, %(value)s, %(model_name)s)
        """, events)
    conn.commit()


if __name__ == "__main__":
    run_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f"[{run_at}] Anomaly inference starting …")

    print("Loading models from MLflow …")
    setup_mlflow()
    if_models                                    = load_if_models()
    ae_model, ae_scaler, ae_thr, ae_seq_len, ae_features = load_autoencoder()
    feature_columns = get_feature_columns()

    conn   = get_db_conn()
    df_raw = fetch_sensor_data(conn)

    if df_raw.empty:
        print("No sensor data in last 60 minutes — nothing to score.")
        conn.close()
        sys.exit(0)

    print(
        f"Fetched {len(df_raw)} rows | "
        f"{df_raw['field_id'].nunique()} field(s) | "
        f"{df_raw['device_id'].nunique()} device(s)"
    )

    df_feat   = build_anomaly_features(df_raw)
    if_events = score_isolation_forest(if_models, df_feat, feature_columns)
    ae_events = score_autoencoder(ae_model, ae_scaler, ae_thr, ae_seq_len, ae_features, df_feat)

    all_events = if_events + ae_events
    insert_anomaly_events(conn, all_events)
    conn.close()

    print(f"IF anomalies : {len(if_events)}")
    print(f"AE anomalies : {len(ae_events)}")
    print(f"Total written: {len(all_events)}")
    print(f"[{run_at}] Done.")
