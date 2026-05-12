"""
Smoke-test for anomaly detection inference.
Downloads Isolation Forest + LSTM Autoencoder from MLflow registry,
builds combined fake sensor data, scores with both models, and prints
side-by-side results per parameter.
"""
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
import datetime

sys.path.insert(0, str(Path(__file__).parent))
from app.services.anomaly_feature_service import build_anomaly_features, get_feature_columns
from app.services.lstm_autoencoder_service import LSTMAutoencoder
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

SPIKE = {
    "soil_moisture": 95.0,
    "soil_temp":     58.0,
    "ambient_temp":  55.0,
    "humidity":      99.0,
    "pressure":      800.0,
    "lux":           119000.0,
}

N_NORMAL = 50  # normal readings before the spike


def _make_fake_df(parameter: str) -> pd.DataFrame:
    """50 normal readings + 1 spike at the end."""
    rng = np.random.default_rng(42)
    lo, hi = FAKE_RANGES[parameter]
    values = rng.uniform(lo, hi, N_NORMAL).tolist() + [SPIKE[parameter]]
    base = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    timestamps = [base + datetime.timedelta(minutes=5 * i) for i in range(N_NORMAL + 1)]
    return pd.DataFrame({
        "id":        range(N_NORMAL + 1),
        "field_id":  "test_field_001",
        "device_id": "device_001",
        "timestamp": timestamps,
        "parameter": parameter,
        "value":     values,
    })


# ── Model loaders ─────────────────────────────────────────────────────────────

def _load_isolation_forest() -> dict[str, object]:
    setup_mlflow()
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions("name='Anomaly_Isolation_Forest'")
    staging  = [v for v in versions if v.tags.get("stage") == "Staging"]
    if not staging:
        raise RuntimeError("No Anomaly_Isolation_Forest model tagged stage=Staging in MLflow.")
    mv     = max(staging, key=lambda v: int(v.version))
    run_id = mv.run_id
    print(f"  IF  — run {run_id[:8]}, v{mv.version}")
    models = {}
    for param in PARAMETERS:
        path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=f"models/isolation_forest_{param}.joblib"
        )
        models[param] = joblib.load(path)
    return models


def _load_autoencoder():
    """Returns (model, scaler, threshold, seq_len, feature_columns)."""
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions("name='Anomaly_LSTM_Autoencoder'")
    staging  = [v for v in versions if v.tags.get("stage") == "Staging"]
    if not staging:
        raise RuntimeError("No Anomaly_LSTM_Autoencoder model tagged stage=Staging in MLflow.")
    mv     = max(staging, key=lambda v: int(v.version))
    run_id = mv.run_id
    print(f"  AE  — run {run_id[:8]}, v{mv.version}")

    model_path  = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="models/lstm_autoencoder.pt"
    )
    scaler_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="models/lstm_autoencoder_scaler.joblib"
    )

    ckpt      = torch.load(model_path, map_location="cpu", weights_only=False)
    model     = LSTMAutoencoder(input_size=ckpt["input_size"], hidden_size=ckpt["hidden_size"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, joblib.load(scaler_path), ckpt["reconstruction_threshold"], \
           ckpt["sequence_length"], ckpt["feature_columns"]


# ── AE scoring ────────────────────────────────────────────────────────────────

def _ae_score_per_parameter(model, scaler, threshold, seq_len, ae_features, df_feat_all):
    """
    Returns dict[param] = {n_anom, total_seqs, last_seq_anom, last_err}
    Sequences are built per (device_id, parameter) group, matching training.
    """
    results = {}
    grouped = df_feat_all.groupby("parameter")
    for param, group_df in grouped:
        group_df = group_df.sort_values("timestamp").copy()
        X = group_df[ae_features].values.astype(np.float32)
        X_scaled = scaler.transform(X)

        n = len(X_scaled)
        if n < seq_len:
            results[param] = {"n_anom": 0, "total_seqs": 0,
                               "last_seq_anom": False, "last_err": 0.0}
            continue

        seqs = np.stack([X_scaled[i:i + seq_len] for i in range(n - seq_len + 1)])
        t    = torch.tensor(seqs, dtype=torch.float32)
        with torch.no_grad():
            recon  = model(t)
            errors = torch.mean((recon - t) ** 2, dim=(1, 2)).numpy()

        n_anom     = int((errors > threshold).sum())
        last_err   = float(errors[-1])
        results[param] = {
            "n_anom":        n_anom,
            "total_seqs":    len(errors),
            "last_seq_anom": last_err > threshold,
            "last_err":      last_err,
        }
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Downloading models from MLflow …")
    setup_mlflow()
    if_models              = _load_isolation_forest()
    ae_model, ae_scaler, ae_thr, ae_seq_len, ae_features = _load_autoencoder()

    feature_columns = get_feature_columns()

    # Build combined feature DataFrame for all parameters at once
    df_all  = pd.concat([_make_fake_df(p) for p in PARAMETERS], ignore_index=True)
    df_feat = build_anomaly_features(df_all)

    # Score all parameters with LSTM AE
    ae_results = _ae_score_per_parameter(
        ae_model, ae_scaler, ae_thr, ae_seq_len, ae_features, df_feat
    )

    print(f"\nFake data: {N_NORMAL} normal + 1 spike per parameter\n")
    print(f"{'':17s}  {'Isolation Forest':30s}  {'LSTM Autoencoder':30s}")
    print(f"{'Parameter':17s}  {'anomalies  spike  score':30s}  {'anom_seqs  spike  recon_err':30s}")
    print("─" * 85)

    for param in PARAMETERS:
        df_p  = df_feat[df_feat["parameter"] == param]
        X     = df_p[feature_columns].values

        # Isolation Forest
        preds      = if_models[param].predict(X)
        scores     = -if_models[param].decision_function(X)
        n_if       = int((preds == -1).sum())
        if_spike   = "ANOMALY" if preds[-1] == -1 else "normal "
        if_score   = scores[-1]

        # LSTM Autoencoder
        ae         = ae_results.get(param, {})
        ae_spike   = "ANOMALY" if ae.get("last_seq_anom") else "normal "
        ae_n       = ae.get("n_anom", 0)
        ae_total   = ae.get("total_seqs", 0)
        ae_err     = ae.get("last_err", 0.0)

        print(f"  {param:15s}  "
              f"{n_if:3d}/{N_NORMAL+1}  {if_spike}  {if_score:+.4f}      "
              f"{ae_n:3d}/{ae_total:3d}  {ae_spike}  {ae_err:.6f}")

    print("\nAnomaly Detection (IF + AE) inference — OK")
