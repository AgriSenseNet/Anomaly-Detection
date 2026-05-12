"""
Sprint 2 – Crop Yield Prediction Training Pipeline
  • LSTM (PyTorch) tuned with Optuna, tracked in MLflow
  • XGBoost tuned with Optuna, tracked in MLflow
  • Better model registered as 'Yield_Prediction' (tag: Staging)
  • Target: RMSE < 15 % of mean yield, R² > 0.78
"""

import lzma
import math
import mlflow
import mlflow.pytorch
import mlflow.xgboost
import numpy as np
import optuna
import pickle
import pathlib
import tempfile
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset

import config
import data_loader
import feature_engineering
from model import CropYieldLSTM

optuna.logging.set_verbosity(optuna.logging.WARNING)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_OPTUNA_TRIALS = 50
EARLY_STOP_PATIENCE = 10
MAX_EPOCHS = 100


# ── metrics ─────────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def rmse_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return rmse(y_true, y_pred) / np.mean(y_true) * 100.0


# ── LSTM training ────────────────────────────────────────────────────────────

def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_lstm_trial(trial: optuna.Trial, splits: dict) -> float:
    hidden  = trial.suggest_categorical("hidden_size", [64, 128, 256])
    lr      = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    batch   = trial.suggest_categorical("batch_size", [32, 64, 128])

    model = CropYieldLSTM(
        input_size=splits["X_lstm_train"].shape[2],
        hidden_size=hidden,
        num_layers=2,
        dropout=dropout,
    ).to(DEVICE)

    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_loader = _make_loader(splits["X_lstm_train"], splits["y_train"], batch, shuffle=True)
    val_loader   = _make_loader(splits["X_lstm_val"],   splits["y_val"],   batch, shuffle=False)

    best_val_rmse = float("inf")
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        model.eval()
        preds, actuals = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.append(model(xb.to(DEVICE)).cpu().numpy())
                actuals.append(yb.numpy())
        val_rmse = rmse(np.concatenate(actuals), np.concatenate(preds))

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                break

        trial.report(val_rmse, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    with mlflow.start_run(run_name=f"lstm_trial_{trial.number}", nested=True):
        mlflow.log_params(trial.params)
        mlflow.log_metric("val_rmse", best_val_rmse)

    return best_val_rmse


def train_best_lstm(best_params: dict, splits: dict) -> tuple[CropYieldLSTM, dict]:
    model = CropYieldLSTM(
        input_size=splits["X_lstm_train"].shape[2],
        hidden_size=best_params["hidden_size"],
        num_layers=2,
        dropout=best_params["dropout"],
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    loss_fn = nn.MSELoss()

    X_tr = np.concatenate([splits["X_lstm_train"], splits["X_lstm_val"]])
    y_tr = np.concatenate([splits["y_train"],      splits["y_val"]])
    train_loader = _make_loader(X_tr, y_tr, best_params["batch_size"], shuffle=True)

    for _ in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE)).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        test_loader = _make_loader(splits["X_lstm_test"], splits["y_test"], 256, shuffle=False)
        preds, actuals = [], []
        for xb, yb in test_loader:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
            actuals.append(yb.numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(actuals)
    metrics = {
        "rmse":     rmse(y_true, y_pred),
        "rmse_pct": rmse_pct(y_true, y_pred),
        "r2":       r2_score(y_true, y_pred),
    }
    return model, metrics


# ── XGBoost training ─────────────────────────────────────────────────────────

def train_xgb_trial(trial: optuna.Trial, splits: dict) -> float:
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "tree_method": "hist",
        "device":      "cuda" if torch.cuda.is_available() else "cpu",
    }
    model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
    model.fit(
        splits["X_xgb_train"], splits["y_train"],
        eval_set=[(splits["X_xgb_val"], splits["y_val"])],
        verbose=False,
    )
    val_preds = model.predict(splits["X_xgb_val"])
    val_rmse = rmse(splits["y_val"], val_preds)

    with mlflow.start_run(run_name=f"xgb_trial_{trial.number}", nested=True):
        mlflow.log_params(trial.params)
        mlflow.log_metric("val_rmse", val_rmse)

    return val_rmse


def train_best_xgb(best_params: dict, splits: dict) -> tuple[xgb.XGBRegressor, dict]:
    X_tr = np.concatenate([splits["X_xgb_train"], splits["X_xgb_val"]])
    y_tr = np.concatenate([splits["y_train"],      splits["y_val"]])

    params = {**best_params,
              "tree_method": "hist",
              "device": "cuda" if torch.cuda.is_available() else "cpu",
              "random_state": 42, "verbosity": 0}
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(splits["X_xgb_test"])
    y_true = splits["y_test"]
    metrics = {
        "rmse":     rmse(y_true, y_pred),
        "rmse_pct": rmse_pct(y_true, y_pred),
        "r2":       r2_score(y_true, y_pred),
    }
    return model, metrics


# ── orchestration ────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data …")
    sensor_df, yield_df = data_loader.load_all()

    print("Building feature matrices …")
    data = feature_engineering.build_datasets(sensor_df, yield_df)
    splits = feature_engineering.train_val_test_split(data)

    mean_yield = float(np.mean(splits["y_train"]))
    print(f"  Train={len(splits['y_train'])}  Val={len(splits['y_val'])}  Test={len(splits['y_test'])}")
    print(f"  Mean yield (train): {mean_yield:.1f} kg/ha  |  device: {DEVICE}")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Crop_Yield_Prediction")
    print(f"  MLflow tracking: {config.MLFLOW_TRACKING_URI}")

    # ── LSTM tuning ───────────────────────────────────────────────────────
    print(f"\nTuning LSTM with Optuna ({N_OPTUNA_TRIALS} trials) …")
    with mlflow.start_run(run_name="LSTM") as lstm_run:
        mlflow.set_tag("device", str(DEVICE))
        lstm_study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        )
        lstm_study.optimize(
            lambda t: train_lstm_trial(t, splits),
            n_trials=N_OPTUNA_TRIALS,
            show_progress_bar=True,
        )
        best_lstm_params = lstm_study.best_params
        print(f"  Best val RMSE: {lstm_study.best_value:.2f}")
        mlflow.log_metric("optuna_best_val_rmse", lstm_study.best_value)

        print("Training final LSTM on train+val …")
        lstm_model, lstm_metrics = train_best_lstm(best_lstm_params, splits)
        mlflow.log_params(best_lstm_params)
        mlflow.log_metrics(lstm_metrics)
        with tempfile.TemporaryDirectory() as _d:
            _p = str(pathlib.Path(_d) / "model")
            mlflow.pytorch.save_model(lstm_model, _p)
            mlflow.log_artifacts(_p, artifact_path="model")
        lstm_run_id = lstm_run.info.run_id

    print(f"  LSTM  → RMSE={lstm_metrics['rmse']:.1f}  "
          f"RMSE%={lstm_metrics['rmse_pct']:.1f}%  R²={lstm_metrics['r2']:.4f}")

    # ── XGBoost tuning ────────────────────────────────────────────────────
    print(f"\nTuning XGBoost with Optuna ({N_OPTUNA_TRIALS} trials) …")
    with mlflow.start_run(run_name="XGBoost") as xgb_run:
        mlflow.set_tag("device", str(DEVICE))
        xgb_study = optuna.create_study(direction="minimize")
        xgb_study.optimize(
            lambda t: train_xgb_trial(t, splits),
            n_trials=N_OPTUNA_TRIALS,
            show_progress_bar=True,
        )
        best_xgb_params = xgb_study.best_params
        print(f"  Best val RMSE: {xgb_study.best_value:.2f}")
        mlflow.log_metric("optuna_best_val_rmse", xgb_study.best_value)

        print("Training final XGBoost on train+val …")
        xgb_model, xgb_metrics = train_best_xgb(best_xgb_params, splits)
        mlflow.log_params(best_xgb_params)
        mlflow.log_metrics(xgb_metrics)
        with tempfile.TemporaryDirectory() as _d:
            _p = pathlib.Path(_d) / "model"
            mlflow.xgboost.save_model(xgb_model, str(_p))
            # Compress model.ubj with lzma: 22 MB → ~6 MB to stay under proxy timeout
            _ubj = _p / "model.ubj"
            _xz = _p / "model.ubj.xz"
            _xz.write_bytes(lzma.compress(_ubj.read_bytes(), preset=6))
            _ubj.unlink()
            _mlm = _p / "MLmodel"
            _mlm.write_text(_mlm.read_text().replace("model.ubj", "model.ubj.xz"))
            mlflow.log_artifacts(str(_p), artifact_path="model")
        xgb_run_id = xgb_run.info.run_id

    print(f"  XGBoost→ RMSE={xgb_metrics['rmse']:.1f}  "
          f"RMSE%={xgb_metrics['rmse_pct']:.1f}%  R²={xgb_metrics['r2']:.4f}")

    # ── compare and register ──────────────────────────────────────────────
    if lstm_metrics["rmse"] <= xgb_metrics["rmse"]:
        winner_name, winner_run_id, winner_metrics = "LSTM",    lstm_run_id, lstm_metrics
        winner_uri = f"runs:/{lstm_run_id}/model"
    else:
        winner_name, winner_run_id, winner_metrics = "XGBoost", xgb_run_id, xgb_metrics
        winner_uri = f"runs:/{xgb_run_id}/model"

    print(f"\nWinner: {winner_name}")

    client = mlflow.tracking.MlflowClient()
    try:
        client.create_registered_model("Yield_Prediction")
    except mlflow.exceptions.MlflowException:
        pass

    mv = client.create_model_version(
        name="Yield_Prediction",
        source=winner_uri,
        run_id=winner_run_id,
    )
    client.set_model_version_tag(mv.name, mv.version, "stage", "Staging")

    print(f"Registered 'Yield_Prediction' v{mv.version} (tag: Staging)")

    # Save scaler + crop list so inference.py can reproduce the same feature transform
    _bundle_path = pathlib.Path(tempfile.mkdtemp()) / "scaler.pkl"
    _bundle_path.write_bytes(
        pickle.dumps({"scaler": splits["scaler"], "crops": splits["crops"]})
    )
    with mlflow.start_run(run_id=winner_run_id):
        mlflow.log_artifact(str(_bundle_path))
    _bundle_path.unlink()
    print("Saved scaler.pkl to winning MLflow run.")

    # ── target checks ─────────────────────────────────────────────────────
    rmse_ok = winner_metrics["rmse_pct"] < 15.0
    r2_ok   = winner_metrics["r2"] > 0.78
    print(f"\nTarget checks:")
    print(f"  RMSE < 15% of mean: {'PASS' if rmse_ok else 'FAIL'}  "
          f"({winner_metrics['rmse_pct']:.1f}%)")
    print(f"  R² > 0.78         : {'PASS' if r2_ok   else 'FAIL'}  "
          f"({winner_metrics['r2']:.4f})")


if __name__ == "__main__":
    main()
