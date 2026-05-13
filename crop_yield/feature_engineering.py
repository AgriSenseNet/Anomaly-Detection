"""
Build training-ready feature matrices from sensor time series + annual yield data.

LSTM input   : (N, window=30, n_sensor+n_crops) — daily sensors + crop one-hot per timestep
XGBoost input: (N, 4*n_sensor + n_crops) — weekly-aggregated sensors + crop one-hot
Target        : yield_kg_per_ha (continuous)

Crop one-hot is added so the model can distinguish per-crop yield baselines
(e.g. cassava ~8700 kg/ha vs sorghum ~700 kg/ha).
Windows are shuffled before splitting to ensure all crop types appear in
every split (sequential split would put entire crops only in train or test).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config


def _sliding_windows(
    feats: np.ndarray,
    yield_kg: float,
    window: int = config.WINDOW_SIZE,
) -> list[tuple[np.ndarray, float]]:
    n = feats.shape[0]
    return [(feats[s : s + window], yield_kg) for s in range(n - window + 1)]


def _weekly_aggregate(window: np.ndarray) -> np.ndarray:
    """(30, n_feat) → (4 * n_feat,) weekly mean vectors."""
    return np.concatenate(
        [w.mean(axis=0) for w in np.array_split(window, 4, axis=0)]
    ).astype(np.float32)


def build_datasets(
    sensor_df: pd.DataFrame,
    yield_df: pd.DataFrame,
    rng_seed: int = 42,
) -> dict:
    """
    Returns a dict with keys:
      X_lstm  : (N, 30, n_sensor+n_crops)  — raw unscaled
      X_xgb   : (N, 4*n_sensor + n_crops)
      y       : (N,)
      crops   : list[str]  — ordered crop names for the one-hot columns
    """
    merged = sensor_df.merge(
        yield_df[["crop_type", "year", "yield_kg_per_ha"]],
        on=["crop_type", "year"],
        how="inner",
    )

    crops = sorted(merged["crop_type"].unique().tolist())
    n_crops = len(crops)
    crop_idx = {c: i for i, c in enumerate(crops)}

    records_lstm, records_xgb, targets = [], [], []

    for (crop, year), group in merged.groupby(["crop_type", "year"]):
        if len(group) < config.WINDOW_SIZE:
            continue
        group = group.sort_values("time")

        sensor_feats = group[config.SENSOR_FEATURES].values.astype(np.float32)
        yield_val = float(group["yield_kg_per_ha"].iloc[0])

        # One-hot vector for this crop (constant across timesteps)
        onehot = np.zeros(n_crops, dtype=np.float32)
        onehot[crop_idx[crop]] = 1.0

        for win, target in _sliding_windows(sensor_feats, yield_val):
            # LSTM: append one-hot to every timestep → (30, n_sensor + n_crops)
            crop_tile = np.tile(onehot, (config.WINDOW_SIZE, 1))
            records_lstm.append(np.concatenate([win, crop_tile], axis=1))

            # XGBoost: weekly aggregated sensors + one-hot appended once
            records_xgb.append(np.concatenate([_weekly_aggregate(win), onehot]))

            targets.append(target)

    X_lstm = np.stack(records_lstm)
    X_xgb  = np.stack(records_xgb)
    y      = np.array(targets, dtype=np.float32)

    # Shuffle so every split sees all crop types
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(len(y))

    return {
        "X_lstm": X_lstm[perm],
        "X_xgb":  X_xgb[perm],
        "y":      y[perm],
        "crops":  crops,
    }


def train_val_test_split(
    data: dict,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
) -> dict:
    n = len(data["y"])
    t = int(n * train_frac)
    v = int(n * (train_frac + val_frac))

    X_lstm_tr = data["X_lstm"][:t]
    n_s, seq, n_feat = X_lstm_tr.shape

    # Scaler fitted on training data only (prevents leakage)
    scaler = StandardScaler()
    scaler.fit(X_lstm_tr.reshape(-1, n_feat))

    def _scale_lstm(X: np.ndarray) -> np.ndarray:
        ns, sq, nf = X.shape
        return scaler.transform(X.reshape(-1, nf)).reshape(ns, sq, nf).astype(np.float32)

    splits = {
        "X_lstm_train": _scale_lstm(data["X_lstm"][:t]),
        "X_lstm_val":   _scale_lstm(data["X_lstm"][t:v]),
        "X_lstm_test":  _scale_lstm(data["X_lstm"][v:]),
        "X_xgb_train":  data["X_xgb"][:t],
        "X_xgb_val":    data["X_xgb"][t:v],
        "X_xgb_test":   data["X_xgb"][v:],
        "y_train":      data["y"][:t],
        "y_val":        data["y"][t:v],
        "y_test":       data["y"][v:],
        "scaler":       scaler,
        "crops":        data["crops"],
    }
    return splits
