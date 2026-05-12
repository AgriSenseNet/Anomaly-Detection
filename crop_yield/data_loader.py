"""
Training data loader: synthetic daily sensor data + Kaggle annual yield CSV.

Sensor data is generated synthetically for each (crop, year) in the Kaggle
dataset, with values statistically correlated to historical yield so the model
has a learnable signal. This mirrors Sri Lanka climate patterns.

For future inference on live fields, use daily_features in PostgreSQL directly.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import config

# ── Sri Lanka climate profile ─────────────────────────────────────────────────
_MONTHLY_TEMP      = [27.5, 28.2, 29.5, 30.8, 30.2, 28.8, 27.9, 28.0, 28.3, 27.8, 27.2, 27.0]
_MONTHLY_RAIN_MM   = [108,  69,   97,   233,  371,  202,  120,  104,  194,  336,  315,  147]
_MONTHLY_SOLAR_H   = [7.8,  8.1,  8.2,  7.5,  6.0,  5.5,  6.0,  6.2,  6.5,  6.0,  5.8,  7.2]
_total_rain = sum(_MONTHLY_RAIN_MM)
_MONTHLY_RAIN_PROB = [r / _total_rain * 12 for r in _MONTHLY_RAIN_MM]

# (start_month, start_day, season_days, t_base_gdd, soil_ph_range)
_CROP_SEASONS = {
    "Rice, paddy":          (10,  1, 130, 10.0, (5.5, 6.5)),
    "Maize":                ( 3,  1, 110, 10.0, (5.8, 7.0)),
    "Cassava":              ( 1,  1, 300,  8.0, (5.5, 6.8)),
    "Soybeans":             ( 5,  1, 110, 10.0, (6.0, 7.0)),
    "Potatoes":             ( 7,  1, 100, 10.0, (5.5, 6.5)),
    "Sweet potatoes":       ( 6,  1, 120, 10.0, (5.5, 6.5)),
    "Sorghum":              ( 2,  1, 100, 10.0, (5.5, 7.0)),
    "Plantains and others": ( 1,  1, 300,  8.0, (5.5, 7.0)),
}


def _generate_season(crop, year, yield_kg_ha, crop_mean_yield, rng):
    params = _CROP_SEASONS.get(crop)
    if params is None:
        return []

    start_month, start_day, season_len, t_base, ph_range = params
    relative_yield = yield_kg_ha / max(crop_mean_yield, 1.0)
    moisture_bias  = 0.30 + (relative_yield - 1.0) * 0.06
    solar_bias     = 1.0  + (relative_yield - 1.0) * 0.12
    crop_soil_ph   = rng.uniform(*ph_range)

    try:
        season_start = datetime(year, start_month, start_day, tzinfo=timezone.utc)
    except ValueError:
        season_start = datetime(year, start_month, 1, tzinfo=timezone.utc)

    gdd_accum = 0.0
    records = []

    for day_idx in range(season_len):
        ts    = season_start + timedelta(days=day_idx)
        month = ts.month

        ambient_temp = float(_MONTHLY_TEMP[month - 1] + rng.normal(0, 0.6))

        rain_prob = min(_MONTHLY_RAIN_PROB[month - 1] * 0.35, 0.95)
        if rng.random() <= rain_prob:
            monthly_mean = _MONTHLY_RAIN_MM[month - 1] / 30.0
            rain_mm = float(np.clip(rng.gamma(shape=1.2, scale=monthly_mean / 1.2), 0, 120))
        else:
            rain_mm = 0.0

        solar_h      = _MONTHLY_SOLAR_H[month - 1]
        cloud_factor = max(0.3, 1.0 - rain_mm / 60.0)
        solar_lux    = float(solar_h * cloud_factor * rng.uniform(7000, 10000)) * solar_bias

        soil_temp     = float(ambient_temp - rng.uniform(0.5, 2.5))
        soil_moisture = float(np.clip(moisture_bias + rain_mm / 600.0 + rng.normal(0, 0.025), 0.10, 0.55))
        humidity      = float(np.clip(0.72 + rain_mm / 200.0 + rng.normal(0, 0.03), 0.45, 0.98))
        pressure      = float(rng.normal(1010.5, 1.8))

        ra  = solar_lux / 10000.0 * 15.0
        et0 = float(np.clip(0.0023 * ra * (ambient_temp + 17.8) * max(0.1, 1.0 - humidity) ** 0.5, 1.5, 9.0))

        gdd_accum += max(0.0, ambient_temp - t_base)
        soil_ph = float(np.clip(crop_soil_ph + rng.normal(0, 0.05), 4.5, 8.5))

        records.append({
            "time":            ts,
            "crop_type":       crop,
            "year":            year,
            "soil_moisture":   round(soil_moisture, 4),
            "soil_temp":       round(soil_temp, 2),
            "ambient_temp":    round(ambient_temp, 2),
            "humidity":        round(humidity, 4),
            "pressure":        round(pressure, 2),
            "solar_lux_sum":   round(solar_lux, 1),
            "et0_nasa":        round(et0, 3),
            "rainfall_api":    round(rain_mm, 2),
            "gdd_accumulated": round(gdd_accum, 2),
            "soil_ph":         round(soil_ph, 3),
        })

    return records


def load_kaggle_yield() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_DIR / "yield_df_sri_lanka.csv")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "Item":                          "crop_type",
        "Year":                          "year",
        "hg/ha_yield":                   "yield_hg_ha",
        "average_rain_fall_mm_per_year": "annual_rain_mm",
        "pesticides_tonnes":             "pesticides_tonnes",
        "avg_temp":                      "annual_avg_temp",
    })
    df["yield_kg_per_ha"] = df["yield_hg_ha"] * 0.1
    return df[["crop_type", "year", "yield_kg_per_ha", "annual_rain_mm",
               "pesticides_tonnes", "annual_avg_temp"]]


def load_sensor_data() -> pd.DataFrame:
    """
    Generate synthetic daily sensor data for every (crop, year) in the Kaggle yield CSV.
    Seed is fixed so training is reproducible across runs.
    """
    yield_df    = load_kaggle_yield()
    crop_means  = yield_df.groupby("crop_type")["yield_kg_per_ha"].mean().to_dict()
    rng         = np.random.default_rng(seed=42)
    all_records = []

    for _, row in yield_df.iterrows():
        crop      = row["crop_type"]
        year      = int(row["year"])
        yield_val = float(row["yield_kg_per_ha"])
        all_records.extend(_generate_season(crop, year, yield_val, crop_means.get(crop, yield_val), rng))

    if not all_records:
        raise RuntimeError("No crops matched _CROP_SEASONS — check yield CSV crop names.")

    df = pd.DataFrame(all_records)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values(["crop_type", "year", "time"]).reset_index(drop=True)


def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (sensor_df, yield_df) ready for feature_engineering.build_datasets()."""
    return load_sensor_data(), load_kaggle_yield()
