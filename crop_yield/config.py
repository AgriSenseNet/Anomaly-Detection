import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # finds /opt/airflow/.env when running inside the container

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.cropwise.garden")

WINDOW_SIZE = 30
SENSOR_FEATURES = [
    "soil_moisture",
    "soil_temp",
    "ambient_temp",
    "humidity",
    "pressure",
    "solar_lux_sum",
    "et0_nasa",
    "rainfall_api",
    "gdd_accumulated",
    "soil_ph",
]
TARGET = "yield_kg_per_ha"

DATA_DIR = Path(__file__).parent / "sri_lanka"
