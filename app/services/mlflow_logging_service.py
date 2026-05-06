import os

from pathlib import Path
import mlflow
from dotenv import load_dotenv

load_dotenv(".env.local")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOCAL_MLFLOW_FOLDER = PROJECT_ROOT / "mlruns"
EXPERIMENT_NAME = "Sensor_Anomaly_Detection"


def setup_mlflow():
    """
    Set up local MLflow tracking.

    This will save MLflow runs inside:
    agri/mlruns/
    """

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"file:{LOCAL_MLFLOW_FOLDER}",
    )    

    if tracking_uri.startswith("file:"):
        LOCAL_MLFLOW_FOLDER.mkdir(parents=True, exist_ok=True)


    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    return EXPERIMENT_NAME