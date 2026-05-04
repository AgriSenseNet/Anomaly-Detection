from pathlib import Path
import mlflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MLFLOW_FOLDER = PROJECT_ROOT / "mlruns"
EXPERIMENT_NAME = "Sensor_Anomaly_Detection"


def setup_mlflow():
    """
    Set up local MLflow tracking.

    This will save MLflow runs inside:
    agri/mlruns/
    """

    MLFLOW_FOLDER.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"file:{MLFLOW_FOLDER}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    return EXPERIMENT_NAME