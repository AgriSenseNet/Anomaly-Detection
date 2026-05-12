from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "thisen",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="crop_yield_train_dag",
    description="Monthly crop yield LSTM+XGBoost training — results logged to MLflow",
    default_args=default_args,
    start_date=datetime(2026, 5, 1),
    schedule="@monthly",
    catchup=False,
    tags=["crop-yield", "training", "mlflow"],
) as dag:

    train = BashOperator(
        task_id="train_crop_yield_model",
        bash_command="cd /opt/airflow/crop_yield && python train.py",
        env={"PYTHONPATH": "/opt/airflow/crop_yield"},
        execution_timeout=timedelta(hours=3),
    )
