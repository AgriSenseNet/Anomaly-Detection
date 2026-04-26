from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "yasindu",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

def placeholder_model_retraining_task():
    """
    Placeholder for Sprint 2 model retraining.
    For Sprint 1, this only proves the DAG appears in Airflow without errors.
    """
    print("Model retraining DAG placeholder is working.")
    print("Actual model retraining will be implemented in Sprint 2.")

with DAG(
    dag_id="model_retraining_dag",
    description="Placeholder DAG for Sprint 2 model retraining",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["c2", "ml", "placeholder"],
) as dag:

    placeholder_task = PythonOperator(
        task_id="placeholder_model_retraining_task",
        python_callable=placeholder_model_retraining_task,
    )
