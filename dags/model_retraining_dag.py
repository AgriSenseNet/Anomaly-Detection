from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "yasindu",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="model_retraining_dag",
    description="Sprint 2 model retraining pipeline with sensor data validation",
    default_args=default_args,
    start_date=datetime(2026, 4, 1),
    schedule=None,
    catchup=False,
    tags=["sprint2", "validation", "ml"],
) as dag:
    
    validate_sensor_data_task = BashOperator(
        task_id = "validate_sensor_data_task",
        bash_command = "python /opt/airflow/scripts/run_validate_sensor_data.py clean",
    )

    validate_sensor_data_model_task = BashOperator(
        task_id = "validate_sensor_data_model_task",
        bash_command = "echo 'Sensor validation passed. Model training can start next.'",

    )

    validate_sensor_data_task >> validate_sensor_data_model_task
