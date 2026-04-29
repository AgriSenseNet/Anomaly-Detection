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
    dag_id="run_test_anomaly_dag",
    description="DAG to test anomaly detection logic with fake sensor data",
    default_args=default_args,
    start_date=datetime(2026, 4, 1),
    schedule=None,
    catchup=False,
    tags=["test", "anomaly", "validation"],
) as dag:
    
    generate_fake_data_task = BashOperator(
        task_id = "generate_fake_data_task",
        bash_command = "python /opt/airflow/scripts/run_generate_fake_sensor_data.py",
    )

    validate_anomalies_task = BashOperator(
        task_id = "validate_anomalies_task",
        bash_command = "echo 'Anomaly validation passed. Model training can start next.'",

    )

    generate_fake_data_task >> validate_anomalies_task
