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


    train_isolation_forest_task = BashOperator(
        task_id="train_isolation_forest_task",
        bash_command="python /opt/airflow/scripts/run_train_isolation_forest.py",
    )    

    score_anomalies_task = BashOperator(
        task_id="score_anomalies_task",
        bash_command="python /opt/airflow/scripts/run_score_anomalies.py",
    )


    validate_sensor_data_task >> train_isolation_forest_task >> score_anomalies_task
