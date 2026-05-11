from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="sentinel_dag",
    start_date=datetime(2026, 5, 1),
    schedule=None,
    catchup=False,
    tags=["c2", "sentinel2", "ndvi", "sprint3"],
) as dag:

    run_sentinel2_ndvi_task = BashOperator(
        task_id="run_sentinel2_ndvi_task",
        bash_command="python /opt/airflow/scripts/run_sentinel2_ndvi.py",
    )

    run_sentinel2_ndvi_task