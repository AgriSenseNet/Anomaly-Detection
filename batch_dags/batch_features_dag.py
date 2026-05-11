# batch_features_dag.py
# Runs batch feature pipeline every day at 01:00

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/opt/airflow')

from spark.jobs.batch_features import run_batch

default_args = {
    'owner': 'thilaksan',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='batch_features_dag',
    default_args=default_args,
    description='Daily batch feature aggregation for ML models',
    schedule_interval='0 1 * * *',
    start_date=datetime(2026, 4, 1),
    catchup=False,
    tags=['c2', 'batch', 'features']
) as dag:

    run_batch_task = PythonOperator(
        task_id='run_batch_features',
        python_callable=run_batch,
    )