from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Import your already completed service functions
from app.services.open_meteo_service import fetch_and_cache_weather
from app.services.nasa_power_service import fetch_and_cache_nasa_power

# Sample field data for testing
# Later, this can come from the fields table.
FIELD_ID = "field_001"
LAT = 41.8781
LON = -93.0977

default_args = {
    "owner": "yasindu",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

def fetch_weather_task():
    """    Airflow task for Open-Meteo.
    This calls your existing Open-Meteo service code.
    """
    result = fetch_and_cache_weather(FIELD_ID, LAT, LON)

    print("NASA POWER result:")
    print(result)

    if not result.get("success"):
        raise Exception(result.get("message", "Open-Meteo task failed"))

    return result

    
def fetch_nasa_power_task():
    """
    Airflow task for NASA POWER.
    This calls your existing NASA POWER service code.
    """
    result = fetch_and_cache_nasa_power(
        field_id=FIELD_ID,
        lat=LAT,
        lon=LON,
    )

    print("NASA POWER result:")
    print(result)

    if not result.get("success"):
        raise Exception(result.get("message", "NASA POWER task failed"))

    return result

with DAG(
    dag_id="external_api_dag",
    description="Fetch Open-Meteo and NASA POWER data and cache them in PostgreSQL",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@hourly",
    catchup=False,
    tags=["c2", "external-api", "weather"],
) as dag:

    open_meteo = PythonOperator(
        task_id="fetch_weather_task",
        python_callable=fetch_weather_task,
    )

    nasa_power = PythonOperator(
        task_id="fetch_nasa_power_task",
        python_callable=fetch_nasa_power_task,
    )

    open_meteo >> nasa_power