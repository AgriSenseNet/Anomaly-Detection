from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from app.config.db import get_db_connection
from app.services.gee_era5_service import fetch_and_cache_gee_era5_data
from app.services.open_meteo_service import fetch_and_cache_weather
from app.services.nasa_power_service import fetch_and_cache_nasa_power

default_args = {
    "owner": "yasindu",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def _load_fields():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT field_id, lat, lon, planting_date FROM fields;")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def fetch_weather_task():
    for field_id, lat, lon in _load_fields():
        result = fetch_and_cache_weather(field_id, lat, lon)
        print(f"Open-Meteo [{field_id}]: {result.get('message')}")
        if not result.get("success"):
            raise Exception(f"Open-Meteo failed for {field_id}: {result.get('message')}")


def fetch_nasa_power_task():
    for field_id, lat, lon in _load_fields():
        result = fetch_and_cache_nasa_power(field_id=field_id, lat=lat, lon=lon)
        print(f"NASA POWER [{field_id}]: {result.get('message')}")
        if not result.get("success"):
            raise Exception(f"NASA POWER failed for {field_id}: {result.get('message')}")

def fetch_gee_era5_task():
    for field_id, lat, lon, planting_date  in _load_fields():
        result = fetch_and_cache_gee_era5_data(
            field_id=field_id,
            lat=lat,
            lon=lon,
            planting_date=planting_date,
        )
        print(f"GEE ERA5-Land [{field_id}]: {result.get('message')}")
        if not result.get("success"):
            raise Exception(f"GEE ERA5-Land failed for {field_id}: {result.get('message')}")


with DAG(
    dag_id="external_api_dag",
    description="Fetch Open-Meteo and NASA POWER data for all fields and cache in PostgreSQL",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule_interval="@hourly",
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

    gee_era5 = PythonOperator(
        task_id="fetch_gee_era5_task",
        python_callable=fetch_gee_era5_task,
    )

    open_meteo >> gee_era5
