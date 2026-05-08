import os
from pathlib import Path

from dotenv import load_dotenv
from influxdb_client import InfluxDBClient

ENV_PATH = Path("/opt/airflow/.env.local")

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv(".env.local")



def get_influxdb_client():
    """
    Create and return an InfluxDB client.
    Keep InfluxDB connection details in one place.
    """

    url = os.getenv("INFLUXDB_URL")
    token = os.getenv("INFLUXDB_TOKEN")
    org = os.getenv("INFLUXDB_ORG")

    if not url or not token or not org:
        raise RuntimeError(
            "InfluxDB settings are missing. Check INFLUXDB_URL, INFLUXDB_TOKEN, and INFLUXDB_ORG in .env.local"
        )

    return InfluxDBClient(
        url=url,
        token=token,
        org=org,
    )


def get_influxdb_org():
    org = os.getenv("INFLUXDB_ORG")

    if not org:
        raise RuntimeError("INFLUXDB_ORG is missing in .env.local")

    return org


def get_influxdb_bucket():
    bucket = os.getenv("INFLUXDB_BUCKET")

    if not bucket:
        raise RuntimeError("INFLUXDB_BUCKET is missing in .env.local")

    return bucket