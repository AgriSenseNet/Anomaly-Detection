from datetime import datetime, timezone

from influxdb_client import Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from app.config.influxdb import (
    get_influxdb_client,
    get_influxdb_org,
    get_influxdb_bucket,
)


def write_satellite_indices(result: dict):
    """
    Write one satellite index result to InfluxDB.

    Measurement:
    - satellite_indices

    Tags:
    - field_id
    - overpass_date
    - source

    Fields:
    - ndvi_mean
    - ndwi_mean
    - evi_mean
    - low_ndvi_zone_percent
    - cloud_cover
    - is_stale
    """

    overpass_time = result.get("overpass_timestamp", datetime.now(timezone.utc))

    point = (
        Point("satellite_indices")
        .tag("field_id", result["field_id"])
        .tag("overpass_date", result["overpass_date"])
        .tag("source", "sentinel_2")
        .field("ndvi_mean", float(result["ndvi_mean"]))
        .field("ndwi_mean", float(result["ndwi_mean"]))
        .field("evi_mean", float(result["evi_mean"]))
        .field("low_ndvi_zone_percent", float(result["low_ndvi_zone_percent"]))
        .field("cloud_cover", float(result.get("cloud_cover", 0)))
        .field("is_stale", bool(result.get("is_stale", False)))
        .time(overpass_time, WritePrecision.NS)
    )

    with get_influxdb_client() as client:
        write_api = client.write_api(write_options=SYNCHRONOUS)
        write_api.write(
            bucket=get_influxdb_bucket(),
            org=get_influxdb_org(),
            record=point,
        )

    return True