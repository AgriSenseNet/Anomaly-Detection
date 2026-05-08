from datetime import datetime, timezone


def build_demo_satellite_result(field_id: str):
    """
    Build one demo satellite result.

    This is not real Sentinel-2 data yet.
    We use this first to test:
    - Python script
    - InfluxDB write
    - Airflow DAG later
    """

    now = datetime.now(timezone.utc)

    return {
        "field_id": field_id,
        "overpass_date": now.date().isoformat(),
        "overpass_timestamp": now,
        "ndvi_mean": 0.62,
        "ndwi_mean": 0.18,
        "evi_mean": 0.48,
        "low_ndvi_zone_percent": 12.5,
        "cloud_cover": 8.0,
        "is_stale": False,
    }