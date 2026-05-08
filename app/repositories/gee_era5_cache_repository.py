import json
from datetime import datetime, timedelta, timezone

from app.config.db import get_db_connection

def insert_gee_era5_cache(field_id, gee_data):
    """
    Save Google Earth Engine ERA5-Land derived features
    into PostgreSQL external_api_cache table.

    We use the same table used by Open-Meteo / NASA POWER before.
    """

    conn = get_db_connection()
    cur = conn.cursor()

    expires_at = datetime.now(timezone.utc) + timedelta(hours=24)  # Convert to UTC+24

    query = """
        INSERT INTO external_api_cache (source, field_id, data, expires_at)
        VALUES (%s, %s, %s::jsonb, %s)
    """

    cur.execute(
        query,
        (
            "gee_era5_land",
            field_id,
            json.dumps(gee_data),
            expires_at,
        ),
    )

    conn.commit()
    cur.close()
    conn.close()

def get_latest_gee_era5_cache(field_id):
    """
    Get latest cached GEE ERA5-Land features for a field.
    This is used as fallback if GEE API fails.
    """

    conn = get_db_connection()
    cur = conn.cursor()

    query = """
        SELECT data
        FROM external_api_cache
        WHERE source = %s AND field_id = %s
        ORDER BY timestamp DESC
        LIMIT 1
    """

    cur.execute(
        query,
        (
            "gee_era5_land",
            field_id,
        ),
    )

    row = cur.fetchone()

    cur.close()
    conn.close()

    if row:
        return row[0]

    return None   