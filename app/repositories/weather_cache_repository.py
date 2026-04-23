import json
from datetime import datetime, timedelta, timezone

from app.config.db import get_db_connection

def insert_weather_cache(field_id, weather_json):
    conn = get_db_connection()
    cur = conn.cursor()

    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
    
    query = """
        insert into external_api_cache (source, field_id, data, expires_at)
        values (%s, %s, %s::jsonb, %s)
        """
    
    cur.execute(
        query,
        (
            "open_meteo",
            field_id,
            json.dumps(weather_json),
            expires_at,
        ),
    )

    conn.commit()
    cur.close()
    conn.close()

def get_latest_weather_cache(field_id):
    conn = get_db_connection()
    cur = conn.cursor()

    query = """
        select data
        from external_api_cache
        where source = %s and field_id = %s
        order by timestamp desc
        limit 1
        """
    
    cur.execute(query,
                (
                    "open_meteo",
                    field_id,
                ))
    row = cur.fetchone()

    cur.close()
    conn.close()

    if row:
        return row[0]
    return None