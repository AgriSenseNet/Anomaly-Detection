from app.config.db import get_db_connection

def update_field_soil_ph(field_id, soil_ph):
    conn = get_db_connection()
    cur = conn.cursor()

    query = """
        update fields
        set soil_ph = %s
        where field_id = %s
        """
    cur.execute(
        query,
        (
            soil_ph,
            field_id,
        ),
    )
    conn.commit()
    cur.close()
    conn.close()


def get_field_by_id(field_id):
    conn = get_db_connection()
    cur = conn.cursor()

    query = """
        select field_id, lat, lon, soil_ph
        from fields
        where field_id = %s
        """
    
    cur.execute(
        query,
        (
            field_id,
        ),
    )

    row = cur.fetchone()
    cur.close()
    conn.close()

    if row:
        return {
            "field_id": row[0],
            "lat": row[1],
            "lon": row[2],
            "soil_ph": row[3],
        }
    return None