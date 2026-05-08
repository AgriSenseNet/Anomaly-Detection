from app.config.db import get_db_connection
from app.services.gee_era5_service import fetch_and_cache_gee_era5_data


def get_field(field_id):
    """
    Get one field from PostgreSQL fields table.
    """

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT field_id, crop_type, planting_date, lat, lon
        FROM fields
        WHERE field_id = %s;
        """,
        (field_id,),
    )

    row = cur.fetchone()

    cur.close()
    conn.close()

    if not row:
        return None

    return {
        "field_id": row[0],
        "crop_type": row[1],
        "planting_date": row[2],
        "lat": float(row[3]),
        "lon": float(row[4]),
    }


def main():
    field_id = "field_001"

    field = get_field(field_id)

    if not field:
        raise RuntimeError(f"No field found for field_id={field_id}")

    # Later this should come from Sentinel-2 DAG / InfluxDB satellite_indices.
    # For now it can be demo value.
    ndvi_mean = 0.62

    result = fetch_and_cache_gee_era5_data(
        field_id=field["field_id"],
        lat=field["lat"],
        lon=field["lon"],
        planting_date=field["planting_date"],
        ndvi_mean=ndvi_mean,
    )

    print("Success:", result["success"])
    print("Source:", result["source"])
    print("Message:", result["message"])

    if result["data"]:
        print("Field:", result["data"]["field_id"])
        print("Source:", result["data"]["source"])
        print("Start date:", result["data"]["start_date"])
        print("End date:", result["data"]["end_date"])
        print("NDVI mean:", result["data"]["ndvi_mean"])
        print("Record count:", len(result["data"]["records"]))

        if result["data"]["records"]:
            print("\nFirst raw record:")
            print(result["data"]["records"][0])


if __name__ == "__main__":
    main()