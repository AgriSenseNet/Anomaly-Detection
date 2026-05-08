from app.config.db import get_db_connection
from app.services.sentinel_pipeline_service import process_field_satellite_indices


def get_fields():
    """
    Read fields from PostgreSQL.
    Sentinel-2 needs field latitude and longitude.
    """

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT field_id, crop_type, planting_date, lat, lon
        FROM fields
        ORDER BY field_id;
        """
    )

    rows = cur.fetchall()

    cur.close()
    conn.close()

    fields = []

    for row in rows:
        fields.append(
            {
                "field_id": row[0],
                "crop_type": row[1],
                "planting_date": row[2],
                "lat": float(row[3]),
                "lon": float(row[4]),
            }
        )

    return fields


def main():
    print("Sentinel-2 NDVI demo pipeline started")

    fields = get_fields()

    if not fields:
        raise RuntimeError("No fields found. Add at least one field with lat/lon first.")

    for field in fields:
        result = process_field_satellite_indices(field)

        print(
            f"Stored satellite indices for {result['field_id']} | "
            f"NDVI={result['ndvi_mean']} | "
            f"NDWI={result['ndwi_mean']} | "
            f"EVI={result['evi_mean']}"
        )

    print("Sentinel-2 NDVI demo pipeline finished")


if __name__ == "__main__":
    main()