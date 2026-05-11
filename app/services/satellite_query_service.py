from app.config.influxdb import (
    get_influxdb_client,
    get_influxdb_org,
    get_influxdb_bucket,
)


def get_latest_satellite_indices(field_id: str):
    """
    Read the latest satellite index values for one field from InfluxDB.
    This is useful for the future GET /satellite/ndvi API endpoint.
    """

    query = f'''
    from(bucket: "{get_influxdb_bucket()}")
      |> range(start: -30d)
      |> filter(fn: (r) => r._measurement == "satellite_indices")
      |> filter(fn: (r) => r.field_id == "{field_id}")
      |> filter(fn: (r) =>
          r._field == "ndvi_mean" or
          r._field == "ndwi_mean" or
          r._field == "evi_mean" or
          r._field == "low_ndvi_zone_percent" or
          r._field == "cloud_cover" or
          r._field == "is_stale"
      )
      |> last()
    '''

    with get_influxdb_client() as client:
        query_api = client.query_api()
        tables = query_api.query(query=query, org=get_influxdb_org())

    result = {
        "field_id": field_id,
    }

    overpass_timestamp = None

    for table in tables:
        for record in table.records:
            result[record.get_field()] = record.get_value()
            overpass_timestamp = record.get_time()

    if overpass_timestamp:
        result["overpass_timestamp"] = overpass_timestamp.isoformat()

    return result