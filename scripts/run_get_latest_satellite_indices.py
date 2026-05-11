from app.services.satellite_query_service import get_latest_satellite_indices


def main():
    field_id = "field_001"

    result = get_latest_satellite_indices(field_id)

    print("Latest satellite indices")
    print("------------------------")
    print(f"Field ID: {result.get('field_id')}")
    print(f"Overpass timestamp: {result.get('overpass_timestamp')}")
    print(f"NDVI mean: {result.get('ndvi_mean')}")
    print(f"NDWI mean: {result.get('ndwi_mean')}")
    print(f"EVI mean: {result.get('evi_mean')}")
    print(f"Low NDVI zone percent: {result.get('low_ndvi_zone_percent')}")
    print(f"Cloud cover: {result.get('cloud_cover')}")
    print(f"Is stale: {result.get('is_stale')}")


if __name__ == "__main__":
    main()