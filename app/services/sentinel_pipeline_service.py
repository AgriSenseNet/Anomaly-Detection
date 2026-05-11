# import os
# from dotenv import load_dotenv

# from app.repositories.satellite_indices_repository import write_satellite_indices
# from app.services.satellite_demo_service import build_demo_satellite_result

# load_dotenv(".env.local")


# def process_field_satellite_indices(field: dict):
#     """
#     Process satellite indices for one field.

#     First version:
#     - uses demo data
#     - writes to InfluxDB

#     Later version:
#     - search Sentinel-2
#     - download/read bands
#     - calculate NDVI/NDWI/EVI
#     - write to InfluxDB
#     """

#     demo_mode = os.getenv("SENTINEL_DEMO_MODE", "true").lower() == "true"

#     if demo_mode:
#         result = build_demo_satellite_result(field["field_id"])
#         write_satellite_indices(result)
#         return result

#     raise NotImplementedError(
#         "Real Sentinel-2 mode is not implemented yet. Keep SENTINEL_DEMO_MODE=true for now."
#     )



import os
from datetime import datetime, timezone
from dotenv import load_dotenv

from app.repositories.satellite_indices_repository import write_satellite_indices
from app.services.satellite_demo_service import build_demo_satellite_result
from app.services.copernicus_auth_service import get_copernicus_access_token
from app.services.sentinel_search_service import search_sentinel2_l2a
from app.services.sentinel_download_service import download_product_zip
from app.services.sentinel_band_reader_service import read_required_bands_from_zip
from app.services.satellite_index_service import compute_indices_from_arrays

load_dotenv(".env.local")


def get_cloud_cover(product: dict) -> float:
    """
    Extract cloud cover value from Copernicus product metadata.
    """

    attributes = product.get("Attributes", [])

    for attribute in attributes:
        if attribute.get("Name") == "cloudCover":
            return float(attribute.get("Value", 0))

    return 0.0


def process_field_satellite_indices(field: dict):
    """
    Process satellite indices for one field.

    Demo mode:
    - creates fixed sample NDVI/NDWI/EVI values
    - writes to InfluxDB

    Real mode:
    - gets Copernicus token
    - searches Sentinel-2 L2A product
    - downloads product ZIP
    - reads B02/B03/B04/B08 bands
    - computes NDVI/NDWI/EVI
    - writes real result to InfluxDB
    """

    demo_mode = os.getenv("SENTINEL_DEMO_MODE", "true").lower() == "true"

    if demo_mode:
        result = build_demo_satellite_result(field["field_id"])
        write_satellite_indices(result)
        return result

    lookback_days = int(os.getenv("SENTINEL_LOOKBACK_DAYS", "180"))
    max_cloud = int(os.getenv("SENTINEL_MAX_CLOUD_COVER", "80"))

    product = search_sentinel2_l2a(
        lat=float(field["lat"]),
        lon=float(field["lon"]),
        lookback_days=lookback_days,
        max_cloud=max_cloud,
    )

    if not product:
        raise RuntimeError(
            f"No Sentinel-2 L2A product found for field {field['field_id']}"
        )

    product_id = product["Id"]
    product_name = product["Name"]
    content_date = product["ContentDate"]["Start"]
    cloud_cover = get_cloud_cover(product)

    print(f"Found product for {field['field_id']}: {product_name}")
    print(f"Product ID: {product_id}")
    print(f"Cloud cover: {cloud_cover}")

    token = get_copernicus_access_token()
    zip_path = download_product_zip(product_id, token)

    bands = read_required_bands_from_zip(zip_path)

    indices = compute_indices_from_arrays(
        blue_b02=bands["B02"],
        green_b03=bands["B03"],
        red_b04=bands["B04"],
        nir_b08=bands["B08"],
    )

    overpass_timestamp = datetime.fromisoformat(
        content_date.replace("Z", "+00:00")
    )

    result = {
        "field_id": field["field_id"],
        "overpass_date": overpass_timestamp.date().isoformat(),
        "overpass_timestamp": overpass_timestamp,
        "ndvi_mean": indices["ndvi_mean"],
        "ndwi_mean": indices["ndwi_mean"],
        "evi_mean": indices["evi_mean"],
        "low_ndvi_zone_percent": indices["low_ndvi_zone_percent"],
        "cloud_cover": cloud_cover,
        "is_stale": False,
    }

    write_satellite_indices(result)

    return result