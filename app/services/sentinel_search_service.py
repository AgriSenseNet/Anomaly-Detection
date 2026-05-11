from datetime import datetime, timedelta, timezone
import os
import requests
from dotenv import load_dotenv

load_dotenv(".env.local")

CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"


def build_small_bbox_polygon(lat: float, lon: float, buffer_deg: float = 0.01) -> str:
    """
    Build a small polygon around the field location.
    OData expects coordinates in lon lat order.
    """

    min_lon = lon - buffer_deg
    max_lon = lon + buffer_deg
    min_lat = lat - buffer_deg
    max_lat = lat + buffer_deg

    return (
        f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, "
        f"{max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
    )


def search_sentinel2_l2a(
    lat: float,
    lon: float,
    lookback_days: int | None = None,
    max_cloud: int | None = None,
):
    """
    Search latest Sentinel-2 Level-2A product for a field location.
    This only searches metadata. It does not download satellite files.
    """

    lookback_days = lookback_days or int(os.getenv("SENTINEL_LOOKBACK_DAYS", "30"))
    max_cloud = max_cloud or int(os.getenv("SENTINEL_MAX_CLOUD_COVER", "20"))

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days)

    polygon = build_small_bbox_polygon(lat, lon)

    filter_text = (
        "Collection/Name eq 'SENTINEL-2' and "
        "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        "and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A') and "
        f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
        f"and att/OData.CSC.DoubleAttribute/Value le {max_cloud}) and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}') and "
        f"ContentDate/Start gt {start.strftime('%Y-%m-%dT%H:%M:%S.000Z')} and "
        f"ContentDate/Start lt {end.strftime('%Y-%m-%dT%H:%M:%S.000Z')}"
    )

    params = {
        "$filter": filter_text,
        "$orderby": "ContentDate/Start desc",
        "$top": "1",
    }

    response = requests.get(CATALOGUE_URL, params=params, timeout=90)
    response.raise_for_status()

    products = response.json().get("value", [])

    if not products:
        return None

    return products[0]