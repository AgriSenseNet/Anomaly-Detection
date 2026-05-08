import os
from dotenv import load_dotenv

from app.repositories.satellite_indices_repository import write_satellite_indices
from app.services.satellite_demo_service import build_demo_satellite_result

load_dotenv(".env.local")


def process_field_satellite_indices(field: dict):
    """
    Process satellite indices for one field.

    First version:
    - uses demo data
    - writes to InfluxDB

    Later version:
    - search Sentinel-2
    - download/read bands
    - calculate NDVI/NDWI/EVI
    - write to InfluxDB
    """

    demo_mode = os.getenv("SENTINEL_DEMO_MODE", "true").lower() == "true"

    if demo_mode:
        result = build_demo_satellite_result(field["field_id"])
        write_satellite_indices(result)
        return result

    raise NotImplementedError(
        "Real Sentinel-2 mode is not implemented yet. Keep SENTINEL_DEMO_MODE=true for now."
    )