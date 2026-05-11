import os
from datetime import datetime, timedelta, date

import ee
from dotenv import load_dotenv

from app.repositories.gee_era5_cache_repository import (
    insert_gee_era5_cache,
    get_latest_gee_era5_cache,
)

load_dotenv(".env.local")


ERA5_LAND_COLLECTION = "ECMWF/ERA5_LAND/HOURLY"


def initialize_gee():
    """
    Initialize Google Earth Engine.

    First time only, run this in terminal:
        earthengine authenticate
    """

    project_id = os.getenv("GEE_PROJECT_ID")
    service_account = os.getenv("GEE_SERVICE_ACCOUNT")
    key_path = os.getenv("GEE_SERVICE_ACCOUNT_KEY_PATH")

    if not project_id:
        raise RuntimeError("GEE_PROJECT_ID is missing in .env.local")

    if not service_account:
        raise RuntimeError("GEE_SERVICE_ACCOUNT is missing in .env.local")

    if not key_path:
        raise RuntimeError("GEE_SERVICE_ACCOUNT_KEY_PATH is missing in .env.local")

    credentials = ee.ServiceAccountCredentials(
        service_account,
        key_path,
    )

    ee.Initialize(
        credentials,
        project=project_id,
    )
    
def get_mean_value_from_image(image, geometry, band_name):
    """
    Get mean value for one ERA5-Land band around the field area.
    """

    value = image.select(band_name).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=11132,
        maxPixels=1e9,
    ).get(band_name)

    return value


def image_to_raw_record(image, geometry):
    """
    Convert one hourly ERA5-Land image into one raw JSON record.

    No feature calculation is done here.
    This only collects values needed by the teammate later.
    """

    temperature_2m = get_mean_value_from_image(
        image,
        geometry,
        "temperature_2m",
    )

    dewpoint_temperature_2m = get_mean_value_from_image(
        image,
        geometry,
        "dewpoint_temperature_2m",
    )

    soil_temperature_level_1 = get_mean_value_from_image(
        image,
        geometry,
        "soil_temperature_level_1",
    )

    volumetric_soil_water_layer_1 = get_mean_value_from_image(
        image,
        geometry,
        "volumetric_soil_water_layer_1",
    )

    surface_solar_radiation_downwards_hourly = get_mean_value_from_image(
        image,
        geometry,
        "surface_solar_radiation_downwards_hourly",
    )

    return ee.Feature(
        None,
        {
            "datetime": image.date().format("YYYY-MM-dd HH:mm:ss"),
            "date": image.date().format("YYYY-MM-dd"),

            # Temperature values are in Kelvin.
            # Teammate can convert to Celsius later.
            "temperature_2m_k": temperature_2m,

            # Dewpoint is used with temperature to calculate humidity later.
            "dewpoint_temperature_2m_k": dewpoint_temperature_2m,

            # Soil temperature level 1 is in Kelvin.
            "soil_temperature_level_1_k": soil_temperature_level_1,

            # Volumetric soil water layer 1.
            # Teammate can use this for soil_moisture_7d_mean/std.
            "volumetric_soil_water_layer_1": volumetric_soil_water_layer_1,

            # Solar radiation for the hour.
            # Teammate can use this for solar accumulation.
            "surface_solar_radiation_downwards_hourly_j_m2": (
                surface_solar_radiation_downwards_hourly
            ),
        },
    )


def fetch_gee_era5_raw_records(lat, lon, start_date, end_date):
    """
    Fetch raw ERA5-Land hourly data from Google Earth Engine.

    This returns hourly records only.
    It does not calculate averages, GDD, humidity, or solar accumulation.
    """

    initialize_gee()

    point = ee.Geometry.Point([float(lon), float(lat)])

    # Small buffer around the field point.
    # ERA5-Land has coarse pixels, so this is enough for project use.
    geometry = point.buffer(1000)

    collection = (
        ee.ImageCollection(ERA5_LAND_COLLECTION)
        .filterDate(start_date.isoformat(), end_date.isoformat())
        .filterBounds(point)
        .select(
            [
                "temperature_2m",
                "dewpoint_temperature_2m",
                "soil_temperature_level_1",
                "volumetric_soil_water_layer_1",
                "surface_solar_radiation_downwards_hourly",
            ]
        )
    )

    feature_collection = collection.map(
        lambda image: image_to_raw_record(image, geometry)
    )

    result = feature_collection.getInfo()

    raw_records = []

    for feature in result.get("features", []):
        raw_records.append(feature.get("properties", {}))

    return raw_records


def build_gee_era5_json_body(field_id, lat, lon, planting_date, ndvi_mean=None):
    """
    Build the JSON body that will be saved in external_api_cache.

    This JSON contains raw GEE ERA5-Land data only.
    Teammate will calculate derived features later.
    """

    if isinstance(planting_date, str):
        planting_date = datetime.strptime(planting_date, "%Y-%m-%d").date()

    if isinstance(planting_date, datetime):
        planting_date = planting_date.date()

    # Use 7 days behind today to avoid missing latest ERA5-Land data.
    end_date = date.today() - timedelta(days=7)

    if planting_date > end_date:
        raise ValueError("planting_date is after available GEE end date window.")

    # GEE filterDate end is exclusive, so add 1 day.
    fetch_end_date = end_date + timedelta(days=1)

    raw_records = fetch_gee_era5_raw_records(
        lat=lat,
        lon=lon,
        start_date=planting_date,
        end_date=fetch_end_date,
    )

    json_body = {
        "field_id": field_id,
        "source": "gee_era5_land",
        "lat": float(lat),
        "lon": float(lon),
        "planting_date": planting_date.isoformat(),
        "start_date": planting_date.isoformat(),
        "end_date": end_date.isoformat(),

        # This comes from Sentinel-2 DAG later.
        # For now it can be demo value or None.
        "ndvi_mean": ndvi_mean,

        # These are the raw records your teammate needs.
        "records": raw_records,

        # Simple note for project clarity.
        "bands": {
            "temperature_2m_k": "Used later for ambient_temp_7d_mean and GDD",
            "dewpoint_temperature_2m_k": "Used later with temperature_2m_k to calculate humidity_7d_mean",
            "soil_temperature_level_1_k": "Used later for soil_temp_7d_mean",
            "volumetric_soil_water_layer_1": "Used later for soil_moisture_7d_mean and soil_moisture_7d_std",
            "surface_solar_radiation_downwards_hourly_j_m2": "Used later for solar_lux_accumulated_since_planting",
            "ndvi_mean": "Joined from Sentinel-2 DAG",
        },
    }

    return json_body


def fetch_and_cache_gee_era5_data(field_id, lat, lon, planting_date, ndvi_mean=None):
    """
    Fetch raw GEE ERA5-Land data and save it to external_api_cache.

    If GEE fails, return latest cached data.
    """

    try:
        gee_json = build_gee_era5_json_body(
            field_id=field_id,
            lat=lat,
            lon=lon,
            planting_date=planting_date,
            ndvi_mean=ndvi_mean,
        )

        insert_gee_era5_cache(field_id, gee_json)

        return {
            "success": True,
            "source": "api",
            "data": gee_json,
            "message": "Google Earth Engine ERA5-Land raw data fetched and cached successfully.",
        }

    except Exception as e:
        cached_data = get_latest_gee_era5_cache(field_id)

        if cached_data:
            return {
                "success": True,
                "source": "cache",
                "data": cached_data,
                "message": f"GEE API failed, returned latest cached data. Error: {str(e)}",
            }

        return {
            "success": False,
            "source": None,
            "data": None,
            "message": f"Failed to fetch GEE ERA5-Land data and no cache found. Error: {str(e)}",
        }