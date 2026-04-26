import requests

from app.repositories.fields_repository import (
    update_field_soil_ph,
)

def fetch_soil_ph(lat, lon):
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"

    params = {
        "lat": lat,
        "lon": lon,
        "property": "phh2o",
        "depth": "0-5cm",
        "value": "mean",
    }

    response = requests.get(url, params = params, timeout = 30)
    response.raise_for_status()

    data = response.json()

    layers = data["properties"]["layers"]
    ph_layer = layers[0]
    depth_data = ph_layer["depths"][0]
    mean_value = depth_data["values"]["mean"]

    soil_ph = mean_value / 10

    return soil_ph

def fetch_and_update_soil_ph(field_id, lat, lon):
    try:
        soil_ph = fetch_soil_ph(lat, lon)
        update_field_soil_ph(field_id, soil_ph)

        return{
            "success": True,
            "data": soil_ph,
            "message": "Soil pH fetched from API(SoilGrids) and updated successfully.",

        }
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "message": f"Failed to fetch soil pH from API(SoilGrids) and update. Error: {str(e)}",
        }