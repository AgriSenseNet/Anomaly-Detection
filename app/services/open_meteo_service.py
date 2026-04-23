import requests

from app.repositories.weather_cache_repository import (
    insert_weather_cache,
    get_latest_weather_cache,   
)

def fetch_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation_probability,"
            "shortwave_radiation,"
            "surface_pressure,"
            "wind_speed_10m"
        ),
    }

    response = requests.get(url, params = params, timeout = 30)
    response.raise_for_status()

    return response.json()

def fetch_and_cache_weather(field_id, lat, lon):
    try:
        weather_data = fetch_weather(lat, lon)
        insert_weather_cache(field_id, weather_data)

        return{
            "success": True,
            "source": "api",
            "data": weather_data,
            "message": "API call successful, data cached.",
        }
    
    except Exception as e:
        cached_data = get_latest_weather_cache(field_id)

        if cached_data:
            return {
                "success": True,
                "source": "cache",
                "data": cached_data,
                "message": f"API failed, returned latest cached data. Error: {str(e)}",
            }
        
        return {
            "success": False,
            "source": None,
            "data": None,
            "message": f"API failed and no cached data available. Error: {str(e)}",
        }