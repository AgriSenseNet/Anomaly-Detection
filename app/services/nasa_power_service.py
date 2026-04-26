import requests

from app.repositories.nasa_power_cache_repository import (
    insert_nasa_power_cache,
    get_latest_nasa_power_cache,
)

def fetch_nasa_power(lat, lon):
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=2)

    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    url ="https://power.larc.nasa.gov/api/temporal/daily/point"

    params = {
        "parameters": "ET0,ALLSKY_SFC_SW_DWN,WS2M,PRECTOTCORR",
        "community": "AG",
        "start": start_str,
        "end": end_str,
        "latitude:": lat,
        "longitude": lon,
        "format": "JSON",
    }

    response = requests.request.get(url, params = params, timeout = 30)
    response.raise_for_status()

    return response.json()

def fetch_and_cache_nasa_power(field_id, lat, lon):
    try:
        nasa_data = fetch_nasa_power(lat, lon)
        insert_nasa_power_cache(field_id, nasa_data)

        return {
            "success": True,
            "source": "api",
            "data": nasa_data,
            "message": "NASA POWER data fetched from API and cached successfully.",

        }
    except Exception as e:
        cached_data = get_latest_nasa_power_cache(field_id)

        if cached_data:
            return {
                "success": True,
                "source": "cache",
                "data": cached_data,
                "message": f"API call failed, returned latest cached data. Error: {str(e)}",
            }
        return {
            "success": False,
            "source": None,
            "data": None,
            "message": "Failed to fetch NASA POWER data from both API and cache."
        }
