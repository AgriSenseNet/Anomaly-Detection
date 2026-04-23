from app.services.open_meteo_service import fetch_and_cache_weather

def main():
    field_id = "field_002"
    lat = 7.2083
    lon = 79.8358

    result = fetch_and_cache_weather(field_id, lat, lon)

    print("success:", result["success"])
    print("source:", result["source"])
    print("data:", result["data"])
    print("message:", result["message"])   

if __name__ == "__main__":
    main()