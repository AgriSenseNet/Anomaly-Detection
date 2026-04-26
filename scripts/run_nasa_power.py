from app.services.nasa_power_service import fetch_and_cache_nasa_power

def main():
    field_id = "field_001"
    lat = 7.2083
    lon = 79.8358

    result = fetch_and_cache_nasa_power(field_id, lat, lon)
    
    print("Success:", result["success"])
    print("Source:", result["source"])
    print("data:", result["data"])
    print("Message:", result["message"])   

if __name__ == "__main__":
    main()