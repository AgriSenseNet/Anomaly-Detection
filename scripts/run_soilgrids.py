from app.services.soilgrids_service import fetch_and_update_soil_ph

def main():
    field_id = "field_004"
    lat = 41.8781  
    lon = -93.0977

    result = fetch_and_update_soil_ph(field_id, lat, lon)

    print("success:", result["success"])
    print("Soil_pH:", result["data"])
    print("message:", result["message"])    

if __name__ == "__main__":
    main()
