import pandas as pd

from app.config.db import get_db_connection
from app.services.data_validation_service import validate_sensor_data

def main():
    conn = get_db_connection()

    query = """
        select field_id, device_id, timestamp, parameter, value
        from sensor_readings
        order by timestamp desc
        """
    
    df = pd.read_sql(query, conn)

    conn.close()

    is_valid, errors = validate_sensor_data(df)

    print ("Sensor Data Validation ")
    print("------------------------")
    print(f"Rows checked : {len(df)}")

    if is_valid:
        print("Validation Result: PASSED")

    else:
        print("Validation Result: FAILED")
        for error in errors[:20]:
            print(f"- {error}")
        
        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more errors")

if __name__ == "__main__":
    main()