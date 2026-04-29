import sys

import pandas as pd

from app.config.db import get_db_connection
from app.services.data_validation_service import validate_sensor_data

def load_sensor_data(mode):

    connection = get_db_connection()

    if mode == "clean":
        query = """
            SELECT
                field_id,
                device_id,
                timestamp,
                parameter,
                value
            FROM sensor_readings
            WHERE
                (
                    parameter = 'soil_moisture'
                    AND value BETWEEN 0 AND 100
                )
                OR
                (
                    parameter = 'soil_temp'
                    AND value BETWEEN 0 AND 60
                )
                OR
                (
                    parameter = 'ambient_temp'
                    AND value BETWEEN 0 AND 60
                )
                OR
                (
                    parameter = 'humidity'
                    AND value BETWEEN 0 AND 100
                )
                OR
                (
                    parameter = 'pressure'
                    AND value BETWEEN 850 AND 1100
                )
                OR
                (
                    parameter = 'solar_radiation'
                    AND value BETWEEN 0 AND 120000
                )
            ORDER BY timestamp DESC;
        """

    else:
        query = """
            SELECT
                field_id,
                device_id,
                timestamp,
                parameter,
                value
            FROM sensor_readings
            ORDER BY timestamp DESC;
        """    
    df = pd.read_sql(query, connection)
    connection.close()

    return df    


def main():

    mode = "all"

    if len(sys.argv) > 1:
        mode = sys.argv[1]

    if mode not in ["all", "clean"]:
        print("Invalid mode.")
        print("Use one of these:")
        print("python run_validate_sensor_data.py all")
        print("python run_validate_sensor_data.py clean")
        raise SystemExit(1)
    
    df = load_sensor_data(mode)

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