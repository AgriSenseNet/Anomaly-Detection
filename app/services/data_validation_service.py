import pandas as pd

VALID_RANGES ={
    "soil_moisture": (0, 100),
    "soil_temp": (0, 60),
    "ambient_temp": (0, 60),
    "humidity": (0, 100),
    "pressure": (300, 1100),
    "solar_radiation": (0, 120000),
}

def validate_sensor_data(df: pd.DataFrame):
    errors = []

    #1 check for empty table
    if df.empty:
        errors.append("sensor_reading table is empty")
        return False, errors
    
    #2 check for null values

    required_columns = ["field_id", "device_id", "timestamp", "parameter", "value"]

    for column in required_columns:
        if df[column].isnull().any():
            errors.append(f"column {column} has null values")
        
    #3 check parameters are valid
    for index, row in df.iterrows():
        parameter = row["parameter"]
        value = row["value"]

        if parameter not in VALID_RANGES:
            errors.append(f"invalid parameter {parameter} at index {index}")
            continue
        
        min_value, max_value = VALID_RANGES[parameter]

        if value < min_value or value > max_value:
            errors.append(f"value {value} for parameter {parameter} at index {index} is out of valid range ({min_value}, {max_value})")

    
    #4 if errors list s empty, validation passed
    if len(errors) == 0:
        return True, errors
    
    return False, errors
