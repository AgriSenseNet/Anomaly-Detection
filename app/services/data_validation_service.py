import panda as pd

SENSOR_VALID_RANGES = {
    "soil_moisture": {
        "min": 0,
        "max": 100,
        "unit": "%VWC",
    },
    "soil_temp": {
        "min": 0,
        "max": 60,
        "unit": "C",
    },
    "ambient_temp": {
        "min": 0,
        "max": 60,
        "unit": "C",
    },
    "humidity": {
        "min": 0,
        "max": 100,
        "unit": "%",
    },
    "pressure": {
        "min": 850,
        "max": 1100,
        "unit": "hPa",
    },
    "solar_radiation": {
        "min": 0,
        "max": 120000,
        "unit": "lux",
    },
}

REQUIRED_COLUMNS = [
    "id",
    "field_id",
    "device_id",
    "timestamp",
    "parameter",
    "value",
    "unit",
]


