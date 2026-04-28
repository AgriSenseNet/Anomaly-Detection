from datetime import datetime, timedelta, timezone
import random
import sys
from pathlib import Path

# Allow script to import from app/ when run from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.config.db import get_db_connection


FIELD_ID = "field_001"
DEVICE_ID = "device_001"

# These are the confirmed Sprint 2 sensor streams:
# Soil Moisture, Soil Temperature, Ambient Temperature, Humidity,
# Atmospheric Pressure, Solar Radiation.
SENSOR_CONFIG = {
    "soil_moisture": {
        "unit": "%VWC",
        "normal_min": 25.0,
        "normal_max": 45.0,
    },
    "soil_temp": {
        "unit": "C",
        "normal_min": 24.0,
        "normal_max": 34.0,
    },
    "ambient_temp": {
        "unit": "C",
        "normal_min": 25.0,
        "normal_max": 36.0,
    },
    "humidity": {
        "unit": "%",
        "normal_min": 55.0,
        "normal_max": 90.0,
    },
    "pressure": {
        "unit": "hPa",
        "normal_min": 995.0,
        "normal_max": 1015.0,
    },
    "solar_radiation": {
        "unit": "lux",
        "normal_min": 0.0,
        "normal_max": 85000.0,
    },
}


def create_normal_value(parameter: str, timestamp: datetime) -> float:
    """
    Create realistic fake sensor values.
    This is not real farm data. It is only for Sprint 2 testing.
    """

    config = SENSOR_CONFIG[parameter]

    # Solar radiation should be low at night and high in daytime
    if parameter == "solar_radiation":
        hour = timestamp.hour

        if 6 <= hour <= 18:
            # Simple day curve: highest around noon
            noon_distance = abs(12 - hour)
            max_light = 85000 - (noon_distance * 9000)
            max_light = max(max_light, 5000)
            return round(random.uniform(3000, max_light), 2)

        return round(random.uniform(0, 800), 2)

    # Humidity usually drops when daytime temperature rises
    if parameter == "humidity":
        hour = timestamp.hour

        if 11 <= hour <= 15:
            return round(random.uniform(55, 72), 2)

        return round(random.uniform(68, 90), 2)

    # Ambient temperature slightly higher during day
    if parameter == "ambient_temp":
        hour = timestamp.hour

        if 10 <= hour <= 16:
            return round(random.uniform(30, 36), 2)

        return round(random.uniform(25, 30), 2)

    return round(random.uniform(config["normal_min"], config["normal_max"]), 2)


def insert_sensor_reading(cursor, field_id, device_id, timestamp, parameter, value, unit):
    cursor.execute(
        """
        INSERT INTO sensor_readings (
            field_id,
            device_id,
            timestamp,
            parameter,
            value,
            unit
        )
        VALUES (%s, %s, %s, %s, %s, %s);
        """,
        (field_id, device_id, timestamp, parameter, value, unit),
    )


def ensure_field_exists(cursor):
    """
    Make sure field_001 exists.
    If it already exists, this does nothing.
    """

    cursor.execute(
        """
        INSERT INTO fields (
            field_id,
            crop_type,
            planting_date,
            lat,
            lon,
            soil_ph,
            field_capacity_vwc
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        )
        ON CONFLICT (field_id) DO NOTHING;
        """,
        (
            FIELD_ID,
            "tomato",
            "2026-04-20",
            7.2083,
            79.8358,
            None,
            35.0,
        ),
    )


def insert_intentional_anomalies(cursor):
    """
    Add bad/suspicious readings so validation and anomaly detection
    have something to catch later.
    """

    now = datetime.now(timezone.utc)

    anomalies = [
        # physically impossible / invalid values
        ("soil_moisture", -10.0, "%VWC", "negative soil moisture"),
        ("humidity", 140.0, "%", "humidity above 100"),
        ("solar_radiation", 200000.0, "lux", "solar radiation spike"),
        ("pressure", 500.0, "hPa", "unrealistic pressure low"),

        # suspicious but not always physically impossible
        ("soil_temp", 60.0, "C", "soil temperature spike"),
        ("ambient_temp", 55.0, "C", "ambient temperature spike"),
    ]

    for index, (parameter, value, unit, reason) in enumerate(anomalies):
        anomaly_time = now - timedelta(minutes=index * 10)

        insert_sensor_reading(
            cursor=cursor,
            field_id=FIELD_ID,
            device_id=DEVICE_ID,
            timestamp=anomaly_time,
            parameter=parameter,
            value=value,
            unit=unit,
        )

        print(f"Inserted anomaly: {parameter}={value} ({reason})")

    # Stuck sensor example: same soil moisture value repeated many times
    stuck_start_time = now - timedelta(hours=2)

    for i in range(12):
        insert_sensor_reading(
            cursor=cursor,
            field_id=FIELD_ID,
            device_id=DEVICE_ID,
            timestamp=stuck_start_time + timedelta(minutes=i * 5),
            parameter="soil_moisture",
            value=31.11,
            unit="%VWC",
        )

    print("Inserted stuck sensor pattern: soil_moisture=31.11 repeated 12 times")


def generate_fake_sensor_data():
    connection = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        ensure_field_exists(cursor)

        # Optional: clean old generated data for repeatable testing
        cursor.execute(
            """
            DELETE FROM sensor_readings
            WHERE field_id = %s
              AND device_id = %s;
            """,
            (FIELD_ID, DEVICE_ID),
        )

        # Generate 7 days of readings, every 30 minutes
        # 7 days * 48 readings/day * 6 sensor types = 2016 normal rows
        start_time = datetime.now(timezone.utc) - timedelta(days=7)
        interval_minutes = 30
        total_steps = 7 * 24 * 2

        inserted_count = 0

        for step in range(total_steps):
            timestamp = start_time + timedelta(minutes=step * interval_minutes)

            for parameter, config in SENSOR_CONFIG.items():
                value = create_normal_value(parameter, timestamp)

                insert_sensor_reading(
                    cursor=cursor,
                    field_id=FIELD_ID,
                    device_id=DEVICE_ID,
                    timestamp=timestamp,
                    parameter=parameter,
                    value=value,
                    unit=config["unit"],
                )

                inserted_count += 1

        insert_intentional_anomalies(cursor)

        connection.commit()

        print("Fake sensor data generation completed.")
        print(f"Normal rows inserted: {inserted_count}")
        print("Intentional anomaly rows inserted: 18")
        print("Target table: sensor_readings")

    except Exception as error:
        if connection:
            connection.rollback()

        print("Failed to generate fake sensor data.")
        print(f"Error: {error}")

    finally:
        if connection:
            connection.close()


if __name__ == "__main__":
    generate_fake_sensor_data()