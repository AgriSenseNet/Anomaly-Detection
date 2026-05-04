from datetime import datetime, timedelta, timezone
import csv
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.config.db import get_db_connection


FIELD_ID = "field_001"
DEVICE_ID = "device_001"

LABEL_FILE = (
    PROJECT_ROOT
    / "docs"
    / "evidence"
    / "sprint2"
    / "generated_anomaly_labels.csv"
)

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
    hour = timestamp.hour

    if parameter == "solar_radiation":
        if 6 <= hour <= 18:
            noon_distance = abs(12 - hour)
            max_light = 85000 - (noon_distance * 9000)
            max_light = max(max_light, 5000)
            return round(random.uniform(3000, max_light), 2)

        return round(random.uniform(0, 800), 2)

    if parameter == "humidity":
        if 11 <= hour <= 15:
            return round(random.uniform(55, 72), 2)

        return round(random.uniform(68, 90), 2)

    if parameter == "ambient_temp":
        if 10 <= hour <= 16:
            return round(random.uniform(30, 36), 2)

        return round(random.uniform(25, 30), 2)

    config = SENSOR_CONFIG[parameter]
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
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id;
        """,
        (field_id, device_id, timestamp, parameter, value, unit),
    )

    return cursor.fetchone()[0]


def ensure_field_exists(cursor):
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
        VALUES (%s, %s, %s, %s, %s, %s, %s)
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


def add_label(labels, row_id, timestamp, parameter, value, anomaly_type):
    labels.append(
        {
            "sensor_reading_id": row_id,
            "device_id": DEVICE_ID,
            "timestamp": timestamp.isoformat(),
            "sensor_type": parameter,
            "value": value,
            "true_label": 1,
            "label_reason": anomaly_type,
        }
    )


def insert_single_anomaly(cursor, labels, timestamp, parameter, value, anomaly_type):
    unit = SENSOR_CONFIG[parameter]["unit"]

    row_id = insert_sensor_reading(
        cursor=cursor,
        field_id=FIELD_ID,
        device_id=DEVICE_ID,
        timestamp=timestamp,
        parameter=parameter,
        value=value,
        unit=unit,
    )

    add_label(labels, row_id, timestamp, parameter, value, anomaly_type)


def insert_stuck_sensor_block(
    cursor,
    labels,
    start_time,
    parameter,
    value,
    count,
    interval_minutes,
):
    unit = SENSOR_CONFIG[parameter]["unit"]

    for i in range(count):
        timestamp = start_time + timedelta(minutes=i * interval_minutes)

        row_id = insert_sensor_reading(
            cursor=cursor,
            field_id=FIELD_ID,
            device_id=DEVICE_ID,
            timestamp=timestamp,
            parameter=parameter,
            value=value,
            unit=unit,
        )

        add_label(
            labels=labels,
            row_id=row_id,
            timestamp=timestamp,
            parameter=parameter,
            value=value,
            anomaly_type="stuck_sensor_pattern",
        )


def insert_drift_block(
    cursor,
    labels,
    start_time,
    parameter,
    start_value,
    step_value,
    count,
    interval_minutes,
):
    unit = SENSOR_CONFIG[parameter]["unit"]

    for i in range(count):
        timestamp = start_time + timedelta(minutes=i * interval_minutes)
        value = round(start_value + (i * step_value), 2)

        row_id = insert_sensor_reading(
            cursor=cursor,
            field_id=FIELD_ID,
            device_id=DEVICE_ID,
            timestamp=timestamp,
            parameter=parameter,
            value=value,
            unit=unit,
        )

        add_label(
            labels=labels,
            row_id=row_id,
            timestamp=timestamp,
            parameter=parameter,
            value=value,
            anomaly_type="sensor_drift_pattern",
        )


def insert_noise_block(
    cursor,
    labels,
    start_time,
    parameter,
    low_value,
    high_value,
    count,
    interval_minutes,
):
    unit = SENSOR_CONFIG[parameter]["unit"]

    for i in range(count):
        timestamp = start_time + timedelta(minutes=i * interval_minutes)

        if i % 2 == 0:
            value = low_value
        else:
            value = high_value

        row_id = insert_sensor_reading(
            cursor=cursor,
            field_id=FIELD_ID,
            device_id=DEVICE_ID,
            timestamp=timestamp,
            parameter=parameter,
            value=value,
            unit=unit,
        )

        add_label(
            labels=labels,
            row_id=row_id,
            timestamp=timestamp,
            parameter=parameter,
            value=value,
            anomaly_type="unstable_sensor_noise",
        )


def insert_anomaly_mix(cursor, mixed_start_time):
    labels = []

    # ---------------------------------------------------------
    # Physically impossible anomalies
    # These should be easy for validation/rules and model to see.
    # ---------------------------------------------------------
    physical_anomalies = [
        ("soil_moisture", -10.0, "negative_soil_moisture"),
        ("soil_moisture", 120.0, "soil_moisture_above_100"),
        ("humidity", 140.0, "humidity_above_100"),
        ("humidity", -5.0, "negative_humidity"),
        ("pressure", 500.0, "unrealistic_pressure_low"),
        ("pressure", 1300.0, "unrealistic_pressure_high"),
        ("solar_radiation", 200000.0, "solar_radiation_spike"),
        ("soil_temp", 70.0, "soil_temperature_spike"),
        ("ambient_temp", 65.0, "ambient_temperature_spike"),
    ]

    for index, (parameter, value, anomaly_type) in enumerate(physical_anomalies):
        timestamp = mixed_start_time + timedelta(hours=2, minutes=index * 7)

        insert_single_anomaly(
            cursor=cursor,
            labels=labels,
            timestamp=timestamp,
            parameter=parameter,
            value=value,
            anomaly_type=anomaly_type,
        )

    # ---------------------------------------------------------
    # In-range spikes
    # These are still physically possible, but unusual compared
    # to nearby normal behavior.
    # ---------------------------------------------------------
    in_range_spikes = [
        ("soil_moisture", 95.0, "in_range_soil_moisture_spike"),
        ("soil_temp", 53.0, "in_range_soil_temperature_spike"),
        ("ambient_temp", 49.0, "in_range_ambient_temperature_spike"),
        ("humidity", 99.0, "in_range_humidity_spike"),
        ("pressure", 1090.0, "in_range_pressure_spike"),
        ("solar_radiation", 118000.0, "in_range_solar_spike"),
    ]

    for index, (parameter, value, anomaly_type) in enumerate(in_range_spikes):
        timestamp = mixed_start_time + timedelta(days=1, hours=3, minutes=index * 9)

        insert_single_anomaly(
            cursor=cursor,
            labels=labels,
            timestamp=timestamp,
            parameter=parameter,
            value=value,
            anomaly_type=anomaly_type,
        )

    # ---------------------------------------------------------
    # Stuck sensor blocks
    # Same value repeats many times.
    # ---------------------------------------------------------
    insert_stuck_sensor_block(
        cursor=cursor,
        labels=labels,
        start_time=mixed_start_time + timedelta(days=2, hours=4),
        parameter="soil_moisture",
        value=31.11,
        count=24,
        interval_minutes=5,
    )

    insert_stuck_sensor_block(
        cursor=cursor,
        labels=labels,
        start_time=mixed_start_time + timedelta(days=3, hours=5),
        parameter="humidity",
        value=77.77,
        count=18,
        interval_minutes=5,
    )

    insert_stuck_sensor_block(
        cursor=cursor,
        labels=labels,
        start_time=mixed_start_time + timedelta(days=4, hours=6),
        parameter="pressure",
        value=1008.88,
        count=18,
        interval_minutes=5,
    )

    # ---------------------------------------------------------
    # Drift anomalies
    # Sensor slowly moves away from normal.
    # ---------------------------------------------------------
    insert_drift_block(
        cursor=cursor,
        labels=labels,
        start_time=mixed_start_time + timedelta(days=5, hours=8),
        parameter="soil_moisture",
        start_value=45.0,
        step_value=2.2,
        count=16,
        interval_minutes=10,
    )

    insert_drift_block(
        cursor=cursor,
        labels=labels,
        start_time=mixed_start_time + timedelta(days=5, hours=12),
        parameter="ambient_temp",
        start_value=36.0,
        step_value=1.4,
        count=12,
        interval_minutes=10,
    )

    # ---------------------------------------------------------
    # Noisy unstable sensor anomalies
    # Jumps up/down quickly.
    # ---------------------------------------------------------
    insert_noise_block(
        cursor=cursor,
        labels=labels,
        start_time=mixed_start_time + timedelta(days=6, hours=2),
        parameter="soil_temp",
        low_value=18.0,
        high_value=55.0,
        count=14,
        interval_minutes=5,
    )

    insert_noise_block(
        cursor=cursor,
        labels=labels,
        start_time=mixed_start_time + timedelta(days=6, hours=5),
        parameter="solar_radiation",
        low_value=100.0,
        high_value=115000.0,
        count=14,
        interval_minutes=5,
    )

    return labels


def write_labels_file(labels):
    LABEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(LABEL_FILE, mode="w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "sensor_reading_id",
            "device_id",
            "timestamp",
            "sensor_type",
            "value",
            "true_label",
            "label_reason",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labels)


def generate_fake_sensor_data():
    connection = None

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        ensure_field_exists(cursor)

        cursor.execute(
            """
            DELETE FROM anomaly_events
            WHERE device_id = %s;
            """,
            (DEVICE_ID,),
        )

        cursor.execute(
            """
            DELETE FROM sensor_readings
            WHERE field_id = %s
              AND device_id = %s;
            """,
            (FIELD_ID, DEVICE_ID),
        )

        # 28 days total:
        # first 21 days = clean baseline for training
        # last 7 days = mixed normal + anomaly data for scoring/testing
        now = datetime.now(timezone.utc)
        start_time = now - timedelta(days=28)
        mixed_start_time = now - timedelta(days=7)

        interval_minutes = 15
        total_steps = 28 * 24 * 4

        normal_count = 0

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

                normal_count += 1

        labels = insert_anomaly_mix(cursor, mixed_start_time)

        connection.commit()

        write_labels_file(labels)

        print("Fake sensor data generation completed.")
        print("--------------------------------------")
        print(f"Normal rows inserted: {normal_count}")
        print(f"Anomaly rows inserted: {len(labels)}")
        print("Training baseline: first 21 days")
        print("Mixed test period: last 7 days")
        print(f"Labels saved to: {LABEL_FILE}")
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