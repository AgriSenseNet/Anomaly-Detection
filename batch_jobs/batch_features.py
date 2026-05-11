# batch_features.py
# Runs daily to compute feature aggregates from raw sensor readings
# Output goes to daily_features table — used by all ML models

import psycopg2
from dotenv import load_dotenv
import os
from datetime import date, timedelta

load_dotenv()

# ─────────────────────────────────────────
# Connect to PostgreSQL
# ─────────────────────────────────────────
def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

# ─────────────────────────────────────────
# Calculate Growing Degree Days
# ─────────────────────────────────────────
def calculate_gdd(t_max, t_min, t_base=10.0):
    """
    GDD = max(0, (T_max + T_min) / 2 - T_base)
    T_base = 10°C for most vegetables
    """
    return max(0, (t_max + t_min) / 2 - t_base)

# ─────────────────────────────────────────
# Main batch job
# ─────────────────────────────────────────
def run_batch(target_date=None):
    if target_date is None:
        target_date = date.today() - timedelta(days=1)

    print(f"Running batch pipeline for date: {target_date}")

    conn = get_connection()
    cursor = conn.cursor()

    # Get all fields
    cursor.execute("SELECT field_id, planting_date FROM fields;")
    fields = cursor.fetchall()

    if not fields:
        print("No fields found in database!")
        return

    for field_id, planting_date in fields:
        print(f"Processing field_id: {field_id}")

        # Get sensor readings for this field and date
        cursor.execute("""
            SELECT parameter, value 
            FROM sensor_readings 
            WHERE field_id = %s 
            AND DATE(timestamp) = %s
        """, (field_id, target_date))

        readings = cursor.fetchall()

        if not readings:
            print(f"  No readings for field {field_id} on {target_date}")
            continue

        # Organize readings by parameter
        params = {}
        for parameter, value in readings:
            if parameter not in params:
                params[parameter] = []
            params[parameter].append(float(value))

        # Calculate aggregates
        def avg(lst): return sum(lst) / len(lst) if lst else None
        def mn(lst): return min(lst) if lst else None
        def mx(lst): return max(lst) if lst else None
        def std(lst):
            if not lst or len(lst) < 2:
                return 0
            mean = avg(lst)
            return (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5

        soil_moisture = params.get('soil_moisture', [])
        soil_temp = params.get('soil_temp', [])
        ambient_temp = params.get('ambient_temp', [])
        humidity = params.get('humidity', [])
        pressure = params.get('pressure', [])
        solar = params.get('solar_radiation', [])

        # Calculate GDD
        if ambient_temp:
            gdd_today = calculate_gdd(mx(ambient_temp), mn(ambient_temp))
        else:
            gdd_today = 0

        # Calculate accumulated GDD since planting
        if planting_date:
            cursor.execute("""
                SELECT COALESCE(SUM(
                    CASE WHEN gdd_accumulated IS NOT NULL 
                    THEN 1 ELSE 0 END
                ), 0)
                FROM daily_features 
                WHERE field_id = %s 
                AND date >= %s 
                AND date < %s
            """, (field_id, planting_date, target_date))
            
            cursor.execute("""
                SELECT COALESCE(SUM(gdd_accumulated), 0)
                FROM daily_features
                WHERE field_id = %s
                AND date >= %s
                AND date < %s
            """, (field_id, planting_date, target_date))
            
            prev_gdd = cursor.fetchone()[0] or 0
            total_gdd = float(prev_gdd) + gdd_today
        else:
            total_gdd = gdd_today

        # Insert into daily_features
        cursor.execute("""
            INSERT INTO daily_features (
                field_id, date,
                soil_moisture_mean, soil_moisture_std,
                soil_temp_mean,
                ambient_temp_mean, ambient_temp_min, ambient_temp_max,
                humidity_mean, pressure_mean,
                solar_lux_sum, gdd_accumulated,
                et0_nasa, rainfall_api
            ) VALUES (
                %s, %s,
                %s, %s,
                %s,
                %s, %s, %s,
                %s, %s,
                %s, %s,
                %s, %s
            )
            ON CONFLICT (field_id, date) DO UPDATE SET
                soil_moisture_mean = EXCLUDED.soil_moisture_mean,
                soil_moisture_std = EXCLUDED.soil_moisture_std,
                soil_temp_mean = EXCLUDED.soil_temp_mean,
                ambient_temp_mean = EXCLUDED.ambient_temp_mean,
                ambient_temp_min = EXCLUDED.ambient_temp_min,
                ambient_temp_max = EXCLUDED.ambient_temp_max,
                humidity_mean = EXCLUDED.humidity_mean,
                pressure_mean = EXCLUDED.pressure_mean,
                solar_lux_sum = EXCLUDED.solar_lux_sum,
                gdd_accumulated = EXCLUDED.gdd_accumulated;
        """, (
            field_id, target_date,
            avg(soil_moisture), std(soil_moisture),
            avg(soil_temp),
            avg(ambient_temp), mn(ambient_temp), mx(ambient_temp),
            avg(humidity), avg(pressure),
            sum(solar) if solar else None, total_gdd,
            None, None  # et0_nasa and rainfall_api filled by Yasindu's DAG
        ))

        print(f"  ✅ Features saved for field {field_id}")

    conn.commit()
    cursor.close()
    conn.close()
    print("Batch pipeline completed successfully!")

if __name__ == "__main__":
    run_batch()