# batch_features.py
# Runs daily to compute feature aggregates from raw sensor readings
# Output goes to daily_features table — used by all ML models

import psycopg2
from dotenv import load_dotenv
import os
from datetime import date, timedelta

load_dotenv()


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )


def calculate_gdd(t_max, t_min, t_base=10.0):
    """GDD = max(0, (T_max + T_min) / 2 - T_base). T_base=10°C for most vegetables."""
    return max(0, (t_max + t_min) / 2 - t_base)


def _get_nasa_api_values(cursor, field_id):
    """
    Extract et0_nasa (mm/day) and rainfall_api (mm/day) from the most recent
    NASA POWER cache entry for this field.

    ET0 is estimated via the Hargreaves-Samani simplified formula (no Tmax/Tmin):
      ET0 ≈ 0.0135 * (Tmean + 17.8) * Rs
    where Rs is ALLSKY_SFC_SW_DWN in MJ/m²/day.

    NASA POWER data has a ~7-day lag, so these are climate-proxy values,
    not same-day measurements.
    """
    cursor.execute("""
        SELECT data FROM external_api_cache
        WHERE source = 'nasa_power' AND field_id = %s
        ORDER BY timestamp DESC LIMIT 1
    """, (field_id,))
    row = cursor.fetchone()
    if not row:
        return None, None

    params = row[0].get("properties", {}).get("parameter", {})
    t2m_data  = params.get("T2M", {})
    rs_data   = params.get("ALLSKY_SFC_SW_DWN", {})
    rain_data = params.get("PRECTOTCORR", {})

    if not t2m_data:
        return None, None

    latest = max(t2m_data.keys())
    t2m  = t2m_data.get(latest, -999.0)
    rs   = rs_data.get(latest, -999.0)
    rain = rain_data.get(latest, -999.0)

    et0      = round(0.0135 * (t2m + 17.8) * rs, 3) if t2m != -999.0 and rs != -999.0 else None
    rainfall = round(rain, 3) if rain != -999.0 else None

    return et0, rainfall


def run_batch(target_date=None):
    if target_date is None:
        target_date = date.today() - timedelta(days=1)

    print(f"Running batch pipeline for date: {target_date}")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT field_id, planting_date FROM fields;")
    fields = cursor.fetchall()

    if not fields:
        print("No fields found in database!")
        return

    for field_id, planting_date in fields:
        print(f"Processing field_id: {field_id}")

        cursor.execute("""
            SELECT parameter, value
            FROM sensor_readings
            WHERE field_id = %s AND DATE(timestamp) = %s
        """, (field_id, target_date))
        readings = cursor.fetchall()

        if not readings:
            print(f"  No readings for field {field_id} on {target_date}")
            continue

        params = {}
        for parameter, value in readings:
            params.setdefault(parameter, []).append(float(value))

        def avg(lst): return sum(lst) / len(lst) if lst else None
        def mn(lst):  return min(lst) if lst else None
        def mx(lst):  return max(lst) if lst else None
        def std(lst):
            if not lst or len(lst) < 2:
                return 0
            mean = avg(lst)
            return (sum((x - mean) ** 2 for x in lst) / len(lst)) ** 0.5

        soil_moisture = params.get('soil_moisture', [])
        soil_temp     = params.get('soil_temp', [])
        ambient_temp  = params.get('ambient_temp', [])
        humidity      = params.get('humidity', [])
        pressure      = params.get('pressure', [])
        lux           = params.get('lux', [])  # sensor_readings stores raw lux, not solar_radiation

        gdd_today = calculate_gdd(mx(ambient_temp), mn(ambient_temp)) if ambient_temp else 0

        if planting_date:
            cursor.execute("""
                SELECT COALESCE(SUM(gdd_accumulated), 0)
                FROM daily_features
                WHERE field_id = %s AND date >= %s AND date < %s
            """, (field_id, planting_date, target_date))
            prev_gdd  = float(cursor.fetchone()[0] or 0)
            total_gdd = prev_gdd + gdd_today
        else:
            total_gdd = gdd_today

        et0_nasa, rainfall_api = _get_nasa_api_values(cursor, field_id)

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
                soil_moisture_std  = EXCLUDED.soil_moisture_std,
                soil_temp_mean     = EXCLUDED.soil_temp_mean,
                ambient_temp_mean  = EXCLUDED.ambient_temp_mean,
                ambient_temp_min   = EXCLUDED.ambient_temp_min,
                ambient_temp_max   = EXCLUDED.ambient_temp_max,
                humidity_mean      = EXCLUDED.humidity_mean,
                pressure_mean      = EXCLUDED.pressure_mean,
                solar_lux_sum      = EXCLUDED.solar_lux_sum,
                gdd_accumulated    = EXCLUDED.gdd_accumulated,
                et0_nasa           = EXCLUDED.et0_nasa,
                rainfall_api       = EXCLUDED.rainfall_api;
        """, (
            field_id, target_date,
            avg(soil_moisture), std(soil_moisture),
            avg(soil_temp),
            avg(ambient_temp), mn(ambient_temp), mx(ambient_temp),
            avg(humidity), avg(pressure),
            sum(lux) if lux else None, total_gdd,
            et0_nasa, rainfall_api,
        ))

        print(f"  Inserted features for {field_id} — et0={et0_nasa}, rain={rainfall_api}")

    conn.commit()
    cursor.close()
    conn.close()
    print("Batch pipeline completed successfully!")


if __name__ == "__main__":
    run_batch()
