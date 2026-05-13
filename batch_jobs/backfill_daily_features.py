"""
One-time backfill: populates daily_features for every date present in sensor_readings.

Run from the project root:
    python batch_jobs/backfill_daily_features.py

Uses the same DB credentials and logic as batch_features.py.
"""

import os
import sys
from datetime import date

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env.local"))

sys.path.insert(0, os.path.dirname(__file__))
from batch_features import get_connection, run_batch


def get_available_dates() -> list[date]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT DATE(timestamp)
        FROM sensor_readings
        ORDER BY 1;
    """)
    dates = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return dates


def main():
    dates = get_available_dates()
    if not dates:
        print("No dates found in sensor_readings.")
        sys.exit(1)

    print(f"Backfilling daily_features for {len(dates)} date(s): {dates[0]} → {dates[-1]}")

    for d in dates:
        print(f"\n--- {d} ---")
        run_batch(target_date=d)

    print("\nBackfill complete.")


if __name__ == "__main__":
    main()
