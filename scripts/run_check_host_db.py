from app.config.db import get_db_connection

conn = get_db_connection()
cursor = conn.cursor()

cursor.execute("""
    SELECT * from anmaly_events;
""")

print(cursor.fetchall())

cursor.close()
conn.close()