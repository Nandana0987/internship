import sqlite3

conn = sqlite3.connect("database/room.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS occupancy(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT,
    entry_time DATETIME,
    last_seen DATETIME
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS statistics(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    total_people INTEGER,
    peak_occupancy INTEGER
)
""")

conn.commit()
conn.close()

print("Database Created")