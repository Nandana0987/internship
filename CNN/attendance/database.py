import sqlite3

# Connect to database
conn = sqlite3.connect("attendance.db")

cursor = conn.cursor()

# Create attendance table
cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id TEXT NOT NULL,
    employee_name TEXT NOT NULL,
    date TEXT NOT NULL,
    time TEXT NOT NULL,
    confidence REAL NOT NULL
)
""")

conn.commit()
conn.close()

print("Database Created Successfully!")