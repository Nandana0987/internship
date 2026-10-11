import sqlite3

conn = sqlite3.connect("database/room.db")
cursor = conn.cursor()

cursor.execute("DELETE FROM occupancy")

conn.commit()
conn.close()

print("All records deleted successfully!")