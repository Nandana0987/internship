import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect("logs.db")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS login_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            login_time TEXT
        )
    """)

    conn.commit()
    conn.close()


def add_login(username):
    conn = sqlite3.connect("logs.db")
    c = conn.cursor()

    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    c.execute("INSERT INTO login_history (username, login_time) VALUES (?, ?)",
              (username, time_now))

    conn.commit()
    conn.close()


def get_logs():
    conn = sqlite3.connect("logs.db")
    c = conn.cursor()

    c.execute("SELECT * FROM login_history ORDER BY id DESC")
    data = c.fetchall()

    conn.close()
    return data