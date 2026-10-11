import sqlite3

def init_db():
    conn = sqlite3.connect("bank.db")
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT,
        password TEXT,
        balance REAL DEFAULT 50000
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        type TEXT,
        amount REAL
    )
    """)

    conn.commit()
    conn.close()