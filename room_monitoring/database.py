import sqlite3

DB = "database/room.db"

def connect():
    return sqlite3.connect(DB)

def add_person(person_id):

    conn = connect()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO occupancy(
        person_id,
        entry_time,
        last_seen
    )
    VALUES(
        ?,
        datetime('now'),
        datetime('now')
    )
    """, (person_id,))

    conn.commit()
    conn.close()


def update_seen(person_id):

    conn = connect()
    cur = conn.cursor()

    cur.execute("""
    UPDATE occupancy
    SET last_seen=datetime('now')
    WHERE person_id=?
    """, (person_id,))

    conn.commit()
    conn.close()


def get_total_unique():

    conn = connect()
    cur = conn.cursor()

    cur.execute("""
    SELECT COUNT(DISTINCT person_id)
    FROM occupancy
    """)

    total = cur.fetchone()[0]

    conn.close()

    return total


def get_records():

    conn = connect()
    cur = conn.cursor()

    cur.execute("""
    SELECT *
    FROM occupancy
    ORDER BY entry_time DESC
    """)

    data = cur.fetchall()

    conn.close()

    return data