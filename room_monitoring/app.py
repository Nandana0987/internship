from flask import Flask
from flask import render_template
import sqlite3

from database import (
    get_total_unique,
    get_records
)

app = Flask(__name__)

@app.route("/")
def dashboard():

    total = get_total_unique()

    records = get_records()

    return render_template(
        "dashboard.html",
        total=total,
        records=records
    )

@app.route("/records")
def records():

    data = get_records()

    return render_template(
        "records.html",
        records=data
    )
@app.route("/analytics")
def analytics():

    labels = [
        "Mon",
        "Tue",
        "Wed",
        "Thu",
        "Fri"
    ]

    values = [
        12,
        18,
        15,
        25,
        20
    ]

    return render_template(
        "analytics.html",
        labels=labels,
        values=values
    )
from flask import redirect

@app.route("/clear")
def clear():

    conn = sqlite3.connect("database/room.db")
    cursor = conn.cursor()

    cursor.execute("DELETE FROM occupancy")

    conn.commit()
    conn.close()

    return redirect("/")
if __name__ == "__main__":
    app.run(debug=True)