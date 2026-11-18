"""Load clearly-labelled DEMO students into the database.

Usage:  python seed_demo.py [count]        (default 60)

These records are randomly generated and DO NOT describe real students.
Roll numbers start with "DEMO" so they are easy to recognise and delete.
It also writes data/sample_students.csv (the same demo rows) for trying the CSV import.
"""
import sys

import numpy as np
import pandas as pd

import config
from app import app, record_prediction_for
from ml.generate_dataset import generate
from models import Student, db

FIRST = ["Aarav", "Diya", "Rohan", "Sneha", "Kabir", "Meera", "Arjun", "Isha", "Vihaan", "Anaya",
         "Aditya", "Priya", "Yash", "Neha", "Omkar", "Pooja", "Sahil", "Riya", "Tejas", "Kavya"]
LAST = ["Patil", "Deshmukh", "Kulkarni", "Jadhav", "Pawar", "Shinde", "More", "Joshi", "Gaikwad", "Wagh"]


def main(count=60):
    rng = np.random.default_rng(7)
    feats = generate(count, seed=7).drop(columns="performance")
    rows = []
    for i, f in feats.iterrows():
        rows.append({
            "name": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
            "roll_number": f"DEMO{i + 1:04d}",
            "department": str(rng.choice(config.DEPARTMENTS[:4])),
            "semester": int(rng.integers(1, 9)),
            **f.to_dict(),
        })
    df = pd.DataFrame(rows)
    df.to_csv(f"{config.DATA_DIR}/sample_students.csv", index=False)

    with app.app_context():
        existing = {r[0] for r in db.session.query(Student.roll_number).all()}
        added = 0
        for r in rows:
            if r["roll_number"] in existing:
                continue
            s = Student(**r)
            db.session.add(s)
            db.session.flush()
            record_prediction_for(s, "import")
            added += 1
        db.session.commit()
    print(f"Added {added} DEMO students (randomly generated, not real). CSV: data/sample_students.csv")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
