"""Smoke test: exercises every page and API against a throwaway database.

Run from the project root:  python -m tests.smoke_test
"""
import io
import os
import sys
import tempfile

tmp = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(tmp, "test.db")

from app import app  # noqa: E402
from models import Prediction, Student, User, db  # noqa: E402

c = app.test_client()
failures = []
with app.app_context():
    teacher = User(name="Test Teacher", email="teacher@test.local", role="teacher")
    teacher.set_password("testpass123")
    db.session.add(teacher)
    db.session.commit()
login_response = c.post("/login", data={"role": "teacher", "email": "teacher@test.local", "password": "testpass123"})
if login_response.status_code != 302:
    print("FAIL teacher test login")
    sys.exit(1)


def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra and not cond else ""))
    if not cond:
        failures.append(name)


def get(path, code=200):
    r = c.get(path)
    check(f"GET {path} -> {code}", r.status_code == code, f"got {r.status_code}")
    return r


# --- empty-database states
get("/", 302)  # signed-in teachers are sent to their dashboard
for p in ["/about", "/dashboard", "/students", "/predict", "/analysis", "/data", "/evaluation"]:
    get(p)
check("dashboard empty state", b"No student records yet" in get("/dashboard").data)
get("/nope", 404)
# Role access checks: teacher pages reject anonymous/student accounts.
student_client = app.test_client()
r = student_client.get("/dashboard")
check("anonymous teacher page redirects to login", r.status_code == 302 and "/login" in r.location)

# --- add student (valid + invalid + duplicate)
good = dict(name="Asha Patil", roll_number="T001", department="Computer Science", semester="5",
            attendance="82", internal_marks="38", assignment_marks="16", previous_semester="74",
            study_hours="3.5", practical_marks="40")
r = c.post("/students/add", data=good)
check("add student redirects", r.status_code == 302)
bad = dict(good, roll_number="T002", attendance="150", study_hours="30", internal_marks="99")
r = c.post("/students/add", data=bad)
check("invalid student rejected", r.status_code == 200 and b"Attendance must be between" in r.data)
r = c.post("/students/add", data=good)
check("duplicate roll rejected", b"already exists" in r.data)

with app.app_context():
    sid = Student.query.filter_by(roll_number="T001").one().id
    check("auto prediction stored", Prediction.query.filter_by(student_id=sid).count() == 1)

# Student account registration links to the existing roll number and sees only its own portal.
r = student_client.post("/register/student", data={"name": "Asha Patil", "roll_number": "T001",
    "email": "asha@test.local", "password": "studentpass123", "confirm_password": "studentpass123"})
check("student registration succeeds", r.status_code == 302)
r = student_client.post("/login", data={"role": "student", "email": "asha@test.local", "password": "studentpass123"})
check("student login succeeds", r.status_code == 302 and "/student/dashboard" in r.location)
r = student_client.get("/student/dashboard")
check("student dashboard loads", r.status_code == 200 and b"Asha Patil" in r.data)
r = student_client.get("/dashboard")
check("student blocked from teacher dashboard", r.status_code == 302 and "/student/dashboard" in r.location)
r = student_client.post("/logout")
check("student logout succeeds", r.status_code == 302)

# --- profile / edit / pages with data
get(f"/students/{sid}")
get(f"/students/{sid}/edit")
get(f"/analysis/{sid}")
get("/students?q=asha")
get("/students?performance=High&department=Computer+Science&semester=5")
r = c.post(f"/students/{sid}/edit", data=dict(good, attendance="60"))
check("edit works", r.status_code == 302)

# --- prediction API
r = c.post("/api/predict", json={k: good[k] for k in
           ["attendance", "internal_marks", "assignment_marks", "previous_semester", "study_hours", "practical_marks"]})
j = r.get_json()
check("api predict ok", r.status_code == 200 and j["label"] in ("Low", "Medium", "High") and j["recommendations"])
r = c.post("/api/predict", json={"attendance": "abc"})
check("api predict validation", r.status_code == 400)
r = c.post("/api/predictions", json=dict(j["inputs"], student_id=sid))
check("save prediction", r.get_json()["ok"])
check("history api", len(c.get(f"/api/students/{sid}/predictions").get_json()["predictions"]) >= 3)

# --- CSV import
csv_text = ("name,roll_number,department,semester,attendance,internal_marks,assignment_marks,previous_semester,study_hours,practical_marks\n"
            "Ravi K,T010,Mechanical,3,70,30,12,60,2.5,30\n"
            "Dup,T001,Mechanical,3,70,30,12,60,2.5,30\n"
            "Bad,T011,Mechanical,3,170,30,12,60,2.5,30\n"
            "Twice,T010,Mechanical,3,70,30,12,60,2.5,30\n")
r = c.post("/data/import", data={"file": (io.BytesIO(csv_text.encode()), "x.csv")}, content_type="multipart/form-data")
check("import reports 1 imported / 3 skipped", b"1 imported, 3 skipped" in r.data)
r = c.post("/data/import", data={"file": (io.BytesIO(b"a,b\n1,2\n"), "x.csv")}, content_type="multipart/form-data")
check("import missing columns", b"Missing required column" in r.data)
for p in ["/data/template.csv", "/data/students.csv", "/data/predictions.csv"]:
    get(p)

# --- dashboard with data + api
check("dashboard api", c.get("/api/dashboard").get_json()["total"] == 2)
get("/dashboard")
get("/api/evaluation")

# --- delete
r = c.post(f"/students/{sid}/delete")
check("delete works", r.status_code == 302)
with app.app_context():
    check("predictions cascade-deleted", Prediction.query.filter_by(student_id=sid).count() == 0)

print("\nALL PASSED" if not failures else f"\n{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
