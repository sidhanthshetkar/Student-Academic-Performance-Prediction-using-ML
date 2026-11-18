"""Central configuration for the Student Performance application."""
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
MODEL_DIR = os.path.join(BASE_DIR, "ml", "artifacts")
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(INSTANCE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ---- Maximum marks (change these to match your institution) ----
MAX_INTERNAL = 50       # internal examination marks
MAX_ASSIGNMENT = 20     # assignment marks
MAX_PRACTICAL = 50      # practical / lab marks
MAX_PREVIOUS = 100      # previous semester percentage
STUDY_HOURS_TARGET = 4  # daily study hours used as a "good" reference (for scoring/advice only)

# ---- Prediction classes (order matters for charts and reports) ----
CLASSES = ["Low", "Medium", "High"]

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_semester",
    "study_hours",
    "practical_marks",
]

FEATURE_LABELS = {
    "attendance": "Attendance (%)",
    "internal_marks": "Internal Marks",
    "assignment_marks": "Assignment Marks",
    "previous_semester": "Previous Semester (%)",
    "study_hours": "Study Hours / Day",
    "practical_marks": "Practical Marks",
}

FEATURE_MAX = {
    "attendance": 100,
    "internal_marks": MAX_INTERNAL,
    "assignment_marks": MAX_ASSIGNMENT,
    "previous_semester": MAX_PREVIOUS,
    "study_hours": 24,
    "practical_marks": MAX_PRACTICAL,
}

DEPARTMENTS = [
    "Computer Science", "Information Technology", "Electronics",
    "Mechanical", "Civil", "Electrical",
]


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-development-secret-before-deployment")
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", "sqlite:///" + os.path.join(INSTANCE_DIR, "students.db")
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB CSV upload limit
