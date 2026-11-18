"""SQLAlchemy database models."""
import json
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

from flask_sqlalchemy import SQLAlchemy

import config

db = SQLAlchemy()


class Student(db.Model):
    __tablename__ = "students"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    roll_number = db.Column(db.String(40), unique=True, nullable=False, index=True)
    department = db.Column(db.String(80), nullable=False)
    semester = db.Column(db.Integer, nullable=False)
    attendance = db.Column(db.Float, nullable=False)
    internal_marks = db.Column(db.Float, nullable=False)
    assignment_marks = db.Column(db.Float, nullable=False)
    previous_semester = db.Column(db.Float, nullable=False)
    study_hours = db.Column(db.Float, nullable=False)
    practical_marks = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

    predictions = db.relationship(
        "Prediction", backref="student", cascade="all, delete-orphan",
        order_by="Prediction.created_at.desc()", lazy="select",
    )

    # ---- helpers ----
    def features(self):
        return {f: getattr(self, f) for f in config.FEATURES}

    @property
    def marks_percentage(self):
        """Average of the recorded marks, as a percentage.

        This is a *descriptive* statistic used only for charts and analysis.
        It is never used as a model input or training label.
        """
        parts = [
            self.internal_marks / config.MAX_INTERNAL * 100,
            self.assignment_marks / config.MAX_ASSIGNMENT * 100,
            self.practical_marks / config.MAX_PRACTICAL * 100,
            self.previous_semester / config.MAX_PREVIOUS * 100,
        ]
        return round(sum(parts) / len(parts), 2)

    @property
    def latest_prediction(self):
        return self.predictions[0] if self.predictions else None

    @property
    def predicted_label(self):
        p = self.latest_prediction
        return p.predicted_label if p else None

    def to_dict(self):
        d = {
            "id": self.id, "name": self.name, "roll_number": self.roll_number,
            "department": self.department, "semester": self.semester,
        }
        d.update(self.features())
        d["predicted_performance"] = self.predicted_label
        return d


class User(db.Model):
    """Login account for a teacher or a student linked to an existing record."""
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(160), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # teacher | student
    student_id = db.Column(db.Integer, db.ForeignKey("students.id"), unique=True, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)

    student = db.relationship("Student", backref=db.backref("account", uselist=False))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("students.id"), nullable=False, index=True)
    predicted_label = db.Column(db.String(20), nullable=False)
    probabilities_json = db.Column(db.Text, nullable=True)
    model_name = db.Column(db.String(80), nullable=False)
    source = db.Column(db.String(20), default="manual")  # manual | auto | import
    # Snapshot of the inputs used, so history stays meaningful if the student is edited later
    attendance = db.Column(db.Float)
    internal_marks = db.Column(db.Float)
    assignment_marks = db.Column(db.Float)
    previous_semester = db.Column(db.Float)
    study_hours = db.Column(db.Float)
    practical_marks = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.now)

    @property
    def probabilities(self):
        try:
            return json.loads(self.probabilities_json) if self.probabilities_json else None
        except ValueError:
            return None

    def to_dict(self):
        d = {
            "id": self.id,
            "student_id": self.student_id,
            "roll_number": self.student.roll_number if self.student else None,
            "student_name": self.student.name if self.student else None,
            "predicted_label": self.predicted_label,
            "model_name": self.model_name,
            "source": self.source,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M"),
            "probabilities": self.probabilities,
        }
        for f in config.FEATURES:
            d[f] = getattr(self, f)
        return d
