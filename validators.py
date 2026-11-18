"""Server-side validation shared by forms, JSON APIs and CSV import."""
import math

import config

NUMERIC_RULES = {
    # field: (label, min, max)
    "attendance": ("Attendance", 0, 100),
    "internal_marks": ("Internal marks", 0, config.MAX_INTERNAL),
    "assignment_marks": ("Assignment marks", 0, config.MAX_ASSIGNMENT),
    "previous_semester": ("Previous semester %", 0, config.MAX_PREVIOUS),
    "study_hours": ("Study hours", 0, 24),
    "practical_marks": ("Practical marks", 0, config.MAX_PRACTICAL),
}


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def validate_features(data):
    """Validate only the six model features. Returns (clean_dict, errors_dict)."""
    clean, errors = {}, {}
    for field, (label, lo, hi) in NUMERIC_RULES.items():
        raw = data.get(field)
        val = _to_float(raw)
        if val is None:
            errors[field] = f"{label} is required and must be a number."
        elif val < lo or val > hi:
            errors[field] = f"{label} must be between {lo} and {hi}."
        else:
            clean[field] = round(val, 2)
    return clean, errors


def validate_student(data):
    """Validate a full student record. Does NOT check roll-number uniqueness (needs the DB)."""
    clean, errors = validate_features(data)

    name = (data.get("name") or "").strip()
    if not name:
        errors["name"] = "Name is required."
    elif len(name) > 120:
        errors["name"] = "Name must be 120 characters or fewer."
    else:
        clean["name"] = name

    roll = (str(data.get("roll_number") or "")).strip()
    if not roll:
        errors["roll_number"] = "Roll number is required."
    elif len(roll) > 40:
        errors["roll_number"] = "Roll number must be 40 characters or fewer."
    else:
        clean["roll_number"] = roll

    dept = (data.get("department") or "").strip()
    if not dept:
        errors["department"] = "Department is required."
    elif len(dept) > 80:
        errors["department"] = "Department must be 80 characters or fewer."
    else:
        clean["department"] = dept

    sem = _to_float(data.get("semester"))
    if sem is None or sem != int(sem) or not (1 <= int(sem) <= 12):
        errors["semester"] = "Semester must be a whole number between 1 and 12."
    else:
        clean["semester"] = int(sem)

    return clean, errors
