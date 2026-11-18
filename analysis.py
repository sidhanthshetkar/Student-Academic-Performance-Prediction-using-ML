"""Aggregations and statistics for the dashboard and analysis pages.

Everything here is computed from the real database records. When there is too
little data for a meaningful result, functions return None / a message instead
of inventing numbers.
"""
import numpy as np

import config
from models import Student

MIN_FOR_STATS = 10  # minimum records before a correlation is reported


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(float(np.mean(vals)), 2) if vals else None


def class_averages(students):
    """Average of each of the six factors across the given students."""
    if not students:
        return None
    return {f: _mean([getattr(s, f) for s in students]) for f in config.FEATURES}


def correlation(xs, ys):
    """Pearson r with a plain-language description, or a message if data is insufficient."""
    if len(xs) < MIN_FOR_STATS:
        return {"r": None, "message": f"At least {MIN_FOR_STATS} student records are needed to describe "
                                      f"a relationship (currently {len(xs)})."}
    if np.std(xs) == 0 or np.std(ys) == 0:
        return {"r": None, "message": "Values do not vary enough to measure a relationship."}
    r = float(np.corrcoef(xs, ys)[0, 1])
    a = abs(r)
    strength = "very weak" if a < 0.2 else "weak" if a < 0.4 else "moderate" if a < 0.6 else \
        "strong" if a < 0.8 else "very strong"
    direction = "positive" if r > 0 else "negative"
    return {"r": round(r, 3), "message": f"{strength.capitalize()} {direction} correlation (r = {r:.2f}) "
                                         f"across {len(xs)} students. Correlation does not prove cause."}


def dashboard_data():
    students = Student.query.all()
    n = len(students)
    if n == 0:
        return {"empty": True, "total": 0}

    counts = {c: 0 for c in config.CLASSES}
    unpredicted = 0
    for s in students:
        lab = s.predicted_label
        if lab in counts:
            counts[lab] += 1
        else:
            unpredicted += 1

    # Per predicted class averages (bar chart)
    by_class = {}
    for c in config.CLASSES:
        grp = [s for s in students if s.predicted_label == c]
        by_class[c] = {
            "count": len(grp),
            "attendance": _mean([s.attendance for s in grp]),
            "marks_percentage": _mean([s.marks_percentage for s in grp]),
            "study_hours": _mean([s.study_hours for s in grp]),
        }

    # Scatter points
    scatter_att, scatter_study = [], []
    for s in students:
        lab = s.predicted_label or "Unpredicted"
        scatter_att.append({"x": s.attendance, "y": s.marks_percentage, "label": lab, "name": s.name})
        scatter_study.append({"x": s.study_hours, "y": s.marks_percentage, "label": lab, "name": s.name})

    # Semester trend
    sems = sorted({s.semester for s in students})
    trend = []
    for sem in sems:
        grp = [s for s in students if s.semester == sem]
        trend.append({
            "semester": sem, "students": len(grp),
            "marks_percentage": _mean([s.marks_percentage for s in grp]),
            "attendance": _mean([s.attendance for s in grp]),
            "previous_semester": _mean([s.previous_semester for s in grp]),
        })

    return {
        "empty": False,
        "total": n,
        "counts": counts,
        "unpredicted": unpredicted,
        "avg_attendance": _mean([s.attendance for s in students]),
        "avg_internal": _mean([s.internal_marks for s in students]),
        "avg_internal_pct": _mean([s.internal_marks / config.MAX_INTERNAL * 100 for s in students]),
        "avg_study": _mean([s.study_hours for s in students]),
        "by_class": by_class,
        "scatter_attendance": scatter_att,
        "scatter_study": scatter_study,
        "semester_trend": trend,
        "corr_attendance": correlation([s.attendance for s in students],
                                       [s.marks_percentage for s in students]),
        "corr_study": correlation([s.study_hours for s in students],
                                  [s.marks_percentage for s in students]),
    }


def student_analysis(student, all_students):
    """Per-student analysis payload."""
    from recommender import factor_breakdown, get_recommender

    feats = student.features()
    breakdown = factor_breakdown(feats)
    ranked = sorted(breakdown, key=lambda r: r["percent"], reverse=True)

    peers = [s for s in all_students if s.id != student.id]
    averages = class_averages(all_students) if len(all_students) >= 2 else None
    comparison = None
    if averages:
        comparison = []
        for f in config.FEATURES:
            comparison.append({
                "factor": f, "label": config.FEATURE_LABELS[f],
                "student": feats[f], "class_avg": averages[f],
                "diff": round(feats[f] - averages[f], 2),
                "student_pct": round(feats[f] / config.FEATURE_MAX[f] * 100, 1),
                "avg_pct": round(averages[f] / config.FEATURE_MAX[f] * 100, 1),
            })

    corr_att = correlation([s.attendance for s in all_students], [s.marks_percentage for s in all_students])
    corr_study = correlation([s.study_hours for s in all_students], [s.marks_percentage for s in all_students])

    label = student.predicted_label
    recs = get_recommender().recommend(feats, label or "Medium")

    return {
        "student": student.to_dict(),
        "marks_percentage": student.marks_percentage,
        "predicted_label": label,
        "breakdown": breakdown,
        "strongest": ranked[0],
        "weakest": ranked[-1],
        "class_size": len(all_students),
        "comparison": comparison,
        "comparison_message": None if comparison else
            "Class averages need at least 2 student records.",
        "scatter_attendance": [{"x": s.attendance, "y": s.marks_percentage, "id": s.id} for s in all_students],
        "scatter_study": [{"x": s.study_hours, "y": s.marks_percentage, "id": s.id} for s in all_students],
        "corr_attendance": corr_att,
        "corr_study": corr_study,
        "recommendations": recs,
        "history": [p.to_dict() for p in student.predictions[:10]],
        "peers": len(peers),
    }
