"""Transparent rule-based academic recommendation engine.

Architecture
------------
BaseRecommender defines the interface:  recommend(features, predicted_label) -> list[dict]
RuleBasedRecommender is the default and works fully offline (no API key).

To plug in a real language model later, subclass BaseRecommender (for example
LLMRecommender), call your provider inside `recommend`, and return the same
list-of-dict structure. Then change `get_recommender()` below to return it when
an API key is configured. Nothing else in the app has to change.

Each recommendation dict contains:
    factor, title, priority ("High"/"Medium"/"Low"), severity (0-100 sort key),
    current (display string), target (display string or None),
    actions (list[str]), reason (why it was generated)

Recommendations describe recorded academic factors only. They make no claims
about a student's intelligence, personality or future.
"""
import os

import config

# Thresholds as percentages of each factor's maximum.
# (value below "weak" -> high priority, below "fair" -> medium priority)
THRESHOLDS = {
    "attendance": (65, 75),
    "internal_marks": (50, 65),
    "assignment_marks": (50, 65),
    "previous_semester": (50, 65),
    "practical_marks": (50, 65),
}
STUDY_WEAK, STUDY_FAIR = 2.0, 3.5  # hours per day


def pct(features, name):
    return features[name] / config.FEATURE_MAX[name] * 100


class BaseRecommender:
    name = "base"

    def recommend(self, features, predicted_label):
        raise NotImplementedError


class RuleBasedRecommender(BaseRecommender):
    name = "Rule-based engine"

    def _priority(self, value, weak, fair):
        if value < weak:
            return "High", 60 + (weak - value)
        if value < fair:
            return "Medium", 30 + (fair - value)
        return None, 0

    def recommend(self, features, predicted_label):
        recs = []

        # ---- Attendance ----
        att = features["attendance"]
        prio, sev = self._priority(att, *THRESHOLDS["attendance"])
        if prio:
            target = 75 if att < 75 else 85
            recs.append({
                "factor": "attendance", "title": "Improve class attendance",
                "priority": prio, "severity": sev,
                "current": f"{att:.0f}%", "target": f"{target}% or higher",
                "actions": [
                    f"Aim to raise attendance from {att:.0f}% to at least {target}%.",
                    "Plan the week so that no scheduled lecture or lab is skipped without a reason.",
                    "Collect notes for any missed class on the same day.",
                ],
                "reason": f"Attendance is {att:.0f}%, below the {THRESHOLDS['attendance'][1]}% level "
                          "commonly required by institutions.",
            })

        # ---- Internal marks ----
        v = pct(features, "internal_marks")
        prio, sev = self._priority(v, *THRESHOLDS["internal_marks"])
        if prio:
            recs.append({
                "factor": "internal_marks", "title": "Strengthen internal exam preparation",
                "priority": prio, "severity": sev,
                "current": f"{features['internal_marks']:.1f}/{config.MAX_INTERNAL} ({v:.0f}%)",
                "target": f"{min(v + 15, 100):.0f}% or higher",
                "actions": [
                    "Revise regularly instead of only before exams.",
                    "Take practice tests under timed conditions.",
                    "Revise the most difficult topics before each internal examination.",
                ],
                "reason": f"Internal marks are {v:.0f}% of the maximum, below {THRESHOLDS['internal_marks'][1]}%.",
            })

        # ---- Assignment marks ----
        v = pct(features, "assignment_marks")
        prio, sev = self._priority(v, *THRESHOLDS["assignment_marks"])
        if prio:
            recs.append({
                "factor": "assignment_marks", "title": "Improve assignment completion and quality",
                "priority": prio, "severity": sev,
                "current": f"{features['assignment_marks']:.1f}/{config.MAX_ASSIGNMENT} ({v:.0f}%)",
                "target": f"{min(v + 15, 100):.0f}% or higher",
                "actions": [
                    "Submit assignments on or before the deadline.",
                    "Divide each assignment into smaller tasks with their own mini-deadlines.",
                    "Ask the teacher for feedback on past submissions.",
                ],
                "reason": f"Assignment marks are {v:.0f}% of the maximum, below {THRESHOLDS['assignment_marks'][1]}%.",
            })

        # ---- Study hours ----
        hrs = features["study_hours"]
        prio, sev = self._priority(hrs, STUDY_WEAK, STUDY_FAIR)
        if prio:
            recs.append({
                "factor": "study_hours", "title": "Build a consistent daily study routine",
                "priority": prio, "severity": sev,
                "current": f"{hrs:.1f} h/day", "target": f"{max(hrs + 1, STUDY_FAIR):.1f} h/day",
                "actions": [
                    "Create a realistic daily timetable with fixed study slots.",
                    "Prefer short, regular sessions over occasional long ones.",
                    "Reserve one weekly session to revise the week's topics.",
                ],
                "reason": f"Recorded study time is {hrs:.1f} hours per day, below {STUDY_FAIR} hours.",
            })

        # ---- Practical marks ----
        v = pct(features, "practical_marks")
        prio, sev = self._priority(v, *THRESHOLDS["practical_marks"])
        if prio:
            recs.append({
                "factor": "practical_marks", "title": "Increase laboratory practice",
                "priority": prio, "severity": sev,
                "current": f"{features['practical_marks']:.1f}/{config.MAX_PRACTICAL} ({v:.0f}%)",
                "target": f"{min(v + 15, 100):.0f}% or higher",
                "actions": [
                    "Spend extra time in the lab or on practice exercises.",
                    "Revisit the concepts behind each experiment before the session.",
                    "Write up each experiment soon after performing it.",
                ],
                "reason": f"Practical marks are {v:.0f}% of the maximum, below {THRESHOLDS['practical_marks'][1]}%.",
            })

        # ---- Previous semester ----
        v = features["previous_semester"]
        prio, sev = self._priority(v, *THRESHOLDS["previous_semester"])
        if prio:
            recs.append({
                "factor": "previous_semester", "title": "Address gaps from the previous semester",
                "priority": prio, "severity": sev,
                "current": f"{v:.0f}%", "target": f"{min(v + 10, 100):.0f}% or higher",
                "actions": [
                    "Identify the subjects where marks were lowest last semester.",
                    "Review previous examination papers for those subjects.",
                    "Revise foundational topics that later subjects build on.",
                ],
                "reason": f"Previous semester result is {v:.0f}%, below {THRESHOLDS['previous_semester'][1]}%.",
            })

        recs.sort(key=lambda r: r["severity"], reverse=True)

        # ---- Overall / strengths ----
        if predicted_label == "High" and not recs:
            recs.append({
                "factor": "overall", "title": "Maintain consistency and go deeper",
                "priority": "Low", "severity": 10, "current": "All recorded factors are at healthy levels",
                "target": None,
                "actions": [
                    "Keep the current routine for attendance, study time and assignments.",
                    "Attempt advanced practice problems and explore topics beyond the syllabus.",
                    "Consider helping peers through group study to consolidate understanding.",
                ],
                "reason": "The prediction is High and no recorded factor fell below its threshold.",
            })
        elif predicted_label == "High":
            recs.append({
                "factor": "overall", "title": "Maintain consistency in strong areas",
                "priority": "Low", "severity": 5, "current": "Predicted High", "target": None,
                "actions": [
                    "Keep up the habits behind your stronger factors.",
                    "Use advanced practice to deepen understanding where marks are already good.",
                ],
                "reason": "The prediction is High; the items above are the only factors with room to improve.",
            })
        elif not recs:
            recs.append({
                "factor": "overall", "title": "Keep building on current habits",
                "priority": "Low", "severity": 5, "current": "No factor below threshold", "target": None,
                "actions": [
                    "No single factor is flagged; focus on steady improvement across all of them.",
                    "Discuss targets with a teacher or mentor.",
                ],
                "reason": f"The prediction is {predicted_label}, but every recorded factor is above the "
                          "rule thresholds, so no specific weakness was found.",
            })
        return recs


def get_recommender():
    """Return the active recommender. Rule-based unless an LLM integration is added."""
    # Example hook for later:
    # if os.environ.get("LLM_API_KEY"): return LLMRecommender(...)
    return RuleBasedRecommender()


def factor_breakdown(features):
    """Each factor as a percentage of its target maximum, with a status label."""
    rows = []
    for f in config.FEATURES:
        if f == "study_hours":
            value_pct = min(features[f] / 6.0 * 100, 100)  # 6 h/day treated as full score for display
            note = "scaled against 6 h/day"
        else:
            value_pct = pct(features, f)
            note = None
        status = "Strong" if value_pct >= 75 else "Fair" if value_pct >= 55 else "Needs attention"
        rows.append({
            "factor": f, "label": config.FEATURE_LABELS[f], "value": features[f],
            "max": config.FEATURE_MAX[f] if f != "study_hours" else 6,
            "percent": round(value_pct, 1), "status": status, "note": note,
        })
    return rows
