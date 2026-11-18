"""Synthetic training-data generator.

!!! IMPORTANT !!!
The data produced here is SYNTHETIC. It does NOT describe real students.
It exists only so the project can be trained and demonstrated immediately.

How the data is made
--------------------
1. A hidden "engagement" variable is drawn for each simulated student. It makes
   attendance, study hours and marks realistically correlated with each other.
2. The six input features are generated from it with random variation and
   clipped to their valid ranges (see config.py).
3. A DEMONSTRATION labelling rule assigns Low / Medium / High:

       score = 0.20 * attendance%
             + 0.25 * internal%
             + 0.10 * assignment%
             + 0.25 * previous semester %
             + 0.10 * practical%
             + 0.10 * study-hours score (6 h/day or more = 100)
             + noise  (Normal, sd = 4 points)

       Low    : score <  55
       Medium : 55 <= score < 70
       High   : score >= 70

   The noise term stands in for factors that are not recorded (health, exam-day
   luck, teaching quality ...), so the label is NOT a deterministic function of
   the inputs and the model cannot simply "read the formula".

Leakage note
------------
The score and the label are used ONLY to create the target column. Neither is
ever passed to the model as an input feature.

Before real use
---------------
These rules are an assumption made for demonstration. Replace them with, or
validate them against, real historical outcomes (e.g. actual end-of-semester
results) before relying on any prediction.
"""
import numpy as np
import pandas as pd

import config

SEED = 42
LOW_CUTOFF = 55.0
HIGH_CUTOFF = 70.0
NOISE_SD = 4.0


def demo_score(df, rng=None):
    """Weighted demonstration score (0-100). Used for label creation only."""
    study_score = np.clip(df["study_hours"] / 6.0, 0, 1) * 100
    score = (
        0.20 * df["attendance"]
        + 0.25 * df["internal_marks"] / config.MAX_INTERNAL * 100
        + 0.10 * df["assignment_marks"] / config.MAX_ASSIGNMENT * 100
        + 0.25 * df["previous_semester"] / config.MAX_PREVIOUS * 100
        + 0.10 * df["practical_marks"] / config.MAX_PRACTICAL * 100
        + 0.10 * study_score
    )
    if rng is not None:
        score = score + rng.normal(0, NOISE_SD, len(df))
    return score


def label_from_score(score):
    return np.where(score < LOW_CUTOFF, "Low", np.where(score < HIGH_CUTOFF, "Medium", "High"))


def generate(n=1500, seed=SEED):
    rng = np.random.default_rng(seed)
    engagement = np.clip(rng.normal(0.60, 0.20, n), 0.02, 1.0)  # hidden variable

    def noisy(mean_frac, sd_frac, maxv):
        return np.clip((mean_frac + rng.normal(0, sd_frac, n)) * maxv, 0, maxv)

    df = pd.DataFrame({
        "attendance": np.clip(35 + 65 * engagement + rng.normal(0, 8, n), 20, 100),
        "internal_marks": noisy(engagement * 0.85 + 0.10, 0.10, config.MAX_INTERNAL),
        "assignment_marks": noisy(engagement * 0.80 + 0.15, 0.12, config.MAX_ASSIGNMENT),
        "previous_semester": np.clip((engagement * 0.75 + 0.15 + rng.normal(0, 0.09, n)) * 100, 25, 100),
        "study_hours": np.clip(engagement * 5.5 + rng.normal(0, 1.0, n), 0, 12),
        "practical_marks": noisy(engagement * 0.75 + 0.20, 0.09, config.MAX_PRACTICAL),
    }).round(2)

    df["performance"] = label_from_score(demo_score(df, rng))
    return df


if __name__ == "__main__":
    data = generate()
    path = f"{config.DATA_DIR}/synthetic_training_data.csv"
    data.to_csv(path, index=False)
    print(f"Wrote {len(data)} SYNTHETIC rows to {path}")
    print(data["performance"].value_counts())
