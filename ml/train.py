"""Train, compare and persist the classifiers.

Run:  python -m ml.train

Steps
-----
1. Load (or generate) the SYNTHETIC dataset.
2. Stratified train/test split (80/20).
3. Build one scikit-learn Pipeline per algorithm (imputer -> optional scaler -> model).
4. 5-fold stratified cross-validation on the TRAINING portion; the model with the
   best mean macro-F1 is selected (the test set is NOT used for selection).
5. Evaluate every model on the held-out test set with macro-averaged metrics.
6. Save the selected pipeline (joblib) and all metrics (JSON).
"""
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config
from ml.generate_dataset import generate

RANDOM_STATE = 42
MODEL_PATH = os.path.join(config.MODEL_DIR, "model.joblib")
METRICS_PATH = os.path.join(config.MODEL_DIR, "metrics.json")
DATASET_PATH = os.path.join(config.DATA_DIR, "synthetic_training_data.csv")


def build_models():
    return {
        "Decision Tree": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", "passthrough"),  # trees do not need scaling
            ("clf", DecisionTreeClassifier(max_depth=5, min_samples_leaf=10,
                                           random_state=RANDOM_STATE)),
        ]),
        "Logistic Regression": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),  # scaling matters for LR
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", "passthrough"),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                           random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
    }


def load_dataset():
    if os.path.exists(DATASET_PATH):
        return pd.read_csv(DATASET_PATH)
    df = generate()
    df.to_csv(DATASET_PATH, index=False)
    return df


def feature_importance(pipe):
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        vals = np.abs(clf.coef_).mean(axis=0)  # mean |coefficient| on standardised inputs
        vals = vals / vals.sum()
    else:
        return None
    return {f: round(float(v), 4) for f, v in zip(config.FEATURES, vals)}


def train(verbose=True):
    df = load_dataset()
    X, y = df[config.FEATURES], df["performance"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results, fitted = {}, {}

    for name, pipe in build_models().items():
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        report = classification_report(y_test, pred, labels=config.CLASSES,
                                       output_dict=True, zero_division=0)
        results[name] = {
            "cv_f1_macro_mean": round(float(cv_scores.mean()), 4),
            "cv_f1_macro_std": round(float(cv_scores.std()), 4),
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "precision_macro": round(precision_score(y_test, pred, average="macro", zero_division=0), 4),
            "recall_macro": round(recall_score(y_test, pred, average="macro", zero_division=0), 4),
            "f1_macro": round(f1_score(y_test, pred, average="macro", zero_division=0), 4),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=config.CLASSES).tolist(),
            "classification_report": report,
            "feature_importance": feature_importance(pipe),
        }
        fitted[name] = pipe
        if verbose:
            r = results[name]
            print(f"{name:20s} CV F1={r['cv_f1_macro_mean']:.3f}±{r['cv_f1_macro_std']:.3f} "
                  f"| test acc={r['accuracy']:.3f} macro-F1={r['f1_macro']:.3f}")

    best = max(results, key=lambda n: results[n]["cv_f1_macro_mean"])

    bundle = {
        "pipeline": fitted[best],
        "model_name": best,
        "features": config.FEATURES,
        "classes": config.CLASSES,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, MODEL_PATH)

    metrics = {
        "selected_model": best,
        "selection_criterion": "highest mean macro-F1 in 5-fold stratified cross-validation on the training split",
        "trained_at": bundle["trained_at"],
        "dataset": {
            "type": "SYNTHETIC (generated; not real student data)",
            "rows": int(len(df)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "class_counts": {k: int(v) for k, v in y.value_counts().items()},
        },
        "labelling_note": ("Labels come from a documented demonstration scoring rule plus random noise "
                           "(see ml/generate_dataset.py). Validate against real outcomes before practical use."),
        "classes": config.CLASSES,
        "models": results,
    }
    with open(METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=2)

    if verbose:
        print(f"\nSelected model: {best}\nSaved: {MODEL_PATH}\nSaved: {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    train()
