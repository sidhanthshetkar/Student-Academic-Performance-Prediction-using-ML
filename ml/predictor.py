"""Load the saved model once and make predictions."""
import json
import os
import threading

import joblib
import pandas as pd

import config
from ml import train as trainer

_lock = threading.Lock()
_bundle = None


def get_bundle():
    """Return the cached model bundle, training it first if no saved model exists."""
    global _bundle
    with _lock:
        if _bundle is None:
            if not os.path.exists(trainer.MODEL_PATH):
                trainer.train(verbose=False)
            _bundle = joblib.load(trainer.MODEL_PATH)
        return _bundle


def reload_bundle():
    global _bundle
    with _lock:
        _bundle = None
    return get_bundle()


def predict(features):
    """features: dict with the six keys in config.FEATURES.

    Returns {"label", "probabilities" (dict or None), "model_name"}.
    Probabilities are the model's estimated class scores, not guarantees.
    """
    bundle = get_bundle()
    pipe = bundle["pipeline"]
    row = pd.DataFrame([[float(features[f]) for f in config.FEATURES]], columns=config.FEATURES)
    label = str(pipe.predict(row)[0])

    probs = None
    if hasattr(pipe, "predict_proba"):
        raw = pipe.predict_proba(row)[0]
        classes = list(pipe.named_steps["clf"].classes_)
        probs = {c: round(float(p) * 100, 1) for c, p in zip(classes, raw)}
        probs = {c: probs.get(c, 0.0) for c in config.CLASSES}

    return {"label": label, "probabilities": probs, "model_name": bundle["model_name"]}


def get_metrics():
    if not os.path.exists(trainer.METRICS_PATH):
        get_bundle()
    with open(trainer.METRICS_PATH) as fh:
        return json.load(fh)
