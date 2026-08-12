import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np
import pandas as pd
import pytest


def load_model_and_data():
    model = joblib.load("models/best_diabetes_model.joblib")
    X_test = pd.read_csv("data/processed/X_test.csv")
    return model, X_test


def test_probability_bounds():
    model, X_test = load_model_and_data()
    probs = model.predict_proba(X_test)[:, 1]
    assert (probs >= 0).all() and (probs <= 1).all(), \
        "Probabilities outside [0, 1] range"


def test_predictions_binary():
    model, X_test = load_model_and_data()
    preds = model.predict(X_test)
    assert set(np.unique(preds)).issubset({0, 1}), \
        "Predictions must be binary (0 or 1)"


def test_output_shape_matches_input():
    model, X_test = load_model_and_data()
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    assert len(preds) == len(X_test), "Prediction count mismatch"
    assert probs.shape == (len(X_test), 2), f"Expected (N,2), got {probs.shape}"


def test_probability_sums_to_one():
    model, X_test = load_model_and_data()
    probs = model.predict_proba(X_test)
    sums = probs.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-6)
