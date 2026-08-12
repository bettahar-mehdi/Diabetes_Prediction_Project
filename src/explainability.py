import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
from train import load_processed_data

MODEL_PATH = "models/best_diabetes_model.joblib"
SHAP_PLOT_PATH = "models/shap_summary.png"


def load_model():
    return joblib.load(MODEL_PATH)


def generate_shap_summary(model, X, save_path=SHAP_PLOT_PATH):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    from sklearn.linear_model import LogisticRegression
    if isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, X)
    else:
        explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot saved to {save_path}")
    return shap_values


def get_top_factors(model, patient_df, X_background, top_n=5):
    from sklearn.linear_model import LogisticRegression
    if isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, X_background)
    else:
        explainer = shap.Explainer(model, X_background)
    shap_values = explainer(patient_df)

    instance = shap_values[0]
    feature_contrib = sorted(
        zip(instance.feature_names, instance.values),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    return feature_contrib[:top_n]


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_processed_data()
    model = load_model()
    generate_shap_summary(model, X_test)
