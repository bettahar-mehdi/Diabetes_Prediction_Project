import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

MODEL_PATH = "models/best_diabetes_model.joblib"


def load_processed_data():
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=200, use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0
        ),
    }


def evaluate_models(X_train, y_train, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    models = get_models()
    results = {}

    for name, model in models.items():
        scores = cross_validate(
            model, X_train, y_train, cv=skf,
            scoring=["recall", "roc_auc"], return_train_score=False
        )
        results[name] = {
            "recall_mean": np.mean(scores["test_recall"]),
            "recall_std": np.std(scores["test_recall"]),
            "roc_auc_mean": np.mean(scores["test_roc_auc"]),
            "roc_auc_std": np.std(scores["test_roc_auc"]),
        }
        print(f"{name:20s} | Recall: {results[name]['recall_mean']:.4f} "
              f"(+/- {results[name]['recall_std']:.4f}) | "
              f"ROC-AUC: {results[name]['roc_auc_mean']:.4f} "
              f"(+/- {results[name]['roc_auc_std']:.4f})")

    return results


def select_best_model(results):
    best = max(results, key=lambda k: (results[k]["recall_mean"], results[k]["roc_auc_mean"]))
    print(f"\nBest model: {best}")
    return best


def train_and_save_best(X_train, y_train, best_name):
    models = get_models()
    best_model = models[best_name]
    best_model.fit(X_train, y_train)

    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved best model to {MODEL_PATH}")
    return best_model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_processed_data()
    results = evaluate_models(X_train, y_train)
    best_name = select_best_model(results)
    train_and_save_best(X_train, y_train, best_name)
