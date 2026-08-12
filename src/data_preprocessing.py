import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SCALER_PATH = "models/scaler.joblib"

ZERO_INVALID_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_data(filepath="data/raw/pima_diabetes.csv"):
    return pd.read_csv(filepath)


def replace_zeros_with_nan(df):
    df = df.copy()
    for col in ZERO_INVALID_COLUMNS:
        df[col] = df[col].replace(0, np.nan)
    return df


def add_grouping_brackets(df):
    df = df.copy()
    df["AgeBracket"] = pd.cut(df["Age"], bins=[20, 30, 40, 50, 60, 100],
                              labels=["21-30", "31-40", "41-50", "51-60", "61+"])
    df["BMIBracket"] = pd.cut(df["BMI"], bins=[0, 18.5, 25, 30, 35, 100],
                              labels=["Under", "Normal", "Over", "ObeseI", "ObeseII+"])
    return df


def impute_group_medians(df):
    df = df.copy()
    for col in ZERO_INVALID_COLUMNS:
        df[col] = df.groupby(["AgeBracket", "BMIBracket"], observed=True)[col] \
            .transform(lambda x: x.fillna(x.median()))
    for col in ZERO_INVALID_COLUMNS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def preprocess_pipeline(filepath="data/raw/pima_diabetes.csv", test_size=0.2, random_state=42):
    df = load_data(filepath)
    df = replace_zeros_with_nan(df)
    df = add_grouping_brackets(df)
    df = impute_group_medians(df)
    df = df.drop(columns=["AgeBracket", "BMIBracket"])

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    X_train_scaled.to_csv("data/processed/X_train.csv", index=False)
    X_test_scaled.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Class distribution (train): {y_train.value_counts().to_dict()}")
    print(f"Class distribution (test):  {y_test.value_counts().to_dict()}")
