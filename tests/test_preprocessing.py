import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from src.data_preprocessing import (
    replace_zeros_with_nan,
    ZERO_INVALID_COLUMNS,
    preprocess_pipeline,
)


@pytest.fixture
def raw_data():
    return pd.read_csv("data/raw/pima_diabetes.csv")


def test_zeros_replaced_with_nan(raw_data):
    df = replace_zeros_with_nan(raw_data)
    for col in ZERO_INVALID_COLUMNS:
        assert (df[col] != 0).all() or df[col].isna().any(), \
            f"Zeros not replaced in {col}"


def test_no_zeros_after_imputation(raw_data):
    from src.data_preprocessing import add_grouping_brackets, impute_group_medians
    df = replace_zeros_with_nan(raw_data)
    df = add_grouping_brackets(df)
    df = impute_group_medians(df)
    for col in ZERO_INVALID_COLUMNS:
        assert not (df[col] == 0).any(), f"Zero still present in {col} after imputation"
        assert not df[col].isna().any(), f"NaN still present in {col} after imputation"


def test_output_dimensions():
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline()
    assert X_train.shape[1] == 8, f"Expected 8 features, got {X_train.shape[1]}"
    assert X_test.shape[1] == 8, f"Expected 8 features, got {X_test.shape[1]}"
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert X_train.shape[0] + X_test.shape[0] == 768


def test_class_distribution_preserved():
    X_train, X_test, y_train, y_test, _ = preprocess_pipeline()
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.05, \
        f"Class ratio mismatch: train={train_ratio:.3f}, test={test_ratio:.3f}"
