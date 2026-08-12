# AGENTS.md — Diabetes Prediction Project

## Project Overview
End-to-end ML project predicting diabetes risk using the **PIMA Indians Diabetes dataset** (768 records). Emphasizes medical-domain correctness, recall-oriented evaluation, and model interpretability (SHAP) via a Dash dashboard.

## Tech Stack
- Python, pandas, numpy, scikit-learn, XGBoost, LightGBM
- SHAP (feature importance & per-patient explanations)
- Dash + dash-bootstrap-components + Plotly (interactive web UI)
- joblib (model persistence)
- pytest (testing)

## Critical Domain Knowledge
In the PIMA dataset, **zero values in these columns are physiologically impossible and represent missing data**:
- `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`

Preprocessing **MUST** replace these zeros with NaN before imputation. Never treat them as valid measurements.

## Directory Structure
```
data/
├── raw/              # Original PIMA CSV (downloaded)
└── processed/        # Cleaned datasets
notebooks/            # EDA (01_eda_and_insights.ipynb)
src/                  # Core library: preprocessing, training, evaluation, explainability
models/               # Saved .joblib models + SHAP plots
app/                  # Dash dashboard (main.py)
tests/                # pytest suite
```

## Evaluation Priorities
- **Primary metrics**: Recall, ROC-AUC — false negatives are dangerous in medical diagnosis
- **Do NOT** use accuracy as the primary metric (class imbalance makes it misleading)
- Use **Stratified K-Fold CV** to preserve class ratios

## Build Order
1. `requirements.txt` + `src/download_data.py` — fetch PIMA data to `data/raw/pima_diabetes.csv`
2. `src/data_preprocessing.py` — zero→NaN, group-median imputation (by Age/BMI brackets), `StandardScaler`, train/test split
3. `src/train.py` — Logistic Regression, Random Forest, XGBoost; save best to `models/best_diabetes_model.joblib`
4. `src/explainability.py` — SHAP summary plot (`models/shap_summary.png`) + per-patient top factors
5. `app/main.py` — Dash UI: sidebar sliders → risk probability → SHAP waterfall (Plotly)
6. `tests/` — pytest: imputation correctness, output dimensions, probability bounds [0,1]

## Key Commands
```bash
pip install -r requirements.txt
python src/download_data.py       # Fetch PIMA dataset
python src/train.py               # Train and save best model
python app/main.py                 # Launch dashboard (Dash dev server)
pytest tests/                     # Run test suite
```

## PIMA Dataset Reference
- **Download URL**: `https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv`
- **Columns**: `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`
- **Target**: `Outcome` (1 = diabetes, 0 = no diabetes)

## Conventions
- PEP 8 compliant, modular single-responsibility functions
- Model artifacts saved/loaded via `joblib`
- All source modules live in `src/` with `__init__.py`
