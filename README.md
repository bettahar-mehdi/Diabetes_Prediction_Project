# Diabetes Prediction — End-to-End ML Pipeline

A production-grade machine learning pipeline predicting diabetes risk using the **PIMA Indians Diabetes Dataset** (768 records). Features medical-domain preprocessing, recall-oriented model evaluation, SHAP interpretability, and an interactive Dash web dashboard.

> **Live Demo:** Run `python app/main.py` → open `http://127.0.0.1:8050`

---

## Problem Statement

Diabetes affects over 400 million people worldwide. Early detection is critical — a missed diagnosis (false negative) can be life-threatening. This project builds a classifier that prioritizes **recall** (catching actual diabetes cases) over raw accuracy, paired with SHAP-based explainability so clinicians can understand *why* the model flagged a patient.

---

## Model Performance (Test Set — 154 patients)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.701 | 0.591 | 0.481 | 0.531 | 0.813 |
| Random Forest | 0.747 | 0.660 | 0.574 | 0.614 | 0.816 |
| **XGBoost** | **0.747** | **0.642** | **0.630** | **0.636** | **0.800** |

> XGBoost selected as best model via **Stratified 5-Fold Cross-Validation** prioritizing Recall and ROC-AUC.

---

## Key Features

### Medical Domain Preprocessing
- Zero values in Glucose, BloodPressure, SkinThickness, Insulin, BMI are replaced with NaN (these are physiologically impossible)
- Group-median imputation by Age and BMI brackets preserves clinical relationships
- StandardScaler normalization for model compatibility

### Model Explainability (SHAP)
- Global feature importance plots showing which clinical measurements drive predictions most
- Per-patient SHAP waterfall charts explaining individual risk assessments
- Top contributing factors ranked by impact magnitude

### Interactive Dashboard (Dash + Plotly)
- Sliders for all 8 clinical features with real-time updates
- Live diabetes risk probability and classification
- Plotly SHAP waterfall visualization per patient
- Bootstrap-responsive UI

---

## Project Structure

```
diabetes_prediction/
├── app/
│   └── main.py                 # Dash web application
├── data/
│   ├── raw/                    # Original PIMA CSV
│   └── processed/              # Scaled train/test splits
├── models/
│   ├── best_diabetes_model.joblib
│   ├── scaler.joblib
│   └── shap_summary.png
├── src/
│   ├── download_data.py        # Fetch PIMA dataset
│   ├── data_preprocessing.py   # Zero→NaN, imputation, scaling
│   ├── train.py                # Model training & evaluation
│   └── explainability.py       # SHAP analysis
├── tests/
│   ├── test_preprocessing.py
│   └── test_model.py
└── requirements.txt
```

---

## Quickstart

```bash
# 1. Clone and setup
git clone https://github.com/YOUR_USERNAME/diabetes_prediction.git
cd diabetes_prediction
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Fetch dataset
python src/download_data.py

# 4. Preprocess and train
python src/train.py

# 5. Run dashboard
python app/main.py            # → http://127.0.0.1:8050

# 6. Run tests
pytest tests/ -v
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| ML & Data | pandas, numpy, scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP, matplotlib |
| Web App | Dash, Plotly, dash-bootstrap-components |
| Testing | pytest |
| Persistence | joblib |

---

## About the Dataset

The **PIMA Indians Diabetes Dataset** contains medical records of **768 female patients** of Pima Indian heritage aged 21+. The `Pregnancies` feature is valid for this population — for male patients, set `Pregnancies = 0`.

| Feature | Description |
|---------|-------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration (mg/dL) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body mass index (weight/(height²)) |
| DiabetesPedigreeFunction | Diabetes pedigree function (genetic risk) |
| Age | Age in years |
| Outcome | 0 = No Diabetes, 1 = Diabetes |

---

## License

MIT
