import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
from dash import Dash, html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "best_diabetes_model.joblib")
SCALER_PATH = os.path.join(ROOT_DIR, "models", "scaler.joblib")
X_BACKGROUND_PATH = os.path.join(ROOT_DIR, "data", "processed", "X_test.csv")

FEATURE_RANGES = {
    "Pregnancies": (0, 17),
    "Glucose": (44, 199),
    "BloodPressure": (24, 122),
    "SkinThickness": (7, 99),
    "Insulin": (14, 846),
    "BMI": (18.2, 67.1),
    "DiabetesPedigreeFunction": (0.078, 2.42),
    "Age": (21, 81),
}

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
X_bg = pd.read_csv(X_BACKGROUND_PATH)

from sklearn.linear_model import LogisticRegression, Ridge
if isinstance(model, (LogisticRegression, Ridge)):
    explainer = shap.LinearExplainer(model, X_bg)
else:
    explainer = shap.Explainer(model, X_bg)

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server


def create_sliders():
    sliders = []
    for feature, (lo, hi) in FEATURE_RANGES.items():
        default = round((lo + hi) / 2, 1)
        sliders.append(
            dbc.Row([
                dbc.Col(dbc.Label(feature, className="fw-bold"), width=4),
                dbc.Col(
                    dcc.Slider(
                        id=f"input-{feature}",
                        min=float(lo), max=float(hi), step=0.1,
                        value=default,
                        marks={float(lo): str(lo), float(hi): str(hi)},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    width=6,
                ),
                dbc.Col(
                    html.Div(id=f"val-{feature}", className="text-muted text-end"),
                    width=2,
                ),
            ], className="mb-2")
        )
    return sliders


app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Diabetes Risk Predictor", className="text-primary mt-3"), width=12),
    ]),
    dbc.Row([
        dbc.Col(html.P("Adjust patient clinical measurements to assess diabetes risk.", className="lead"), width=12),
    ]),

    dbc.Row([
        # Input sidebar
        dbc.Col([
            html.H4("Patient Data", className="mt-3"),
            html.Hr(),
            *create_sliders(),
        ], width=4, className="bg-light p-3 rounded"),

        # Results panel
        dbc.Col([
            html.H4("Prediction Results", className="mt-3"),
            html.Hr(),
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Probability", className="card-title text-muted"),
                        html.H2(id="output-probability", className="text-primary"),
                    ])
                ]), width=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Risk Level", className="card-title text-muted"),
                        html.H2(id="output-risk"),
                    ])
                ]), width=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Model", className="card-title text-muted"),
                        html.H2(id="output-model", className="text-secondary"),
                    ])
                ]), width=4),
            ], className="g-2 mb-3"),

            html.Div(id="output-alert"),

            html.H4("Feature Contribution (SHAP Waterfall)", className="mt-4"),
            dcc.Graph(id="shap-waterfall", config={"displayModeBar": False}),

            html.H4("Top Contributing Factors", className="mt-4"),
            html.Div(id="top-factors"),

        ], width=8),
    ]),
], fluid=True)


@callback(
    [Output("output-probability", "children"),
     Output("output-risk", "children"),
     Output("output-risk", "className"),
     Output("output-model", "children"),
     Output("output-alert", "children"),
     Output("shap-waterfall", "figure"),
     Output("top-factors", "children")],
    [Input(f"input-{feature}", "value") for feature in FEATURE_RANGES],
)
def predict_and_explain(*values):
    features = list(FEATURE_RANGES.keys())
    patient_df = pd.DataFrame([dict(zip(features, values))])

    patient_scaled = pd.DataFrame(
        scaler.transform(patient_df), columns=features, index=patient_df.index
    )

    probability = model.predict_proba(patient_scaled)[0][1]
    prediction = int(model.predict(patient_scaled)[0])
    model_name = type(model).__name__

    prob_str = f"{probability:.1%}"
    risk_text = "High Risk" if prediction == 1 else "Low Risk"
    risk_class = "text-danger" if prediction == 1 else "text-success"

    if prediction == 1:
        alert = dbc.Alert(
            "High risk of diabetes detected. Consult a healthcare professional.",
            color="danger", dismissable=True,
        )
    else:
        alert = dbc.Alert(
            "Low risk of diabetes detected. Maintain healthy habits.",
            color="success", dismissable=True,
        )

    shap_values = explainer(patient_scaled)
    instance = shap_values[0]

    feature_names = list(instance.feature_names)
    shap_vals = [float(v) for v in instance.values]
    bv = instance.base_values
    base_value = float(bv) if isinstance(bv, (float, np.floating)) else float(np.asarray(bv).flatten()[0])

    sorted_idx = sorted(range(len(shap_vals)), key=lambda i: abs(shap_vals[i]), reverse=True)
    feature_names = [feature_names[i] for i in sorted_idx]
    shap_vals = [shap_vals[i] for i in sorted_idx]

    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in shap_vals]

    waterfall_fig = go.Figure(go.Bar(
        x=shap_vals,
        y=feature_names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in shap_vals],
        textposition="auto",
    ))
    waterfall_fig.update_layout(
        title="SHAP Feature Contributions",
        xaxis_title="SHAP Value (impact on prediction)",
        yaxis_title="",
        template="plotly_white",
        height=350,
        margin=dict(l=120, r=20, t=40, b=40),
    )

    top_factors = []
    for feat, val in zip(feature_names[:5], shap_vals[:5]):
        direction = "increases" if val > 0 else "decreases"
        color = "danger" if val > 0 else "success"
        top_factors.append(
            dbc.Row([
                dbc.Col(html.Strong(feat), width=5),
                dbc.Col(html.Span(f"{direction} risk", className=f"text-{color}"), width=4),
                dbc.Col(html.Code(f"{val:+.4f}"), width=3),
            ], className="mb-1 border-bottom pb-1")
        )

    return prob_str, risk_text, risk_class, model_name, alert, waterfall_fig, top_factors


for feature in FEATURE_RANGES:
    @callback(
        Output(f"val-{feature}", "children"),
        Input(f"input-{feature}", "value"),
    )
    def update_val(value, feat=feature):
        return f"{value:.1f}"


if __name__ == "__main__":
    app.run(debug=True)
