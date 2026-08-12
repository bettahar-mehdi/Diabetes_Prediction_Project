import os
import pandas as pd

DATA_DIR = "data/raw"
PIMA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]


def fetch_pima_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    df = pd.read_csv(PIMA_URL, names=COLUMNS)
    output_path = os.path.join(DATA_DIR, "pima_diabetes.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path} ({len(df)} records)")
    return output_path


if __name__ == "__main__":
    fetch_pima_data()
