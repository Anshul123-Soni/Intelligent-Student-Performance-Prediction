from pathlib import Path
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import FEATURES, TARGET, load_and_prepare_data

DATA_PATH = ROOT  / "student_data.csv"
MODEL_PATH = ROOT / "model" / "student_performance_model.pkl"

def main():
    df = load_and_prepare_data(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    print("Model Evaluation")
    print("-----------------")
    print(f"Accuracy : {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, average='weighted', zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, predictions, average='weighted', zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_test, predictions, average='weighted', zero_division=0):.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
