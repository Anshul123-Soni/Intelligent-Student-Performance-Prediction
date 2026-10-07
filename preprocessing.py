import pandas as pd

FEATURES = [
    "attendance",
    "assignment_average",
    "internal_marks",
    "previous_exam_marks",
    "study_hours",
]

TARGET = "performance_category"

def load_and_prepare_data(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates().copy()

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET] = df[TARGET].astype(str).str.strip()

    df = df.dropna(subset=FEATURES + [TARGET])

    return df
