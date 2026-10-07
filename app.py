from pathlib import Path
import sqlite3
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "student_performance_model.pkl"
DB_PATH = BASE_DIR / "database" / "students.db"

app = Flask(__name__)

FEATURES = [
    "attendance",
    "assignment_average",
    "internal_marks",
    "previous_exam_marks",
    "study_hours",
]

def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                attendance REAL NOT NULL,
                assignment_average REAL NOT NULL,
                internal_marks REAL NOT NULL,
                previous_exam_marks REAL NOT NULL,
                study_hours REAL NOT NULL,
                prediction TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            '''
        )

def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    model = load_model()
    if model is None:
        return render_template(
            "index.html",
            error="Model not found. Run: python src/train_model.py"
        )

    try:
        name = request.form["name"].strip()
        values = {
            "attendance": float(request.form["attendance"]),
            "assignment_average": float(request.form["assignment_average"]),
            "internal_marks": float(request.form["internal_marks"]),
            "previous_exam_marks": float(request.form["previous_exam_marks"]),
            "study_hours": float(request.form["study_hours"]),
        }

        if not name:
            raise ValueError("Student name is required.")

        if not 0 <= values["attendance"] <= 100:
            raise ValueError("Attendance must be between 0 and 100.")
        if not 0 <= values["assignment_average"] <= 100:
            raise ValueError("Assignment average must be between 0 and 100.")
        if not 0 <= values["internal_marks"] <= 100:
            raise ValueError("Internal marks must be between 0 and 100.")
        if not 0 <= values["previous_exam_marks"] <= 100:
            raise ValueError("Previous exam marks must be between 0 and 100.")
        if values["study_hours"] < 0:
            raise ValueError("Study hours cannot be negative.")

        input_df = pd.DataFrame([values], columns=FEATURES)
        prediction = model.predict(input_df)[0]

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                '''
                INSERT INTO predictions
                (name, attendance, assignment_average, internal_marks,
                 previous_exam_marks, study_hours, prediction)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    name,
                    values["attendance"],
                    values["assignment_average"],
                    values["internal_marks"],
                    values["previous_exam_marks"],
                    values["study_hours"],
                    str(prediction),
                ),
            )

        return render_template(
            "result.html",
            name=name,
            values=values,
            prediction=prediction,
        )

    except (ValueError, KeyError) as exc:
        return render_template("index.html", error=str(exc))

@app.route("/history")
def history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC"
        ).fetchall()
    return render_template("history.html", rows=rows)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
