# Intelligent Student Performance Prediction System

A college-level Machine Learning project based on the project synopsis. The system analyzes academic information such as attendance, assignment marks, internal/examination marks and previous performance, then predicts a student's expected performance category.

## Features

- Student information input through a Flask web interface
- Data preprocessing
- Feature selection
- Random Forest based performance prediction
- Performance categories: Low, Average, High
- SQLite database for storing student inputs and predictions
- Simple result and history pages
- Training script and sample dataset
- Model evaluation using accuracy, precision, recall and F1-score

## Project Workflow

Student Data -> Data Preprocessing -> Feature Selection -> ML Model
-> Performance Prediction -> Performance Classification -> Result

## Technology Stack

- Python 3.x
- Flask
- Pandas
- NumPy
- Scikit-learn
- SQLite
- HTML/CSS
- Joblib

## How to Run

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
python src/train_model.py
```

4. Start the Flask application:

```bash
python app.py
```

5. Open the local address shown by Flask in your browser.

## Input Parameters

- Attendance percentage
- Assignment average
- Internal marks
- Previous exam marks
- Study hours per week

## Output

The system predicts one of:

- Low Performance
- Average Performance
- High Performance

## Project Modules

1. User Interface Module
2. Student Data Collection Module
3. Data Preprocessing Module
4. Feature Selection Module
5. Machine Learning Module
6. Performance Prediction Module
7. Result and Database Module

## Note

The included CSV is a small demonstration dataset for project execution. For a final academic submission, replace it with a sufficiently large and institutionally appropriate dataset and retrain the model.
