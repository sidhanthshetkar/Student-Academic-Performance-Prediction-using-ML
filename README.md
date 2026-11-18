# Student Academic Performance Prediction and AI-Based Analysis Using Machine Learning

A Flask web application that predicts whether a student's academic performance is **Low, Medium or High**,
explains the result, and gives prioritised, personalised recommendations.

> **Important:** the bundled training data is **synthetic** (randomly generated) and the training labels come from a
> documented *demonstration* scoring rule. Nothing here describes real students. Replace or validate the rules with real
> historical outcomes before any practical use. Probabilities are model estimates, not guarantees.

## Quick start (Visual Studio Code)

```bash
# 1. Open the project folder in VS Code, then open a terminal (Ctrl+`)
python -m venv venv

# 2. Activate the virtual environment
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) load 60 clearly-labelled DEMO students so the dashboard has data
python seed_demo.py

# 5. Run
python app.py
```

Open <http://127.0.0.1:5000>. The model trains automatically on first use (a few seconds) and is then saved with
joblib in `ml/artifacts/`, so it is **not** retrained per prediction. To retrain manually: `python -m ml.train`
(or use the *Retrain model* button on the Model page).

Run the self-test any time: `python -m tests.smoke_test`

## Project structure

```
app.py                 Flask app: pages, REST APIs, CSV import/export
config.py              Max marks, class names, feature list, DB URI  (edit max marks here)
models.py              SQLAlchemy models: Student, Prediction
validators.py          Shared backend validation (forms, API, CSV)
recommender.py         Rule-based recommendation engine (modular; LLM-ready interface)
analysis.py            Dashboard / per-student statistics from real DB records
seed_demo.py           Loads DEMO students; writes data/sample_students.csv
ml/generate_dataset.py Synthetic data + documented labelling rule
ml/train.py            Pipelines, stratified split, 5-fold CV, model selection, joblib + metrics.json
ml/predictor.py        Loads the saved model once and predicts
templates/, static/    Jinja2 pages, CSS, JS (Bootstrap 5, Chart.js, Font Awesome via CDN)
tests/smoke_test.py    Exercises every route against a temporary database
```

## Pages

| Page | URL | What it does |
|---|---|---|
| Home | `/` | Overview, objectives, features, how prediction works |
| Dashboard | `/dashboard` | Stats, doughnut, bar, scatter and semester charts from DB records; empty-state when no data |
| Students | `/students` | Add / edit / delete (with confirmation), search, filter, profiles |
| Prediction | `/predict` | Predict, class scores, factor breakdown, recommendations, save, history |
| Analysis | `/analysis` | Class comparison (radar), correlations, strongest/weakest factor, recommendations |
| Import/Export | `/data` | Validated CSV upload, template, student and prediction exports |
| Model | `/evaluation` | Accuracy, macro precision/recall/F1, confusion matrices, classification reports |
| About | `/about` | Stack, limitations, future work |

REST endpoints: `POST /api/predict`, `POST /api/predictions`, `GET /api/students/<id>/predictions`,
`GET /api/dashboard`, `GET /api/evaluation`.

## Machine-learning design

* **Features (6):** attendance, internal marks, assignment marks, previous semester %, study hours/day, practical marks.
* **Classes:** Low, Medium, High.
* **Models:** Decision Tree, Logistic Regression (standardised inputs), Random Forest, each in a scikit-learn `Pipeline`
  (median imputer → optional scaler → classifier).
* **Evaluation:** stratified 80/20 split; 5-fold stratified cross-validation on the training part; the model with the
  best mean **macro-F1** is selected (the test set is never used for selection). Reported metrics are macro-averaged.
* **Labels:** `ml/generate_dataset.py` builds a weighted demonstration score plus random noise (standing in for
  unrecorded factors) and cuts it at 55 / 70. The score and label are used **only** to create the target column and are
  never fed to the model, so there is no direct target leakage. Because the labels come from a rule over these same
  inputs, high accuracy on synthetic data does **not** imply accuracy on real students.

## Recommendation engine

`recommender.py` compares each factor with visible thresholds (e.g. attendance < 65 % = high priority, < 75 % = medium),
sorts by severity, and states **why** each item appears. It works offline with no API key. To add a language model later,
subclass `BaseRecommender`, return the same list-of-dicts, and switch `get_recommender()`.

## CSV format

Download the template from the Import/Export page. Required columns:
`name, roll_number, department, semester, attendance, internal_marks, assignment_marks, previous_semester, study_hours, practical_marks`.
Rows with invalid values or duplicate roll numbers are skipped and listed with their line number; nothing invalid is imported silently.

## Configuration

* Maximum marks: `MAX_INTERNAL`, `MAX_ASSIGNMENT`, `MAX_PRACTICAL`, `MAX_PREVIOUS` in `config.py`.
  If you change them, re-run `python -m ml.train` after regenerating data (`python -m ml.generate_dataset`).
* Database: SQLite by default (`instance/students.db`). Set the `DATABASE_URL` environment variable to use MySQL
  (e.g. `mysql+pymysql://user:pass@localhost/students`, after `pip install pymysql`).

## Suggested demo flow

1. `python seed_demo.py`, then open the Dashboard.
2. Add a student on the Students page; note the automatic prediction.
3. Open Prediction, pick the student, change values, press Predict, then Save.
4. Open Analysis for the student; show the radar chart and recommendations.
5. Show the Model page and explain the comparison, confusion matrix and the synthetic-data limitation.


## Login system (teacher and student)

- Teacher pages require a teacher account. Create one from the project root with `python create_teacher.py`; the script securely prompts for the teacher name, email and password. Teacher self-registration is intentionally disabled.
- A teacher first adds a student record from the Students page. The student can then open Student Registration and register using the exact name and roll number already recorded by the teacher.
- Students can view only their own academic summary and recommendations at the Student Dashboard. Teacher management, import/export, prediction and model evaluation pages are restricted to teacher accounts.
- Both account types use the same login page (`/login`) and the Logout button in the top navigation. Passwords are stored as hashes, not plain text.

After updating an existing copy of the project, run `pip install -r requirements.txt`, create a teacher with `python create_teacher.py`, then start the app with `python app.py`. Existing student and prediction data are retained; SQLAlchemy creates the new `users` table automatically.
