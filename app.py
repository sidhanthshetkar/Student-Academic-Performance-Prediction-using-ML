"""Flask application: pages, REST APIs, CSV import/export.

Run:  python app.py      then open  http://127.0.0.1:5000
"""
import csv
import io
import json
from functools import wraps

import pandas as pd
from flask import (Flask, Response, abort, flash, jsonify, redirect, render_template,
                   request, session, url_for)

import config
from analysis import dashboard_data, student_analysis
from ml import predictor
from models import Prediction, Student, User, db
from recommender import factor_breakdown, get_recommender
from validators import validate_features, validate_student

CSV_COLUMNS = ["name", "roll_number", "department", "semester"] + config.FEATURES


def record_prediction_for(student, source="auto"):
    """Predict for a saved student and add the Prediction row to the session (caller commits)."""
    result = predictor.predict(student.features())
    pred = Prediction(
        student_id=student.id, predicted_label=result["label"],
        probabilities_json=json.dumps(result["probabilities"]) if result["probabilities"] else None,
        model_name=result["model_name"], source=source, **student.features(),
    )
    db.session.add(pred)
    return pred



def current_user():
    """Return the signed-in account, if its session is still valid."""
    user_id = session.get("user_id")
    return db.session.get(User, user_id) if user_id else None


def role_required(role):
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            user = current_user()
            if not user:
                flash("Please log in to continue.", "warning")
                return redirect(url_for("login", next=request.path))
            if user.role != role:
                flash("You do not have permission to open that page.", "danger")
                return redirect(url_for("student_dashboard" if user.role == "student" else "dashboard"))
            return view(*args, **kwargs)
        return wrapped
    return decorator

def create_app():
    app = Flask(__name__)
    app.config.from_object(config.Config)
    db.init_app(app)
    with app.app_context():
        db.create_all()

    record_prediction = record_prediction_for

    @app.context_processor
    def inject_globals():
        return {"cfg": config, "classes": config.CLASSES, "current_user": current_user()}

    # -------------------------------------------------------------------- pages
    @app.route("/")
    def index():
        user = current_user()
        if user:
            return redirect(url_for("student_dashboard" if user.role == "student" else "dashboard"))
        return render_template("index.html")

    @app.route("/about")
    def about():
        return render_template("about.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if current_user():
            user = current_user()
            return redirect(url_for("student_dashboard" if user.role == "student" else "dashboard"))
        if request.method == "POST":
            email = (request.form.get("email") or "").strip().lower()
            password = request.form.get("password") or ""
            role = request.form.get("role") or ""
            user = User.query.filter_by(email=email, role=role).first()
            if user and user.check_password(password):
                session.clear()
                session["user_id"] = user.id
                flash(f"Welcome, {user.name}!", "success")
                target = request.args.get("next") or request.form.get("next")
                if target and target.startswith("/") and not target.startswith("//"):
                    return redirect(target)
                return redirect(url_for("student_dashboard" if user.role == "student" else "dashboard"))
            flash("Invalid email, password or account type.", "danger")
        return render_template("login.html", next=request.args.get("next", ""))

    @app.route("/register/student", methods=["GET", "POST"])
    def student_register():
        if current_user():
            return redirect(url_for("index"))
        if request.method == "POST":
            name = (request.form.get("name") or "").strip()
            email = (request.form.get("email") or "").strip().lower()
            roll = (request.form.get("roll_number") or "").strip()
            password = request.form.get("password") or ""
            confirm = request.form.get("confirm_password") or ""
            student = Student.query.filter(db.func.lower(Student.roll_number) == roll.lower()).first() if roll else None
            errors = []
            if not name or not email or not roll or not password:
                errors.append("Please complete every field.")
            if len(password) < 8:
                errors.append("Password must contain at least 8 characters.")
            if password != confirm:
                errors.append("The passwords do not match.")
            if User.query.filter_by(email=email).first():
                errors.append("An account with this email already exists.")
            if not student:
                errors.append("Roll number not found. Ask your teacher to add your student record first.")
            elif student.name.strip().casefold() != name.casefold():
                errors.append("Name does not match the student record. Check with your teacher.")
            elif User.query.filter_by(student_id=student.id).first():
                errors.append("An account is already registered for this roll number.")
            if errors:
                for error in errors:
                    flash(error, "danger")
            else:
                user = User(name=name, email=email, role="student", student_id=student.id)
                user.set_password(password)
                db.session.add(user)
                db.session.commit()
                flash("Student account created. You can now log in.", "success")
                return redirect(url_for("login"))
        return render_template("register.html")

    @app.route("/logout", methods=["POST"])
    def logout():
        session.clear()
        flash("You have been logged out.", "info")
        return redirect(url_for("login"))

    @app.route("/student/dashboard")
    @role_required("student")
    def student_dashboard():
        user = current_user()
        student = db.session.get(Student, user.student_id) if user else None
        if not student:
            session.clear()
            flash("Your student record is no longer available. Please contact your teacher.", "danger")
            return redirect(url_for("login"))
        recs = get_recommender().recommend(student.features(), student.predicted_label or "Medium")
        return render_template("student_dashboard.html", s=student, recs=recs)

    @app.route("/dashboard")
    @role_required("teacher")
    def dashboard():
        data = dashboard_data()
        recent = Prediction.query.order_by(Prediction.created_at.desc(), Prediction.id.desc()).limit(8).all()
        return render_template("dashboard.html", data=data, recent=recent)

    # ----------------------------------------------------------------- students
    @app.route("/students")
    @role_required("teacher")
    def students():
        q = (request.args.get("q") or "").strip()
        dept = request.args.get("department") or ""
        sem = request.args.get("semester") or ""
        perf = request.args.get("performance") or ""

        query = Student.query
        if q:
            like = f"%{q}%"
            query = query.filter(db.or_(Student.name.ilike(like), Student.roll_number.ilike(like)))
        if dept:
            query = query.filter(Student.department == dept)
        if sem.isdigit():
            query = query.filter(Student.semester == int(sem))
        rows = query.order_by(Student.roll_number).all()
        if perf in config.CLASSES:
            rows = [s for s in rows if s.predicted_label == perf]
        elif perf == "Unpredicted":
            rows = [s for s in rows if s.predicted_label is None]

        departments = [d[0] for d in db.session.query(Student.department).distinct().order_by(Student.department)]
        semesters = [s[0] for s in db.session.query(Student.semester).distinct().order_by(Student.semester)]
        return render_template("students.html", students=rows, departments=departments,
                               semesters=semesters, q=q, dept=dept, sem=sem, perf=perf)

    def save_student_from_form(student=None):
        clean, errors = validate_student(request.form)
        if "roll_number" in clean:
            dup = Student.query.filter(Student.roll_number == clean["roll_number"])
            if student:
                dup = dup.filter(Student.id != student.id)
            if dup.first():
                errors["roll_number"] = "This roll number already exists."
        if errors:
            return None, errors, request.form
        is_new = student is None
        if is_new:
            student = Student(**clean)
            db.session.add(student)
        else:
            for k, v in clean.items():
                setattr(student, k, v)
        db.session.flush()
        record_prediction(student, source="auto")
        db.session.commit()
        return student, {}, None

    @app.route("/students/add", methods=["GET", "POST"])
    @role_required("teacher")
    def student_add():
        if request.method == "POST":
            student, errors, form = save_student_from_form()
            if student:
                flash(f"Student {student.name} added.", "success")
                return redirect(url_for("student_profile", student_id=student.id))
            return render_template("student_form.html", student=None, errors=errors, form=form)
        return render_template("student_form.html", student=None, errors={}, form={})

    @app.route("/students/<int:student_id>/edit", methods=["GET", "POST"])
    @role_required("teacher")
    def student_edit(student_id):
        student = db.get_or_404(Student, student_id)
        if request.method == "POST":
            saved, errors, form = save_student_from_form(student)
            if saved:
                flash("Student updated and prediction refreshed.", "success")
                return redirect(url_for("student_profile", student_id=student.id))
            return render_template("student_form.html", student=student, errors=errors, form=form)
        return render_template("student_form.html", student=student, errors={}, form=student.to_dict())

    @app.route("/students/<int:student_id>/delete", methods=["POST"])
    @role_required("teacher")
    def student_delete(student_id):
        student = db.get_or_404(Student, student_id)
        name = student.name
        if student.account:
            db.session.delete(student.account)
        db.session.delete(student)
        db.session.commit()
        flash(f"Deleted {name}.", "info")
        return redirect(url_for("students"))

    @app.route("/students/<int:student_id>")
    @role_required("teacher")
    def student_profile(student_id):
        student = db.get_or_404(Student, student_id)
        recs = get_recommender().recommend(student.features(), student.predicted_label or "Medium")
        return render_template("student_profile.html", s=student, breakdown=factor_breakdown(student.features()),
                               recs=recs)

    # ----------------------------------------------------------------- predict
    @app.route("/predict")
    @role_required("teacher")
    def predict_page():
        students_list = Student.query.order_by(Student.roll_number).all()
        selected = request.args.get("student_id", type=int)
        return render_template("predict.html", students=students_list, selected=selected,
                               metrics=predictor.get_metrics())

    @app.route("/api/predict", methods=["POST"])
    @role_required("teacher")
    def api_predict():
        data = request.get_json(silent=True) or request.form.to_dict()
        clean, errors = validate_features(data)
        if errors:
            return jsonify({"ok": False, "errors": errors}), 400
        result = predictor.predict(clean)
        recs = get_recommender().recommend(clean, result["label"])
        return jsonify({
            "ok": True, "label": result["label"], "probabilities": result["probabilities"],
            "model_name": result["model_name"], "breakdown": factor_breakdown(clean),
            "recommendations": recs, "inputs": clean,
        })

    @app.route("/api/predictions", methods=["POST"])
    @role_required("teacher")
    def api_save_prediction():
        data = request.get_json(silent=True) or {}
        student = db.session.get(Student, data.get("student_id"))
        if not student:
            return jsonify({"ok": False, "error": "Select a saved student to store the prediction."}), 400
        clean, errors = validate_features(data)
        if errors:
            return jsonify({"ok": False, "errors": errors}), 400
        result = predictor.predict(clean)
        pred = Prediction(
            student_id=student.id, predicted_label=result["label"],
            probabilities_json=json.dumps(result["probabilities"]) if result["probabilities"] else None,
            model_name=result["model_name"], source="manual", **clean)
        db.session.add(pred)
        db.session.commit()
        return jsonify({"ok": True, "prediction": pred.to_dict()})

    @app.route("/api/students/<int:student_id>/predictions")
    @role_required("teacher")
    def api_student_predictions(student_id):
        student = db.get_or_404(Student, student_id)
        return jsonify({"student": student.to_dict(), "predictions": [p.to_dict() for p in student.predictions]})

    # ---------------------------------------------------------------- analysis
    @app.route("/analysis")
    @role_required("teacher")
    def analysis_index():
        students_list = Student.query.order_by(Student.name).all()
        return render_template("analysis.html", students=students_list, data=None)

    @app.route("/analysis/<int:student_id>")
    @role_required("teacher")
    def analysis_student(student_id):
        student = db.get_or_404(Student, student_id)
        students_list = Student.query.order_by(Student.name).all()
        data = student_analysis(student, students_list)
        return render_template("analysis.html", students=students_list, data=data, selected=student)

    @app.route("/api/dashboard")
    @role_required("teacher")
    def api_dashboard():
        return jsonify(dashboard_data())

    # -------------------------------------------------------------- evaluation
    @app.route("/evaluation")
    @role_required("teacher")
    def evaluation():
        return render_template("evaluation.html", m=predictor.get_metrics())

    @app.route("/api/evaluation")
    @role_required("teacher")
    def api_evaluation():
        return jsonify(predictor.get_metrics())

    @app.route("/evaluation/retrain", methods=["POST"])
    @role_required("teacher")
    def retrain():
        from ml import train as trainer
        trainer.train(verbose=False)
        predictor.reload_bundle()
        flash("Model retrained on the synthetic dataset and reloaded.", "success")
        return redirect(url_for("evaluation"))

    # ---------------------------------------------------------- import / export
    @app.route("/data")
    @role_required("teacher")
    def data_page():
        return render_template("data.html", result=None)

    def csv_response(rows, header, filename):
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(header)
        w.writerows(rows)
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={filename}"})

    @app.route("/data/template.csv")
    @role_required("teacher")
    def csv_template():
        example = ["Asha Patil", "CS2026001", "Computer Science", 5, 82, 38, 16, 74, 3.5, 40]
        return csv_response([example], CSV_COLUMNS, "student_template.csv")

    @app.route("/data/students.csv")
    @role_required("teacher")
    def csv_students():
        rows = [[getattr(s, c) for c in CSV_COLUMNS] + [s.predicted_label or ""]
                for s in Student.query.order_by(Student.roll_number)]
        return csv_response(rows, CSV_COLUMNS + ["predicted_performance"], "students.csv")

    @app.route("/data/predictions.csv")
    @role_required("teacher")
    def csv_predictions():
        header = ["created_at", "roll_number", "student_name", "predicted_label", "model_name", "source"] \
                 + config.FEATURES + ["prob_Low", "prob_Medium", "prob_High"]
        rows = []
        for p in Prediction.query.order_by(Prediction.created_at.desc()):
            probs = p.probabilities or {}
            rows.append([p.created_at.strftime("%Y-%m-%d %H:%M:%S"), p.student.roll_number, p.student.name,
                         p.predicted_label, p.model_name, p.source]
                        + [getattr(p, f) for f in config.FEATURES]
                        + [probs.get(c, "") for c in config.CLASSES])
        return csv_response(rows, header, "predictions.csv")

    @app.route("/data/import", methods=["POST"])
    @role_required("teacher")
    def csv_import():
        file = request.files.get("file")
        if not file or not file.filename:
            flash("Choose a CSV file first.", "danger")
            return redirect(url_for("data_page"))
        if not file.filename.lower().endswith(".csv"):
            flash("Only .csv files are accepted.", "danger")
            return redirect(url_for("data_page"))

        result = {"imported": 0, "skipped": 0, "errors": [], "fatal": None}
        try:
            df = pd.read_csv(file, dtype=str, keep_default_na=False)
        except Exception as exc:  # malformed / empty file
            result["fatal"] = f"Could not read the file as CSV: {exc}"
            return render_template("data.html", result=result)

        df.columns = [c.strip().lower() for c in df.columns]
        missing = [c for c in CSV_COLUMNS if c not in df.columns]
        if missing:
            result["fatal"] = "Missing required column(s): " + ", ".join(missing) + \
                              ". Download the template to see the expected format."
            return render_template("data.html", result=result)
        if df.empty:
            result["fatal"] = "The file contains no data rows."
            return render_template("data.html", result=result)

        existing = {r[0] for r in db.session.query(Student.roll_number).all()}
        seen = set()
        for idx, row in df.iterrows():
            line = idx + 2  # header is line 1
            clean, errors = validate_student(row.to_dict())
            roll = clean.get("roll_number")
            if roll in existing:
                errors["roll_number"] = "Roll number already exists in the database."
            elif roll in seen:
                errors["roll_number"] = "Duplicate roll number inside this file."
            if errors:
                result["skipped"] += 1
                result["errors"].append({"line": line, "roll": row.get("roll_number", ""),
                                         "messages": list(errors.values())})
                continue
            seen.add(roll)
            student = Student(**clean)
            db.session.add(student)
            db.session.flush()
            record_prediction(student, source="import")
            result["imported"] += 1
        db.session.commit()
        return render_template("data.html", result=result)

    @app.errorhandler(404)
    def not_found(e):
        return render_template("error.html", code=404, message="Page not found."), 404

    @app.errorhandler(413)
    def too_large(e):
        return render_template("error.html", code=413, message="The uploaded file is too large (max 5 MB)."), 413

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
