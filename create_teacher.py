"""Create a teacher login account. Run: python create_teacher.py"""
from getpass import getpass

from app import app
from models import User, db


def main():
    print("Create a teacher account for EduPredict")
    name = input("Teacher name: ").strip()
    email = input("Teacher email: ").strip().lower()
    password = getpass("Password (minimum 8 characters): ")
    confirm = getpass("Confirm password: ")

    if not name or not email:
        print("Name and email are required.")
        return
    if len(password) < 8:
        print("Password must contain at least 8 characters.")
        return
    if password != confirm:
        print("Passwords do not match.")
        return

    with app.app_context():
        if User.query.filter_by(email=email).first():
            print("An account with this email already exists.")
            return
        teacher = User(name=name, email=email, role="teacher")
        teacher.set_password(password)
        db.session.add(teacher)
        db.session.commit()
        print(f"Teacher account created successfully for {email}.")


if __name__ == "__main__":
    main()
