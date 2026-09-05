"""Session-based auth: signup/login/logout + a login_required guard.
Passwords are hashed with werkzeug's PBKDF2 implementation (ships with Flask).
No external auth provider needed.
"""
from __future__ import annotations

import re
from functools import wraps

from flask import Blueprint, jsonify, request, session
from werkzeug.security import check_password_hash, generate_password_hash

import db

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "login required"}), 401
        return fn(*args, **kwargs)
    return wrapper


def current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    return db.get_user_by_id(uid)


def _public_user(user: dict) -> dict:
    usage = db.usage_summary(user["id"])
    return {
        "id": user["id"],
        "email": user["email"],
        "plan": user["plan"],
        "subscription_status": user["subscription_status"],
        "usage": usage,
    }


@auth_bp.post("/signup")
def signup():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip()
    password = data.get("password") or ""

    if not EMAIL_RE.match(email):
        return jsonify({"error": "Please enter a valid email address"}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
    if db.get_user_by_email(email):
        return jsonify({"error": "An account with this email already exists"}), 409

    user_id = db.create_user(email, generate_password_hash(password))
    session["user_id"] = user_id
    session.permanent = True
    return jsonify(_public_user(db.get_user_by_id(user_id)))


@auth_bp.post("/login")
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip()
    password = data.get("password") or ""

    user = db.get_user_by_email(email)
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid email or password"}), 401

    session["user_id"] = user["id"]
    session.permanent = True
    return jsonify(_public_user(user))


@auth_bp.post("/logout")
def logout():
    session.clear()
    return jsonify({"ok": True})


@auth_bp.get("/me")
def me():
    user = current_user()
    if not user:
        return jsonify({"error": "not logged in"}), 401
    return jsonify(_public_user(user))
