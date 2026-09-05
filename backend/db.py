"""Tiny SQLite-backed user store. No external DB needed — stdlib sqlite3 only.

Schema:
  users(id, email, password_hash, plan, created_at,
        stripe_customer_id, stripe_subscription_id, subscription_status,
        period_start, clips_used_in_period)

Plans: "free" | "creator" | "studio"
  - free:    3 clips per rolling 30-day period, 1080p max export
  - creator: unlimited clips, up to 4K export (paid via Stripe)
  - studio:  unlimited + priority (currently sales-assisted, not self-serve)
"""
from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "jobs", "klipp.db")
DB_PATH = os.path.normpath(DB_PATH)

FREE_CLIP_LIMIT = 3
FREE_PERIOD_SECONDS = 30 * 24 * 3600


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_conn():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                plan TEXT NOT NULL DEFAULT 'free',
                created_at REAL NOT NULL,
                stripe_customer_id TEXT,
                stripe_subscription_id TEXT,
                subscription_status TEXT,
                period_start REAL NOT NULL,
                clips_used_in_period INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS contact_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                email TEXT NOT NULL,
                plan TEXT NOT NULL,
                message TEXT,
                created_at REAL NOT NULL
            )
        """)


def create_contact_request(user_id: int, email: str, plan: str, message: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO contact_requests (user_id, email, plan, message, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, email, plan, message, time.time()),
        )


def create_user(email: str, password_hash: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO users (email, password_hash, plan, created_at, period_start, "
            "clips_used_in_period) VALUES (?, ?, 'free', ?, ?, 0)",
            (email.lower().strip(), password_hash, time.time(), time.time()),
        )
        return cur.lastrowid


def get_user_by_email(email: str):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def get_user_by_stripe_customer(customer_id: str):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE stripe_customer_id = ?", (customer_id,)).fetchone()
        return dict(row) if row else None


def set_stripe_customer(user_id: int, customer_id: str):
    with get_conn() as conn:
        conn.execute("UPDATE users SET stripe_customer_id = ? WHERE id = ?", (customer_id, user_id))


def update_subscription(user_id: int, *, plan: str, subscription_id: str | None,
                         status: str | None):
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET plan = ?, stripe_subscription_id = ?, subscription_status = ? "
            "WHERE id = ?",
            (plan, subscription_id, status, user_id),
        )


def _maybe_reset_period(conn, user: dict):
    if time.time() - user["period_start"] > FREE_PERIOD_SECONDS:
        conn.execute(
            "UPDATE users SET period_start = ?, clips_used_in_period = 0 WHERE id = ?",
            (time.time(), user["id"]),
        )
        user["period_start"] = time.time()
        user["clips_used_in_period"] = 0


def check_and_reserve_quota(user_id: int, clips_requested: int = 6) -> tuple[int, str]:
    """For free-plan users, reserves up to the remaining rolling-period quota
    and returns how many clips are actually allowed (may be less than
    clips_requested, or 0 if the quota is exhausted). Paid plans always get
    the full amount requested. Returns (allowed_count, reason_if_zero)."""
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row:
            return 0, "user not found"
        user = dict(row)
        if user["plan"] != "free":
            return clips_requested, ""

        _maybe_reset_period(conn, user)
        remaining = FREE_CLIP_LIMIT - user["clips_used_in_period"]
        if remaining <= 0:
            return 0, (
                f"Free plan limit reached ({FREE_CLIP_LIMIT} clips per 30 days). "
                "Upgrade to Creator for unlimited clips."
            )
        allowed = min(clips_requested, remaining)
        conn.execute(
            "UPDATE users SET clips_used_in_period = clips_used_in_period + ? WHERE id = ?",
            (allowed, user_id),
        )
        return allowed, ""


def refund_quota(user_id: int, clips: int = 1):
    """Called when a reserved job fails before producing clips, so a failed
    upload doesn't burn the user's free-plan quota."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET clips_used_in_period = MAX(0, clips_used_in_period - ?) WHERE id = ?",
            (clips, user_id),
        )


def usage_summary(user_id: int) -> dict:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        user = dict(row)
        _maybe_reset_period(conn, user)
        if user["plan"] == "free":
            return {
                "plan": "free",
                "clips_used": user["clips_used_in_period"],
                "clips_limit": FREE_CLIP_LIMIT,
                "max_resolution": "1080p",
            }
        return {
            "plan": user["plan"],
            "clips_used": user["clips_used_in_period"],
            "clips_limit": None,
            "max_resolution": "4K",
        }
