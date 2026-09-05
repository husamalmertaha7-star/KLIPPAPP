"""Stripe billing via raw REST calls (the `stripe` PyPI package could not be
installed in the build sandbox — no network there — so this talks to
https://api.stripe.com directly with `requests`, using the same API Stripe's
official SDKs wrap). Same caveat as transcription.py: this is implemented to
the documented Stripe API shape but could NOT be exercised against a live
Stripe account in this sandbox (api.stripe.com was unreachable there too).
Test it for real once deployed with an actual Stripe account + keys.

Studio plan is intentionally NOT a self-serve Stripe checkout — like the
"Talk to us" button on the original landing page, it just records a contact
request (contact_requests table) for a human to follow up on.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time

import requests
from flask import Blueprint, jsonify, request

import db
from auth import current_user, login_required

billing_bp = Blueprint("billing", __name__, url_prefix="/api/billing")

STRIPE_API = "https://api.stripe.com/v1"


def _secret_key() -> str:
    key = os.environ.get("STRIPE_SECRET_KEY")
    if not key:
        raise RuntimeError("STRIPE_SECRET_KEY is not set on the server")
    return key


def _base_url() -> str:
    return os.environ.get("APP_BASE_URL", "http://localhost:8000").rstrip("/")


def _flatten(data: dict, parent_key: str = "") -> dict:
    """Stripe's form-encoded API expects nested structures as bracket keys,
    e.g. line_items[0][price]=price_123. This flattens a Python dict/list
    into that form."""
    items: dict = {}
    for k, v in data.items():
        key = f"{parent_key}[{k}]" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(_flatten(v, key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                ik = f"{key}[{i}]"
                if isinstance(item, dict):
                    items.update(_flatten(item, ik))
                else:
                    items[ik] = item
        else:
            items[key] = v
    return items


def _stripe_post(path: str, data: dict) -> dict:
    resp = requests.post(f"{STRIPE_API}/{path}", data=_flatten(data),
                          auth=(_secret_key(), ""), timeout=30)
    if resp.status_code >= 400:
        raise RuntimeError(f"Stripe API error ({path}): {resp.status_code} {resp.text[:500]}")
    return resp.json()


@billing_bp.post("/checkout")
@login_required
def create_checkout():
    """Starts a Stripe Checkout subscription session for the Creator plan.
    Studio is sales-assisted (see /contact-studio below), not self-serve."""
    user = current_user()
    data = request.get_json(silent=True) or {}
    plan = data.get("plan")

    if plan != "creator":
        return jsonify({"error": "Only the Creator plan is available for self-serve checkout. "
                                  "For Studio, use /api/billing/contact-studio."}), 400

    price_id = os.environ.get("STRIPE_PRICE_CREATOR")
    if not price_id:
        return jsonify({"error": "Server is not configured with STRIPE_PRICE_CREATOR yet"}), 503

    params = {
        "mode": "subscription",
        "line_items": [{"price": price_id, "quantity": 1}],
        "success_url": f"{_base_url()}/account.html?checkout=success",
        "cancel_url": f"{_base_url()}/account.html?checkout=cancelled",
        "client_reference_id": str(user["id"]),
        "subscription_data": {"metadata": {"user_id": str(user["id"])}},
    }
    if user.get("stripe_customer_id"):
        params["customer"] = user["stripe_customer_id"]
    else:
        params["customer_email"] = user["email"]

    try:
        session_obj = _stripe_post("checkout/sessions", params)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 502
    return jsonify({"checkout_url": session_obj["url"]})


@billing_bp.post("/portal")
@login_required
def billing_portal():
    """Stripe's self-service billing portal, for cancelling/updating payment method."""
    user = current_user()
    if not user.get("stripe_customer_id"):
        return jsonify({"error": "No billing account yet — subscribe to Creator first"}), 400
    try:
        session_obj = _stripe_post("billing_portal/sessions", {
            "customer": user["stripe_customer_id"],
            "return_url": f"{_base_url()}/account.html",
        })
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 502
    return jsonify({"portal_url": session_obj["url"]})


@billing_bp.post("/contact-studio")
@login_required
def contact_studio():
    user = current_user()
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()[:2000]
    db.create_contact_request(user["id"], user["email"], "studio", message)
    return jsonify({"ok": True, "note": "Thanks — we'll follow up by email."})


def _verify_signature(payload: bytes, sig_header: str, secret: str, tolerance: int = 300):
    parts = dict(p.split("=", 1) for p in sig_header.split(",") if "=" in p)
    timestamp, v1 = parts.get("t"), parts.get("v1")
    if not timestamp or not v1:
        raise ValueError("malformed Stripe-Signature header")
    if abs(time.time() - int(timestamp)) > tolerance:
        raise ValueError("timestamp outside tolerance (possible replay)")
    signed_payload = f"{timestamp}.".encode() + payload
    expected = hmac.new(secret.encode(), signed_payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, v1):
        raise ValueError("signature mismatch")


@billing_bp.post("/webhook")
def webhook():
    """Stripe webhook receiver. Verifies the Stripe-Signature header manually
    (HMAC-SHA256 over "{timestamp}.{raw_body}") since the official SDK isn't
    installed here — see module docstring."""
    secret = os.environ.get("STRIPE_WEBHOOK_SECRET")
    if not secret:
        return jsonify({"error": "STRIPE_WEBHOOK_SECRET not configured"}), 500

    payload = request.get_data()  # raw bytes — required for signature verification
    try:
        _verify_signature(payload, request.headers.get("Stripe-Signature", ""), secret)
    except ValueError as exc:
        return jsonify({"error": f"invalid signature: {exc}"}), 400

    event = json.loads(payload)
    etype = event.get("type")
    obj = event.get("data", {}).get("object", {})

    if etype == "checkout.session.completed":
        user_id = obj.get("client_reference_id")
        customer_id = obj.get("customer")
        subscription_id = obj.get("subscription")
        if user_id and customer_id:
            db.set_stripe_customer(int(user_id), customer_id)
            db.update_subscription(int(user_id), plan="creator",
                                    subscription_id=subscription_id, status="active")

    elif etype in ("customer.subscription.updated", "customer.subscription.created"):
        customer_id = obj.get("customer")
        status = obj.get("status")
        user = db.get_user_by_stripe_customer(customer_id)
        if user:
            plan = "creator" if status in ("active", "trialing") else "free"
            db.update_subscription(user["id"], plan=plan, subscription_id=obj.get("id"), status=status)

    elif etype == "customer.subscription.deleted":
        customer_id = obj.get("customer")
        user = db.get_user_by_stripe_customer(customer_id)
        if user:
            db.update_subscription(user["id"], plan="free", subscription_id=None, status="canceled")

    return jsonify({"received": True})
