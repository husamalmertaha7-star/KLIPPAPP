import os
import secrets
import threading
import time
import traceback
import uuid

from flask import Flask, request, jsonify, send_file, send_from_directory, session

import db
from auth import auth_bp, current_user, login_required
from billing import billing_bp
from pipeline import run_pipeline
from transcription import openai_enabled

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(os.path.dirname(BASE_DIR), "jobs")
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")
os.makedirs(JOBS_DIR, exist_ok=True)

ALLOWED_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1 GB
DEFAULT_CLIPS_REQUESTED = 6

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
# SECRET_KEY signs the session cookie. Set a real one via env var in production —
# a random one is generated per process start otherwise (fine for local testing,
# but it means everyone's session is invalidated on restart).
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)

app.register_blueprint(auth_bp)
app.register_blueprint(billing_bp)

db.init_db()

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _set_job(job_id, **kwargs):
    with JOBS_LOCK:
        JOBS[job_id].update(kwargs)


def _process(job_id, input_path, n_clips, resolution_tier, reserved_clips, user_id):
    job_dir = os.path.join(JOBS_DIR, job_id)

    def progress_cb(stage, pct):
        _set_job(job_id, stage=stage, progress=pct)

    try:
        _set_job(job_id, status="processing", stage="Starting", progress=1)
        manifest = run_pipeline(job_dir, input_path, progress_cb,
                                 n_clips=n_clips, resolution_tier=resolution_tier)
        produced = len(manifest.get("clips", []))
        if produced < reserved_clips:
            db.refund_quota(user_id, reserved_clips - produced)
        _set_job(job_id, status="done", progress=100, stage="Done", manifest=manifest)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        db.refund_quota(user_id, reserved_clips)
        _set_job(job_id, status="error", error=str(exc))


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/<path:filename>")
def frontend_assets(filename):
    full = os.path.join(FRONTEND_DIR, filename)
    if os.path.isfile(full):
        return send_from_directory(FRONTEND_DIR, filename)
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/api/config")
def api_config():
    return jsonify({
        "ai_mode_available": openai_enabled(),
        "stripe_configured": bool(os.environ.get("STRIPE_PRICE_CREATOR")),
    })


@app.post("/api/jobs")
@login_required
def create_job():
    user = current_user()

    if "video" not in request.files:
        return jsonify({"error": "No 'video' file in request"}), 400
    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXT)}"}), 400

    allowed_clips, reason = db.check_and_reserve_quota(user["id"], DEFAULT_CLIPS_REQUESTED)
    if allowed_clips <= 0:
        return jsonify({"error": reason or "Plan limit reached"}), 402

    resolution_tier = "1080p" if user["plan"] == "free" else "4k"

    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    input_path = os.path.join(job_dir, f"input{ext}")
    file.save(input_path)

    with JOBS_LOCK:
        JOBS[job_id] = {"status": "queued", "stage": "Queued", "progress": 0,
                         "created": time.time(), "user_id": user["id"]}

    thread = threading.Thread(target=_process, args=(job_id, input_path, allowed_clips,
                                                       resolution_tier, allowed_clips, user["id"]),
                               daemon=True)
    thread.start()

    return jsonify({"job_id": job_id, "clips_allowed": allowed_clips, "resolution_tier": resolution_tier})


def _owned_job_or_404(job_id):
    user = current_user()
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job or not user or job.get("user_id") != user["id"]:
        return None
    return job


@app.get("/api/jobs/<job_id>")
@login_required
def job_status(job_id):
    job = _owned_job_or_404(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    return jsonify(job)


@app.get("/api/jobs/<job_id>/download/<path:rel_path>")
@login_required
def download_clip(job_id, rel_path):
    if not _owned_job_or_404(job_id):
        return jsonify({"error": "unknown job"}), 404
    job_dir = os.path.join(JOBS_DIR, job_id)
    full = os.path.normpath(os.path.join(job_dir, rel_path))
    if not full.startswith(os.path.normpath(job_dir)):
        return jsonify({"error": "invalid path"}), 400
    if not os.path.isfile(full):
        return jsonify({"error": "not found"}), 404
    return send_file(full)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
