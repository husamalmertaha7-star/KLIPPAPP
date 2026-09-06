# Klipp — deployable image. Built from python:3.11-slim + ffmpeg.
#
# NOTE: this Dockerfile could not be build-tested in the sandbox this project
# was built in (no network access at all there — not even to pull this base
# image, confirmed via `docker pull`). It follows a standard, well-worn
# pattern (slim base + apt ffmpeg + pip install + gunicorn), but treat the
# first real `docker build` / deploy as the actual first test of it.
FROM python:3.11-slim

# ffmpeg is the only system package this app needs — everything else (numpy,
# opencv, scipy) ships as prebuilt Python wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend
COPY frontend ./frontend

# Ephemeral by default — mount a persistent disk here in production and set
# DATA_DIR=/data (see README) or accounts/clips reset on every restart.
RUN mkdir -p /app/jobs
ENV DATA_DIR=/app/jobs

ENV PYTHONUNBUFFERED=1
EXPOSE 10000

WORKDIR /app/backend
# Single worker: job status is currently kept in-process memory (see app.py),
# so multiple gunicorn workers would each have their own job list and break
# status polling. Fine for the free-tier CPU budget this targets; revisit
# (e.g. move job state into the SQLite db) before scaling workers up.
CMD ["sh", "-c", "python -m gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 8 --timeout 300 app:app"]
