"""
Pluggable transcription + LLM ranking backend.

- Heuristic mode (default, works fully offline): no transcript is produced;
  highlight_scoring.py's audio-signal analysis is used instead, and burned-in
  captions are simply skipped (nothing dishonest is drawn on screen).

- OpenAI mode (opt-in via OPENAI_API_KEY env var): uses the real Whisper
  transcription endpoint for word-level timestamps (real burned-in captions)
  and, optionally, a chat-completion call to semantically rank hook moments
  from the transcript instead of relying only on audio signal.

  IMPORTANT: this mode is fully implemented and should work against the real
  OpenAI API, but it could NOT be exercised end-to-end in the build sandbox
  used to create this project — that sandbox has no network access at all
  (no pypi, no api.openai.com, nothing). Test it against a real key once you
  deploy this somewhere with normal internet access.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass
class Word:
    start: float
    end: float
    text: str


def openai_enabled() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def transcribe_openai(audio_path: str, api_key: str | None = None, timeout: int = 180):
    """Calls OpenAI's /v1/audio/transcriptions with word-level timestamps.

    Returns a list[Word] on success, or None if no API key is configured.
    Raises RuntimeError on API failure (caller should catch and fall back).
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    import requests  # local import: only required when this path is used

    url = "https://api.openai.com/v1/audio/transcriptions"
    with open(audio_path, "rb") as f:
        files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
        data = {
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=timeout)

    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI transcription failed: {resp.status_code} {resp.text[:500]}")

    payload = resp.json()
    words = payload.get("words") or []
    return [Word(start=float(w["start"]), end=float(w["end"]), text=str(w["word"]))
            for w in words]


def anthropic_enabled() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def rank_hooks_with_claude(words: list[Word], api_key: str | None = None,
                            n_clips: int = 6, timeout: int = 60):
    """Alternative to rank_hooks_with_gpt: asks Claude instead of ChatGPT to pick
    the best moments from the transcript. Swapped in automatically when
    ANTHROPIC_API_KEY is set (see pipeline.py). Same network caveat as the
    OpenAI calls above — implemented, not sandbox-testable.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not words:
        return None

    import requests

    transcript_lines = "\n".join(f"[{w.start:.1f}] {w.text}" for w in words)
    prompt = (
        "You are selecting short-form highlight clips from a long-form video "
        "transcript with word start times in seconds. Respond with ONLY a JSON "
        f"array (no other text, no markdown fences) of up to {n_clips} objects: "
        '{"start": float, "end": float, "label": "short hook label", '
        '"reason": "why this moment works as a clip"}. '
        "Clips should be 15-60 seconds, non-overlapping, ordered by start time, "
        "and chosen for hooks, punchlines, or payoffs. "
        "Transcript:\n" + transcript_lines[:12000]
    )
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Claude ranking failed: {resp.status_code} {resp.text[:500]}")
    content = resp.json()["content"][0]["text"].strip()
    # Claude sometimes wraps JSON in ```json fences despite instructions — strip them.
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
    parsed = json.loads(content)
    return parsed if isinstance(parsed, list) else parsed.get("clips", [])


def rank_hooks_with_gpt(words: list[Word], api_key: str | None = None,
                         n_clips: int = 6, timeout: int = 60):
    """Optional: ask a GPT model to pick the best moments from the transcript.

    Returns a list of {start, end, label, reason} dicts, or None if unavailable.
    Same network caveat as transcribe_openai — implemented, not sandbox-testable.
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key or not words:
        return None

    import requests

    transcript_lines = "\n".join(f"[{w.start:.1f}] {w.text}" for w in words)
    prompt = (
        "You are selecting short-form highlight clips from a long-form video "
        "transcript with word start times in seconds. Return a JSON array of "
        f"up to {n_clips} objects: "
        '{"start": float, "end": float, "label": "short hook label", '
        '"reason": "why this moment works as a clip"}. '
        "Clips should be 15-60 seconds, non-overlapping, ordered by start time, "
        "and chosen for hooks, punchlines, or payoffs. "
        "Transcript:\n" + transcript_lines[:12000]
    )
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI ranking failed: {resp.status_code} {resp.text[:500]}")
    content = resp.json()["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    return parsed if isinstance(parsed, list) else parsed.get("clips", [])
