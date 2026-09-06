"""Orchestrates the full Klipp pipeline for one uploaded video:

  1. extract audio + analyze signal (offline, always runs)
  2. transcribe (OpenAI, only if OPENAI_API_KEY set) -> word timestamps
  3. select highlight clips (Claude- or GPT-ranked transcript if available,
     else audio-signal heuristic)
  4. per clip: trim internal silence, detect speaker face, render 3 aspect
     ratios with burned-in captions (if available) and a score badge
  5. write manifest.json describing everything that was produced
"""
from __future__ import annotations

import json
import os
import traceback

from audio_analysis import analyze
from highlight_scoring import ClipCandidate, select_clips
from reframe import detect_face_center_x
from render import aspect_sizes, build_keep_segments, trim_to_file, render_aspect, make_thumbnail
from captions import build_ass
from transcription import (
    openai_enabled, transcribe_openai, rank_hooks_with_gpt,
    anthropic_enabled, rank_hooks_with_claude, Word,
)


def remap_words_to_trimmed(segments, words: list[Word]) -> list[Word]:
    """Maps absolute-time words onto the new timeline produced by concatenating
    `segments` (list of (abs_start, abs_end), in order, silence already removed)."""
    offsets = []
    acc = 0.0
    for s, e in segments:
        offsets.append(acc)
        acc += (e - s)

    out = []
    for w in words:
        for (s, e), off in zip(segments, offsets):
            lo, hi = max(s, w.start), min(e, w.end)
            if hi > lo:
                out.append(Word(start=off + (lo - s), end=off + (hi - s), text=w.text))
                break
    return out


def run_pipeline(job_dir: str, input_path: str, progress_cb, n_clips: int = 6,
                  resolution_tier: str = "1080p"):
    """progress_cb(stage: str, pct: float) is called throughout. Returns the
    manifest dict on success. Raises on unrecoverable failure."""
    wav_path = os.path.join(job_dir, "audio.wav")

    progress_cb("Extracting audio", 4)
    feat = analyze(input_path, wav_path)

    mode = "heuristic"
    words: list[Word] | None = None
    warnings: list[str] = []

    if openai_enabled():
        progress_cb("Transcribing with OpenAI Whisper", 12)
        try:
            words = transcribe_openai(wav_path)
            if words:
                mode = "ai"
        except Exception as exc:  # noqa: BLE001 - best-effort fallback
            warnings.append(f"OpenAI transcription failed, falling back to offline mode: {exc}")
            words = None

    progress_cb("Scoring hook moments", 20)
    clips: list[ClipCandidate] = []
    if words and mode == "ai":
        ranked = None
        try:
            # Prefer Claude when ANTHROPIC_API_KEY is set, else fall back to GPT.
            if anthropic_enabled():
                ranked = rank_hooks_with_claude(words, n_clips=n_clips)
            elif openai_enabled():
                ranked = rank_hooks_with_gpt(words, n_clips=n_clips)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"AI ranking failed, using audio heuristic instead: {exc}")
            ranked = None
        if ranked:
            for i, c in enumerate(ranked):
                s, e = float(c["start"]), float(c["end"])
                s = max(0.0, min(s, feat.duration))
                e = max(s + 5.0, min(e, feat.duration))
                clips.append(ClipCandidate(start=round(s, 2), end=round(e, 2), score=95.0 - i,
                                            label=c.get("label", "Clip"),
                                            reasons=[c.get("reason", "")]))
        else:
            clips = select_clips(feat, n_clips=n_clips)
    else:
        clips = select_clips(feat, n_clips=n_clips)

    if not clips:
        raise RuntimeError("No clips could be selected from this video (too short or too quiet).")

    manifest_clips = []
    total = len(clips)
    for idx, clip in enumerate(clips):
        base_pct = 25 + int(65 * idx / total)
        progress_cb(f"Rendering clip {idx + 1} of {total}", base_pct)

        clip_dir = os.path.join(job_dir, f"clip_{idx}")
        os.makedirs(clip_dir, exist_ok=True)

        segments = build_keep_segments(clip.start, clip.end, feat.silence)
        trimmed_path = os.path.join(clip_dir, "trimmed.mp4")
        trim_to_file(input_path, segments, trimmed_path)

        clip_words = None
        ass_path = None
        if words:
            clip_words = remap_words_to_trimmed(segments, [
                w for w in words if w.end > clip.start and w.start < clip.end
            ])

        center_x, src_w, src_h = detect_face_center_x(input_path, clip.start, clip.end)

        badge_text = f"{int(clip.score)}%  {clip.label}"
        files = {}
        sizes = aspect_sizes(resolution_tier)
        for aspect in sizes:
            out_path = os.path.join(clip_dir, f"{aspect}.mp4")
            ass_for_aspect = None
            if clip_words:
                target_w, target_h = sizes[aspect]
                ass_for_aspect = os.path.join(clip_dir, f"captions_{aspect}.ass")
                build_ass(clip_words, ass_for_aspect, target_w, target_h)
            render_aspect(trimmed_path, out_path, aspect,
                          center_x_norm=center_x, ass_path=ass_for_aspect,
                          badge_text=badge_text, tier=resolution_tier)
            files[aspect] = os.path.relpath(out_path, job_dir)

        thumb_path = os.path.join(clip_dir, "thumb.jpg")
        try:
            make_thumbnail(trimmed_path, thumb_path)
        except Exception:
            thumb_path = None

        manifest_clips.append({
            "index": idx,
            "label": clip.label,
            "score": clip.score,
            "start": clip.start,
            "end": clip.end,
            "duration": round(clip.end - clip.start, 1),
            "has_captions": clip_words is not None and len(clip_words) > 0,
            "face_detected": center_x is not None,
            "reasons": [r for r in clip.reasons if r],
            "files": files,
            "thumbnail": os.path.relpath(thumb_path, job_dir) if thumb_path else None,
        })

    progress_cb("Done", 100)
    manifest = {
        "mode": mode,
        "resolution_tier": resolution_tier,
        "warnings": warnings,
        "source_duration": round(feat.duration, 1),
        "clips": manifest_clips,
    }
    with open(os.path.join(job_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
