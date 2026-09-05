"""
Turns AudioFeatures (see audio_analysis.py) into a ranked shortlist of
non-overlapping "clip" candidates — the offline stand-in for Klipp's
"AI ranks every moment" step.

Method (fully explainable, no black box):
  1. Combine energy / zcr-variance / spectral-flux into one per-frame
     "hook" signal.
  2. Slide a window (default 25s, step 2.5s) across the video and sum the
     hook signal inside it, penalizing windows that are mostly silence.
  3. Greedily pick the highest-scoring window, suppress overlapping
     windows (classic 1D non-max suppression), repeat until N clips
     are chosen or the video is exhausted.
  4. Snap each window's start/end to the nearest nearby silence gap so
     cuts land in a natural pause instead of mid-word.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from audio_analysis import AudioFeatures


@dataclass
class ClipCandidate:
    start: float
    end: float
    score: float          # 0-100
    label: str
    reasons: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


def _rolling_variance(x: np.ndarray, win: int) -> np.ndarray:
    if win < 2:
        return np.zeros_like(x)
    csum = np.cumsum(np.insert(x, 0, 0))
    csum2 = np.cumsum(np.insert(x ** 2, 0, 0))
    out = np.zeros_like(x)
    half = win // 2
    n = len(x)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        cnt = hi - lo
        s = csum[hi] - csum[lo]
        s2 = csum2[hi] - csum2[lo]
        mean = s / cnt
        out[i] = max(0.0, s2 / cnt - mean ** 2)
    return out


def compute_hook_signal(feat: AudioFeatures) -> np.ndarray:
    # pace variance (from zcr) rewards lively, dynamic delivery over a flat drone
    pace_var = _rolling_variance(feat.zcr, win=max(3, int(1.0 / feat.frame_hop_s)))
    pace_var = pace_var / (pace_var.max() + 1e-9)

    hook = 0.45 * feat.energy + 0.30 * feat.flux + 0.25 * pace_var

    # zero out frames that fall inside a detected silence
    for s, e in feat.silence:
        mask = (feat.times >= s) & (feat.times <= e)
        hook[mask] *= 0.05
    return hook


def _silence_ratio(feat: AudioFeatures, start: float, end: float) -> float:
    if end <= start:
        return 1.0
    covered = 0.0
    for s, e in feat.silence:
        lo, hi = max(s, start), min(e, end)
        if hi > lo:
            covered += hi - lo
    return covered / (end - start)


def _nearest_silence_edge(feat: AudioFeatures, t: float, tolerance: float = 2.5):
    best = None
    best_d = tolerance
    for s, e in feat.silence:
        mid = (s + e) / 2
        d = abs(mid - t)
        if d < best_d:
            best_d = d
            best = mid
    return best if best is not None else t


LABELS = ["Best moment", "Strong hook", "High energy", "Clip", "Worth clipping"]


def select_clips(
    feat: AudioFeatures,
    n_clips: int = 6,
    win_s: float = 25.0,
    step_s: float = 2.5,
    min_gap_s: float = 8.0,
    min_clip_s: float = 12.0,
    max_clip_s: float = 60.0,
) -> list[ClipCandidate]:
    hook = compute_hook_signal(feat)
    dt = feat.frame_hop_s
    win_frames = max(1, int(win_s / dt))
    step_frames = max(1, int(step_s / dt))

    csum = np.cumsum(np.insert(hook, 0, 0))
    n = len(hook)
    starts = list(range(0, max(1, n - win_frames), step_frames))
    window_scores = []
    for i in starts:
        j = min(n, i + win_frames)
        total = csum[j] - csum[i]
        window_scores.append(total / max(1, j - i))  # mean hook value in window
    window_scores = np.array(window_scores) if starts else np.array([0.0])

    order = np.argsort(-window_scores)
    chosen: list[ClipCandidate] = []
    taken_ranges: list[tuple[float, float]] = []

    max_score = float(window_scores.max()) if len(window_scores) else 1.0
    max_score = max(max_score, 1e-6)

    for idx in order:
        if len(chosen) >= n_clips:
            break
        i = starts[idx]
        j = min(n, i + win_frames)
        start_t = feat.times[i] if i < n else 0.0
        end_t = feat.times[j - 1] if j - 1 < n else feat.duration

        # skip windows that are mostly silence (e.g. trailing dead air)
        if _silence_ratio(feat, start_t, end_t) > 0.6:
            continue

        # enforce a minimum gap from already-chosen clips (non-max suppression)
        overlaps = any(not (end_t < ts - min_gap_s or start_t > te + min_gap_s)
                        for ts, te in taken_ranges)
        if overlaps:
            continue

        snapped_start = _nearest_silence_edge(feat, start_t)
        snapped_end = _nearest_silence_edge(feat, end_t)
        snapped_start = max(0.0, min(snapped_start, feat.duration - min_clip_s))
        snapped_end = min(feat.duration, max(snapped_end, snapped_start + min_clip_s))
        snapped_end = min(snapped_end, snapped_start + max_clip_s)

        score = round(100 * window_scores[idx] / max_score, 1)
        label = LABELS[len(chosen)] if len(chosen) < len(LABELS) else "Clip"
        chosen.append(ClipCandidate(
            start=round(snapped_start, 2),
            end=round(snapped_end, 2),
            score=score,
            label=label,
            reasons=["high energy" if feat.energy[i:j].mean() > 0.5 else None,
                     "dynamic pacing" if feat.zcr[i:j].std() > feat.zcr.std() else None,
                     ],
        ))
        taken_ranges.append((snapped_start, snapped_end))

    chosen.sort(key=lambda c: c.start)
    return chosen
