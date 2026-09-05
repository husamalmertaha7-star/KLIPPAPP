"""
Fully offline audio-signal analysis used to find "hook" moments in a video
without any transcript. All of this runs locally via ffmpeg + numpy/scipy —
no network, no model downloads.

Signals computed per short frame (default 50ms hop):
  - RMS energy            -> loudness / energy
  - Zero-crossing rate     -> proxy for speech pace / sibilance / noisiness
  - Spectral flux          -> proxy for emphasis, bursts, laughter, applause
  - Silence intervals      -> via ffmpeg's silencedetect filter

These are combined by highlight_scoring.py into a single "hook score".
"""
from __future__ import annotations

import re
import subprocess
import wave
from dataclasses import dataclass

import numpy as np


def extract_audio_wav(src_path: str, wav_path: str, sr: int = 16000) -> None:
    """Extract mono PCM16 WAV audio from any video/audio file via ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-vn", "-ac", "1", "-ar", str(sr), "-sample_fmt", "s16",
        wav_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def read_wav_mono(wav_path: str):
    with wave.open(wav_path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return sr, data


def probe_duration(path: str) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", path]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
    return float(out)


_SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9.]+)")


def detect_silence(src_path: str, noise_db: float = -30.0, min_dur: float = 0.35):
    """Returns a list of (start, end) silence intervals in seconds, via ffmpeg silencedetect."""
    cmd = [
        "ffmpeg", "-i", src_path,
        "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}",
        "-f", "null", "-",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    log = p.stderr
    starts = [float(x) for x in _SILENCE_START_RE.findall(log)]
    ends = [float(x) for x in _SILENCE_END_RE.findall(log)]
    n = min(len(starts), len(ends))
    return list(zip(starts[:n], ends[:n]))


@dataclass
class AudioFeatures:
    sr: int
    frame_hop_s: float
    times: np.ndarray      # center time of each frame, seconds
    energy: np.ndarray     # RMS energy, normalized 0-1
    zcr: np.ndarray        # zero crossing rate, normalized 0-1
    flux: np.ndarray       # spectral flux, normalized 0-1
    silence: list          # list of (start,end) seconds
    duration: float


def _minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def analyze(src_path: str, wav_path: str, frame_ms: float = 50.0) -> AudioFeatures:
    extract_audio_wav(src_path, wav_path)
    sr, data = read_wav_mono(wav_path)
    duration = len(data) / sr

    frame_len = max(1, int(sr * frame_ms / 1000))
    hop_len = frame_len  # non-overlapping frames for energy/zcr
    n_frames = max(1, len(data) // hop_len)

    energy = np.zeros(n_frames)
    zcr = np.zeros(n_frames)
    for i in range(n_frames):
        seg = data[i * hop_len:(i + 1) * hop_len]
        if seg.size == 0:
            continue
        energy[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12)
        signs = np.sign(seg)
        signs[signs == 0] = 1
        zcr[i] = np.mean(np.abs(np.diff(signs))) / 2.0

    # Spectral flux on a slightly larger window for a smoother "burst" signal.
    spec_frame = max(frame_len * 2, 512)
    window = np.hanning(spec_frame)
    flux = np.zeros(n_frames)
    prev_mag = None
    for i in range(n_frames):
        start = i * hop_len
        seg = data[start:start + spec_frame]
        if seg.size < spec_frame:
            seg = np.pad(seg, (0, spec_frame - seg.size))
        spec = np.abs(np.fft.rfft(seg * window))
        if prev_mag is not None:
            diff = spec - prev_mag
            diff[diff < 0] = 0
            flux[i] = np.sum(diff)
        prev_mag = spec

    times = (np.arange(n_frames) + 0.5) * (hop_len / sr)
    silence = detect_silence(src_path)

    return AudioFeatures(
        sr=sr,
        frame_hop_s=hop_len / sr,
        times=times,
        energy=_minmax(energy),
        zcr=_minmax(zcr),
        flux=_minmax(flux),
        silence=silence,
        duration=duration,
    )
