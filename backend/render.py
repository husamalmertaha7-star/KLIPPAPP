"""ffmpeg rendering: internal silence trimming, face-aware crop, caption burn-in,
score-badge overlay, multi-aspect export. Everything here shells out to the
ffmpeg/ffprobe binaries already installed on the system.
"""
from __future__ import annotations

import json
import subprocess

from reframe import crop_rect_for_aspect

ASPECTS = {
    "9x16": (1080, 1920),
    "1x1": (1080, 1080),
    "16x9": (1920, 1080),
}

# Resolution tiers gated by plan (see db.py / pipeline.py). "4k" quadruples
# pixel count vs 1080p, same aspect ratios.
RESOLUTION_TIERS = {
    "1080p": {
        "9x16": (1080, 1920),
        "1x1": (1080, 1080),
        "16x9": (1920, 1080),
    },
    "4k": {
        "9x16": (2160, 3840),
        "1x1": (2160, 2160),
        "16x9": (3840, 2160),
    },
}


def aspect_sizes(tier: str = "1080p") -> dict:
    return RESOLUTION_TIERS.get(tier, RESOLUTION_TIERS["1080p"])

POPPINS_BOLD = "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf"


def probe_size(path: str):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height", "-of", "json", path]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    st = json.loads(out)["streams"][0]
    return int(st["width"]), int(st["height"])


def build_keep_segments(clip_start: float, clip_end: float, silence_intervals,
                         min_gap: float = 0.6, pad: float = 0.12):
    """Absolute-time (seconds) segments to KEEP inside [clip_start, clip_end],
    i.e. the clip minus any internal silence gaps longer than min_gap."""
    cuts = []
    for s, e in silence_intervals:
        s2, e2 = max(s, clip_start), min(e, clip_end)
        if e2 - s2 >= min_gap:
            cuts.append((s2 + pad, e2 - pad))
    cuts.sort()

    segments = []
    cursor = clip_start
    for s, e in cuts:
        if s > cursor:
            segments.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < clip_end:
        segments.append((cursor, clip_end))

    segments = [(a, b) for a, b in segments if b - a > 0.25]
    return segments or [(clip_start, clip_end)]


def trim_to_file(src_path: str, segments, out_path: str):
    """Cuts+concats the given absolute-time segments out of src_path into out_path."""
    if len(segments) == 1:
        s, e = segments[0]
        cmd = ["ffmpeg", "-y", "-ss", f"{s:.3f}", "-i", src_path, "-t", f"{max(0.1, e - s):.3f}",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-c:a", "aac", out_path]
        subprocess.run(cmd, check=True, capture_output=True)
        return

    filter_parts = []
    concat_v, concat_a = [], []
    for idx, (s, e) in enumerate(segments):
        filter_parts.append(f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{idx}]")
        filter_parts.append(f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{idx}]")
        concat_v.append(f"[v{idx}]")
        concat_a.append(f"[a{idx}]")
    n = len(segments)
    filter_complex = ";".join(filter_parts) + ";" + "".join(concat_v) + "".join(concat_a) + \
        f"concat=n={n}:v=1:a=1[vout][aout]"
    cmd = ["ffmpeg", "-y", "-i", src_path, "-filter_complex", filter_complex,
           "-map", "[vout]", "-map", "[aout]",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-c:a", "aac", out_path]
    subprocess.run(cmd, check=True, capture_output=True)


def _badge_filter(text: str, out_h: int) -> str:
    fontsize = max(22, int(out_h * 0.032))
    safe = text.replace("\\", "").replace(":", "\\:").replace("'", "")
    return (f"drawtext=fontfile={POPPINS_BOLD}:text='{safe}':expansion=none:"
            f"fontsize={fontsize}:fontcolor=white:box=1:boxcolor=black@0.55:"
            f"boxborderw=14:x=28:y=28")


def render_aspect(trimmed_path: str, out_path: str, aspect: str,
                   center_x_norm: float | None = None, ass_path: str | None = None,
                   badge_text: str | None = None, tier: str = "1080p"):
    w, h = probe_size(trimmed_path)
    tw, th = aspect_sizes(tier)[aspect]
    cw, ch, x, y = crop_rect_for_aspect(w, h, tw, th, center_x_norm)

    vf = [f"crop={cw}:{ch}:{x}:{y}", f"scale={tw}:{th}"]
    if ass_path:
        escaped = ass_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        vf.append(f"subtitles='{escaped}'")
    if badge_text:
        vf.append(_badge_filter(badge_text, th))

    cmd = ["ffmpeg", "-y", "-i", trimmed_path, "-vf", ",".join(vf),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-c:a", "aac", "-movflags", "+faststart", out_path]
    subprocess.run(cmd, check=True, capture_output=True)


def make_thumbnail(video_path: str, out_path: str, at_s: float = 0.3):
    cmd = ["ffmpeg", "-y", "-ss", f"{at_s:.2f}", "-i", video_path,
           "-frames:v", "1", "-q:v", "3", out_path]
    subprocess.run(cmd, check=True, capture_output=True)
