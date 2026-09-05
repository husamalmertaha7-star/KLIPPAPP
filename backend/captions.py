"""
Builds an .ass subtitle file from word-level timestamps (only available when
the OpenAI transcription backend is enabled — see transcription.py). Rendered
via ffmpeg's built-in libass, so no extra dependency is needed.

If no words are available (heuristic-only mode), build_ass() returns None and
the renderer simply skips the subtitles filter — no fake captions are drawn.
"""
from __future__ import annotations

from transcription import Word

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {res_x}
PlayResY: {res_y}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Poppins,{fontsize},&H00FFFFFF,&H000000FF,&H00101010,&H99000000,1,0,0,0,100,100,0,0,1,{outline},2,2,60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ts(t: float) -> str:
    if t < 0:
        t = 0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _chunk_words(words: list[Word], max_words: int = 5, max_dur: float = 2.4):
    chunk: list[Word] = []
    for w in words:
        if chunk and (len(chunk) >= max_words or (w.end - chunk[0].start) > max_dur):
            yield chunk
            chunk = []
        chunk.append(w)
    if chunk:
        yield chunk


def build_ass(words: list[Word] | None, out_path: str, res_x: int, res_y: int) -> str | None:
    if not words:
        return None

    fontsize = max(28, int(res_y * 0.045))
    outline = max(2, int(fontsize * 0.08))
    margin_v = int(res_y * 0.12)

    lines = [ASS_HEADER.format(res_x=res_x, res_y=res_y, fontsize=fontsize,
                                outline=outline, margin_v=margin_v)]
    for chunk in _chunk_words(words):
        start, end = chunk[0].start, chunk[-1].end
        if end <= start:
            continue
        text = " ".join(w.text.strip() for w in chunk if w.text.strip())
        text = text.replace("{", "(").replace("}", ")")
        if not text:
            continue
        lines.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Caption,,0,0,0,,{text.upper()}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return out_path
