import os
import tempfile
from py_subtitle_extractor import (
    extract_subtitle_tracks,
    extract_subtitles_as_srt,
)

def list_external_subtitles(folder_path):
    try:
        names = os.listdir(folder_path)
    except OSError:
        return []
    paths = sorted(
        os.path.join(folder_path, n)
        for n in names
        if n.lower().endswith(".srt")
    )
    return [
        {"label": os.path.basename(p), "type": "external", "path": p}
        for p in paths
    ]

def list_embedded_subtitles(video_path):
    return [
        {
            "type": "embedded",
            "video_path": video_path,
            "index": t["track_number"],
            "language": t["language"] or "und"
        }
        for t in extract_subtitle_tracks(video_path)
        if t["codec_id"].startswith("S_TEXT")
    ]

def list_available_subtitles(video_path):
    folder = os.path.dirname(video_path)
    external = list_external_subtitles(folder)
    embedded = list_embedded_subtitles(video_path) if video_path.lower().endswith(".mkv") else []
    return external + embedded

def extract_to_temp_srt(video_path, index, on_progress=None):
    srt_text = extract_subtitles_as_srt(video_path, index, on_progress)

    tracks = extract_subtitle_tracks(video_path)
    lang = next((t["language"] for t in tracks if t["track_number"] == index), "und")

    folder = os.path.dirname(video_path)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(folder, f"{base}.{lang}.extracted.srt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(srt_text)

    return out_path


