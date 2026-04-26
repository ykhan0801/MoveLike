# YouTube downloads and video trimming. Kept UI-free so it can be
# dropped into a backend service later without changes.

import os
import tempfile
import yt_dlp
from moviepy import VideoFileClip


def download_youtube_video(url: str, output_dir: str) -> str:
    # Grab a video from YouTube and return the local .mp4 path.
    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        if not filename.endswith(".mp4"):
            filename = os.path.splitext(filename)[0] + ".mp4"

    if not os.path.exists(filename):
        mp4_files = [f for f in os.listdir(output_dir) if f.endswith(".mp4")]
        if not mp4_files:
            raise RuntimeError("Download completed but no .mp4 file found in output directory.")
        filename = os.path.join(output_dir, mp4_files[0])

    return filename


def trim_video(input_path: str, start_sec: float, end_sec: float, output_path: str) -> str:
    # Cut a clip down to [start_sec, end_sec] and write it to output_path.
    with VideoFileClip(input_path) as clip:
        duration = clip.duration
        start_sec = max(0, start_sec)
        end_sec = min(duration, end_sec)

        if end_sec <= start_sec:
            raise ValueError(f"End time ({end_sec}s) must be after start time ({start_sec}s)")

        trimmed = clip.subclipped(start_sec, end_sec)
        trimmed.write_videofile(
            output_path,
            codec="libx264",
            audio=False,
            logger=None,
        )

    return output_path


def get_video_duration(path: str) -> float:
    with VideoFileClip(path) as clip:
        return clip.duration


def seconds_to_hhmmss(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
