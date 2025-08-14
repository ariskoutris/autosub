#!/usr/bin/env python3
"""
Audio extraction utilities using ffmpeg.

- check_ffmpeg_available() -> bool
- is_video_file(path: str) -> bool
- extract_audio(input_path: str, output_dir: Optional[str] = None,
                fmt: str = "mp3", channels: int = 1, rate: int = 16000,
                bitrate: str = "64k") -> str

This module intentionally keeps no external Python dependencies and shells out
to the local ffmpeg binary. Users must install ffmpeg (e.g., on macOS via Homebrew).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Optional


VIDEO_EXTS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv",
}


def check_ffmpeg_available() -> bool:
    """Return True if ffmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None


def is_video_file(path: str) -> bool:
    """True if file extension indicates a video container (used to decide extraction)."""
    ext = os.path.splitext(path)[1].lower()
    return ext in VIDEO_EXTS


def _resolve_output_path(input_path: str, output_dir: Optional[str], fmt: str) -> str:
    """Choose output path for the extracted audio; defaults to a temp dir if none given."""
    base = os.path.splitext(os.path.basename(input_path))[0]
    filename = f"{base}-audio.{fmt}"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)
    tmpdir = tempfile.mkdtemp(prefix="extract-audio-")
    return os.path.join(tmpdir, filename)


def extract_audio(
    input_path: str,
    output_dir: Optional[str] = None,
    fmt: str = "mp3",
    channels: int = 1,
    rate: int = 16000,
    bitrate: str = "64k",
) -> str:
    """Extract audio from a media file using ffmpeg and return the output path.

    The audio will be re-encoded to the specified format, channels, sample rate,
    and bitrate for optimal speech-to-text performance and small upload sizes.
    """
    if not check_ffmpeg_available():
        raise RuntimeError(
            "ffmpeg not found. Please install it (e.g., `brew install ffmpeg` on macOS)."
        )

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = _resolve_output_path(input_path, output_dir, fmt)

    # Build ffmpeg command (-y overwrite, -vn drop video, -ac channels, -ar sample rate, -b:a bitrate)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", input_path,
        "-vn",
        "-ac", str(channels),
        "-ar", str(rate),
        "-b:a", str(bitrate),
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        # If conversion to mp3 fails (e.g., due to licensing on some builds), try AAC in M4A as a fallback
        if fmt.lower() == "mp3":
            alt_output = _resolve_output_path(input_path, output_dir, "m4a")
            alt_cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-i", input_path,
                "-vn",
                "-ac", str(channels),
                "-ar", str(rate),
                "-c:a", "aac",
                "-b:a", str(bitrate),
                alt_output,
            ]
            subprocess.run(alt_cmd, check=True)
            return alt_output
        raise RuntimeError(f"ffmpeg failed: {e}")

    return output_path
