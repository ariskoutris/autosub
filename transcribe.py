#!/usr/bin/env python3
"""
Transcribe an audio/video file using OpenAI's transcription API and write in raw text lines or subtitle format.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Iterable, Optional, Tuple, List, Any

try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    _dotenv_path = find_dotenv(usecwd=True)
    if _dotenv_path:
        load_dotenv(_dotenv_path, override=False)
except Exception:
    pass

try:
    from openai import OpenAI
except Exception:
    sys.stderr.write(
        "Error: Failed to import the OpenAI Python SDK. Install dependencies with 'pip install -r requirements.txt'\n"
    )
    raise


def eprint(*args: object, **kwargs: object) -> None:
    """Print to stderr (wrapper to keep callsites terse)."""
    print(*args, file=sys.stderr, **kwargs)

# Debug printer (no-op unless --debug)
_DEBUG_ENABLED = False

def dprint(*args: object, **kwargs: object) -> None:
    """Conditional debug print to stderr controlled by --debug."""
    if _DEBUG_ENABLED:
        eprint(*args, **kwargs)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio/video to raw text lines")
    parser.add_argument(
        "input",
        help="Path to the audio/video file to transcribe (e.g., .mp3, .wav, .m4a, .mp4)",
    )
    parser.add_argument(
        "--model",
        default="whisper-1",
        help="Model to use (default: whisper-1). Example alternatives: gpt-4o-transcribe",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language code hint (e.g., 'en', 'es'). If omitted, model will auto-detect.",
    )
    parser.add_argument(
        "--extract-audio",
        choices=["auto", "always", "never"],
        default="auto",
        help=(
            "If 'auto', extract audio from video inputs using ffmpeg. "
            "If 'always', always extract to a compact audio file before uploading. "
            "If 'never', upload the original file as-is."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save transcript. If a directory, saves as <dir>/<basename>.txt. If omitted, saves to ./<basename>.txt",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print transcript to stdout; only save to file.",
    )
    parser.add_argument(
        "--srt",
        action="store_true",
        help="Also save an .srt subtitle file. Requires whisper-1.",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Translate speech to English instead of same-language transcription. Requires whisper-1.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging to stderr for chunking and I/O.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        eprint("OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)
    return OpenAI()


def print_lines(lines: Iterable[str]) -> None:
    for line in lines:
        line = (line or "").strip()
        if line:
            print(line)


def _ensure_dir_for(path: str) -> None:
    """Create the parent directory for a file path, handling relative roots."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent if os.path.isabs(parent) else os.path.join(os.getcwd(), parent), exist_ok=True)


def _compute_output_base(in_path: str, out_arg: Optional[str]) -> str:
    """Return the base output path WITHOUT the .txt/.srt suffix."""
    in_base = os.path.splitext(os.path.basename(in_path))[0]
    if not out_arg:
        out_dir = os.getcwd()
        return os.path.join(out_dir, in_base)
    if os.path.isdir(out_arg) or out_arg.endswith(os.sep):
        out_dir = out_arg if os.path.isabs(out_arg) else os.path.join(os.getcwd(), out_arg)
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, in_base)
    # Treat as a file path base
    root, ext = os.path.splitext(out_arg)
    if ext.lower() in (".txt", ".srt"):
        base = root
    else:
        base = out_arg
    _ensure_dir_for(base)
    return base


def resolve_output_path(in_path: str, out_arg: Optional[str]) -> str:
    """Return the TXT output path, enforcing a .txt suffix based on the rules above."""
    base = _compute_output_base(in_path, out_arg)
    return base + ".txt"


def resolve_output_path_with_ext(in_path: str, out_arg: Optional[str], ext: str) -> str:
    """Return the output path with the given extension using the same base rules."""
    base = _compute_output_base(in_path, out_arg)
    return base + ext


def _format_srt_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp 'HH:MM:SS,mmm'."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000.0))
    hours = total_ms // 3_600_000
    remain = total_ms % 3_600_000
    minutes = remain // 60_000
    remain = remain % 60_000
    secs = remain // 1000
    ms = remain % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


# Max content size (25 MiB) for Whisper API uploads
MAX_CONTENT_BYTES = 25 * 1024 * 1024  # 26,214,400
_SAFETY_MARGIN = 64 * 1024  # leave a little room for headers/container


def _file_size(path: str) -> int:
    """Return file size in bytes; 0 if unavailable."""
    try:
        return os.path.getsize(path)
    except Exception:
        return 0


def _run_ffprobe_duration(path: str) -> Optional[float]:
    """Return media duration in seconds using ffprobe, or None if unavailable."""
    try:
        import subprocess
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        out = (proc.stdout or "").strip()
        return float(out) if out else None
    except Exception:
        return None


def _ffmpeg_available() -> bool:
    """True if ffmpeg binary is found on PATH."""
    try:
        import shutil
        return shutil.which("ffmpeg") is not None
    except Exception:
        return False


def _split_with_ffmpeg_by_time_copy(
    input_path: str, chunk_seconds: int, out_dir: Optional[str] = None, base_prefix: Optional[str] = None
) -> Tuple[List[str], Optional[str]]:
    """Time-split input using ffmpeg with stream copy; return (chunk_paths, tmp_dir_or_None)."""
    import tempfile
    import subprocess
    import os as _os

    tmpdir = out_dir or tempfile.mkdtemp(prefix="transcribe-chunks-")
    base_name = base_prefix or os.path.splitext(os.path.basename(input_path))[0]
    ext = os.path.splitext(input_path)[1] or ".mp4"
    pattern = _os.path.join(tmpdir, f"{base_name}.part-%03d{ext}")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        input_path,
        "-map",
        "0",
        "-c",
        "copy",
        "-f",
        "segment",
        "-segment_time",
        str(int(max(1, chunk_seconds))),
        "-reset_timestamps",
        "1",
        pattern,
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
    # Clean up temp dir we created and propagate
        if out_dir is None:
            try:
                import shutil as _shutil
                _shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass
        raise RuntimeError(f"ffmpeg split failed: {e}")

    files = sorted([
        os.path.join(tmpdir, f)
        for f in os.listdir(tmpdir)
        if f.startswith(base_name + ".part-") and f.endswith(ext)
    ])
    return files, (None if out_dir else tmpdir)


def _split_to_size_limit(input_path: str, max_bytes: int) -> Tuple[List[str], Optional[str]]:
    """Split file into <= max_bytes chunks via repeated time splits (no re-encode)."""
    import math
    import tempfile
    import shutil

    root_tmp = tempfile.mkdtemp(prefix="transcribe-chunks-")
    size_bytes = max(_file_size(input_path), 1)
    est_chunks = max(2, math.ceil(size_bytes / max_bytes))
    duration = _run_ffprobe_duration(input_path)
    if duration and duration > 0:
        chunk_seconds = int(math.ceil(duration / est_chunks))
    else:
        chunk_seconds = 600  # fallback 10 minutes

    chunks, _ = _split_with_ffmpeg_by_time_copy(input_path, chunk_seconds, out_dir=root_tmp)
    if not chunks:
        # nothing produced; copy original file into tmp and return it
        try:
            base = os.path.basename(input_path)
            dst = os.path.join(root_tmp, base)
            shutil.copy2(input_path, dst)
            chunks = [dst]
        except Exception as e:
            shutil.rmtree(root_tmp, ignore_errors=True)
            raise RuntimeError(f"Failed to prepare chunks: {e}")

    # Iteratively split any oversized chunks further
    i = 0
    while i < len(chunks):
        p = chunks[i]
        if _file_size(p) <= max_bytes:
            i += 1
            continue
    # Split this chunk further based on its duration
        c_dur = _run_ffprobe_duration(p)
        if not c_dur or c_dur <= 1:
            # If duration unknown or too small, split into 2 equal parts by time
            sub_chunk_seconds = max(1, chunk_seconds // 2 or 1)
        else:
            n = max(2, math.ceil(_file_size(p) / max_bytes))
            sub_chunk_seconds = int(math.ceil(float(c_dur) / n))
            sub_chunk_seconds = max(1, sub_chunk_seconds)

    # Place sub-chunks into a subdirectory to avoid name collisions
        sub_dir_name = os.path.splitext(os.path.basename(p))[0]
        sub_out_dir = os.path.join(root_tmp, sub_dir_name)
        os.makedirs(sub_out_dir, exist_ok=True)
        sub_base_prefix = sub_dir_name
        try:
            subs, _ = _split_with_ffmpeg_by_time_copy(p, sub_chunk_seconds, out_dir=sub_out_dir, base_prefix=sub_base_prefix)
        except Exception as e:
            # If splitting fails, keep the original to avoid infinite loop
            eprint(f"Warning: failed to further split chunk '{p}': {e}")
            i += 1
            continue

        if not subs:
            # No output; keep original
            i += 1
            continue

    # Replace the current chunk with its sub-chunks in order
        chunks.pop(i)
        for j, sp in enumerate(subs):
            chunks.insert(i + j, sp)
        # Do not increment i; re-check the first inserted sub-chunk for size

    # Final size guard: ensure all are within limit (best-effort)
    overs = [p for p in chunks if _file_size(p) > max_bytes]
    if overs:
        eprint("Warning: Some chunks still exceed the size limit; API may reject them:")
        for p in overs:
            eprint(f" - {os.path.basename(p)}: {_file_size(p)} bytes > {max_bytes}")

    return chunks, root_tmp


def _parse_transcription_response(resp: Any, use_verbose: bool) -> Tuple[List[str], Optional[List[dict]]]:
    """Normalize API response into text lines and optional segments."""
    lines: List[str] = []
    segments: Optional[List[dict]] = None
    try:
        if use_verbose:
            if hasattr(resp, "segments") and getattr(resp, "segments"):
                segs = getattr(resp, "segments")
                segments = []
                for seg in segs:
                    text = getattr(seg, "text", "") or ""
                    start = getattr(seg, "start", None)
                    end = getattr(seg, "end", None)
                    lines.append(text)
                    segments.append({"text": text, "start": start, "end": end})
            elif hasattr(resp, "text") and getattr(resp, "text"):
                lines.extend(str(getattr(resp, "text")).splitlines())
            else:
                try:
                    raw = json.loads(resp.json()) if hasattr(resp, "json") else json.loads(str(resp))
                    if isinstance(raw, dict):
                        if "segments" in raw and isinstance(raw["segments"], list):
                            segments = []
                            for s in raw["segments"]:
                                if isinstance(s, dict):
                                    text = str(s.get("text", ""))
                                    lines.append(text)
                                    segments.append({
                                        "text": text,
                                        "start": s.get("start"),
                                        "end": s.get("end"),
                                    })
                        elif "text" in raw:
                            lines.extend(str(raw.get("text", "")).splitlines())
                except Exception:
                    pass
        else:
            if hasattr(resp, "text") and getattr(resp, "text"):
                lines.extend(str(getattr(resp, "text")).splitlines())
            else:
                try:
                    raw = json.loads(resp.json()) if hasattr(resp, "json") else json.loads(str(resp))
                    if isinstance(raw, dict) and "text" in raw:
                        lines.extend(str(raw.get("text", "")).splitlines())
                except Exception:
                    pass
    except Exception as e:
        raise RuntimeError(f"Failed to parse transcription response: {e}")

    return lines, segments


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    global _DEBUG_ENABLED
    _DEBUG_ENABLED = bool(getattr(args, "debug", False))

    if getattr(args, "srt", False) and args.model != "whisper-1":
        eprint("Error: --srt requires model 'whisper-1'.")
        return 1
    if getattr(args, "translate", False) and args.model != "whisper-1":
        eprint("Error: --translate requires model 'whisper-1'.")
        return 1

    if not os.path.isfile(args.input):
        eprint(f"Error: File not found: {args.input}")
        return 1

    dprint(f"Input: {args.input}")

    client = get_client()

    input_path = args.input
    temp_cleanup: list[str] = []
    try:
        from extract_audio import is_video_file, extract_audio, check_ffmpeg_available  # type: ignore
    except Exception:
        is_video_file = None  # type: ignore
        extract_audio = None  # type: ignore
        check_ffmpeg_available = None  # type: ignore

    def _should_extract() -> bool:
        if args.extract_audio == "never":
            return False
        if args.extract_audio == "always":
            return True
        if is_video_file is not None:
            try:
                return bool(is_video_file(args.input))
            except Exception:
                return False
        return False

    extracted_path: Optional[str] = None
    if _should_extract() and extract_audio is not None:
        try:
            if check_ffmpeg_available and not check_ffmpeg_available():
                eprint("ffmpeg not found on PATH; proceeding without extraction.")
            else:
                tmpdir = tempfile.mkdtemp(prefix="transcribe-audio-")
                dprint("Extracting audio to temp dir:", tmpdir)
                extracted_path = extract_audio(args.input, output_dir=tmpdir, fmt="mp3", channels=1, rate=16000, bitrate="64k")
                input_path = extracted_path
                temp_cleanup.append(tmpdir)
                dprint("Extracted audio path:", input_path, "size=", os.path.getsize(input_path))
        except Exception as ex:
            eprint(f"Audio extraction failed, continuing with original file: {ex}")

    response_format = "verbose_json" if getattr(args, "srt", False) else "json"
    use_verbose = (response_format == "verbose_json")
    dprint("Model:", args.model, "translate=", getattr(args, "translate", False), "srt=", getattr(args, "srt", False), "response_format=", response_format)

    def _transcribe_one(path: str) -> Tuple[List[str], Optional[List[dict]]]:
        dprint(f"Transcribing file: {path} (size={_file_size(path)})")
        try:
            with open(path, "rb") as f:
                common_kwargs = {
                    "model": args.model,
                    "file": f,
                    "response_format": response_format,
                }

                if getattr(args, "translate", False):
                    resp = client.audio.translations.create(**common_kwargs)
                else:
                    if args.language:
                        common_kwargs["language"] = args.language
                    resp = client.audio.transcriptions.create(**common_kwargs)
        except Exception as e:
            raise RuntimeError(f"Transcription request failed: {e}")
        lines, segments = _parse_transcription_response(resp, use_verbose)
        dprint(
            f"Transcribed: lines={len(lines)}; segments={'None' if segments is None else len(segments)}"
        )
        return lines, segments

    max_bytes = MAX_CONTENT_BYTES - _SAFETY_MARGIN
    oversized = _file_size(input_path) > max_bytes
    dprint(
        f"max_bytes={max_bytes}; input_size={_file_size(input_path)}; oversized={oversized}"
    )

    lines: List[str] = []
    combined_segments: Optional[List[dict]] = None

    if not oversized and _file_size(input_path) <= max_bytes:
        dprint("Single upload path; no chunking required.")
        try:
            lns, segs = _transcribe_one(input_path)
        except Exception as e:
            eprint(str(e))
            # Clean up temp files if any
            for p in temp_cleanup:
                try:
                    if os.path.isdir(p):
                        import shutil
                        shutil.rmtree(p, ignore_errors=True)
                except Exception:
                    pass
            return 1
        lines = lns
        combined_segments = segs
    else:
        dprint("Chunking required; checking ffmpeg availability…")
        if not _ffmpeg_available():
            eprint("Input exceeds 25MB and ffmpeg is not available to split into chunks. Please install ffmpeg.")
            # Clean up temp files if any
            for p in temp_cleanup:
                try:
                    if os.path.isdir(p):
                        import shutil
                        shutil.rmtree(p, ignore_errors=True)
                except Exception:
                    pass
            return 1

        size_bytes = max(_file_size(input_path), 1)
        import math
        est_chunks = max(2, math.ceil(size_bytes / max_bytes))
        duration = _run_ffprobe_duration(input_path)
        if duration and duration > 0:
            chunk_seconds = int(math.ceil(duration / est_chunks))
        else:
            chunk_seconds = 600
        dprint(
            f"estimated chunks={est_chunks}; duration={duration}; initial chunk_seconds={chunk_seconds}"
        )

        try:
            chunk_paths, chunk_tmpdir = _split_to_size_limit(input_path, max_bytes)
            if chunk_tmpdir:
                temp_cleanup.append(chunk_tmpdir)
            dprint(f"Produced {len(chunk_paths)} chunk(s):")
            for i, ch in enumerate(chunk_paths, 1):
                dprint(
                    f"  [{i}] {os.path.basename(ch)} size={_file_size(ch)} bytes; dur={_run_ffprobe_duration(ch)} s"
                )
        except Exception as e:
            eprint(str(e))
            for p in temp_cleanup:
                try:
                    if os.path.isdir(p):
                        import shutil
                        shutil.rmtree(p, ignore_errors=True)
                except Exception:
                    pass
            return 1

        offset = 0.0
        for idx, ch in enumerate(chunk_paths):
            dprint(f"Processing chunk {idx+1}/{len(chunk_paths)}: {os.path.basename(ch)} size={_file_size(ch)}")
            try:
                lns, segs = _transcribe_one(ch)
            except Exception as e:
                eprint(f"Chunk {idx+1}/{len(chunk_paths)} failed: {e}")
                continue
            lines.extend(lns)
            dprint(f"  -> chunk produced lines={len(lns)}; segments={'None' if segs is None else len(segs)}")
            if segs:
                if combined_segments is None:
                    combined_segments = []
                for s in segs:
                    start = s.get("start")
                    end = s.get("end")
                    text = s.get("text", "")
                    if start is not None and end is not None:
                        try:
                            s_start = float(start) + float(offset)
                            s_end = float(end) + float(offset)
                        except Exception:
                            s_start, s_end = start, end
                        combined_segments.append({"text": text, "start": s_start, "end": s_end})
                    else:
                        combined_segments.append({"text": text, "start": start, "end": end})
            ch_dur = _run_ffprobe_duration(ch)
            if ch_dur:
                offset += float(ch_dur)
            else:
                offset += float(chunk_seconds)
            dprint(f"  -> updated offset={offset}")

    if not lines:
        eprint("No text lines found in transcription response.")
        return 2

    out_path = resolve_output_path(args.input, args.output)
    dprint(f"Writing transcript to: {out_path}")
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            for ln in lines:
                if ln:
                    fh.write(ln.strip() + "\n")
        dprint(f"Wrote transcript bytes={os.path.getsize(out_path)}")
    except Exception as e:
        eprint(f"Failed to write transcript to {out_path}: {e}")
        return 1

    if not args.quiet:
        print_lines(lines)

    if args.srt and combined_segments:
        def _seg_get(seg, key, default=None):
            if isinstance(seg, dict):
                return seg.get(key, default)
            return getattr(seg, key, default)

        srt_path = resolve_output_path_with_ext(args.input, args.output, ".srt")
        dprint(f"Writing SRT to: {srt_path}")
        try:
            with open(srt_path, "w", encoding="utf-8") as sf:
                idx = 1
                for seg in combined_segments:
                    text = (_seg_get(seg, "text", "") or "").strip()
                    start = _seg_get(seg, "start", None)
                    end = _seg_get(seg, "end", None)
                    # If no timing, skip SRT line
                    if text and start is not None and end is not None:
                        sf.write(f"{idx}\n")
                        sf.write(f"{_format_srt_timestamp(float(start))} --> {_format_srt_timestamp(float(end))}\n")
                        sf.write(f"{text}\n\n")
                        idx += 1
            dprint(f"Wrote SRT entries count={len(combined_segments)}")
        except Exception as e:
            eprint(f"Failed to write SRT: {e}")
    return 0




if __name__ == "__main__":
    sys.exit(main())
