#!/usr/bin/env python3
"""Batch-transcribe media files in a folder using transcribe.py."""
from __future__ import annotations

import argparse
import concurrent.futures
import glob
import os
import subprocess
import sys
import time
from typing import Iterable, Optional, Sequence

try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore

    _dotenv_path = find_dotenv(usecwd=True)
    if _dotenv_path:
        load_dotenv(_dotenv_path, override=False)
except Exception:
    pass


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch transcribe a folder of media files")
    p.add_argument("input_dir", help="Directory containing media files")
    p.add_argument(
        "--output-dir",
        default="transcripts",
        help="Directory to write transcripts (default: ./transcripts)",
    )
    p.add_argument(
        "--pattern",
        default="*.{mp3,wav,m4a,mp4,mov,aac,flac,ogg,mkv}",
        help="Glob pattern for files (default: '*.{mp3,wav,m4a,mp4,mov,aac,flac,ogg,mkv}')",
    )
    p.add_argument("--model", default="whisper-1")
    p.add_argument("--language", default=None)
    p.add_argument("--concurrency", type=int, default=min(4, os.cpu_count() or 2))
    p.add_argument("--skip-existing", action="store_true", help="Skip files whose .txt already exists")
    p.add_argument("--srt", action="store_true", help="Also write .srt files")
    p.add_argument("--translate", action="store_true", help="Translate speech to English instead of same-language transcription")
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable live heartbeat progress line during processing",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def discover_files(input_dir: str, pattern: str) -> list[str]:
    """Return files in input_dir matching pattern; supports brace sets like '*.{mp3,wav}'."""
    files: list[str] = []
    lbrace = pattern.find("{")
    rbrace = pattern.find("}", lbrace + 1) if lbrace != -1 else -1
    if lbrace != -1 and rbrace != -1 and rbrace > lbrace:
        prefix = pattern[:lbrace]
        suffix = pattern[rbrace + 1 :]
        inner = pattern[lbrace + 1 : rbrace]
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        for part in parts:
            subpattern = f"{prefix}{part}{suffix}"
            files.extend(glob.glob(os.path.join(input_dir, subpattern)))
    else:
        files = glob.glob(os.path.join(input_dir, pattern))
    return sorted([f for f in files if os.path.isfile(f)])


def output_txt_path(input_path: str, output_dir: str) -> str:
    """Compute the .txt transcript path under output_dir and ensure the directory exists."""
    base = os.path.splitext(os.path.basename(input_path))[0] + ".txt"
    out_dir = output_dir if os.path.isabs(output_dir) else os.path.join(os.getcwd(), output_dir)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, base)


def build_cmd(transcribe_py: str, in_path: str, out_dir: str, args: argparse.Namespace) -> Sequence[str]:
    """Build the command to invoke transcribe.py for a single input file."""
    cmd = [sys.executable, transcribe_py, in_path, "--output", out_dir, "--model", args.model]
    if args.language:
        cmd += ["--language", args.language]
    if args.srt:
        cmd += ["--srt"]
    if getattr(args, "translate", False):
        cmd += ["--translate"]
    cmd += ["--quiet"]
    return cmd


def run_one(transcribe_py: str, in_path: str, out_dir: str, args: argparse.Namespace) -> tuple[str, int, str]:
    """Execute transcription for one file; return (input_path, exit_code, message)."""
    cmd = build_cmd(transcribe_py, in_path, out_dir, args)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        code = proc.returncode
        if code == 0:
            return (in_path, 0, "ok")
        else:
            msg = proc.stderr.strip() or proc.stdout.strip()
            return (in_path, code, msg)
    except Exception as e:
        return (in_path, 1, f"exception: {e}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    input_dir = os.path.abspath(args.input_dir)
    # Resolve and create output directory upfront so transcribe.py detects it as a directory
    out_dir = args.output_dir
    out_dir_abs = out_dir if os.path.isabs(out_dir) else os.path.join(os.getcwd(), out_dir)
    os.makedirs(out_dir_abs, exist_ok=True)

    if not os.path.isdir(input_dir):
        print(f"Error: input directory not found: {input_dir}", file=sys.stderr)
        return 1

    # Find transcribe.py next to this script
    here = os.path.dirname(os.path.abspath(__file__))
    transcribe_py = os.path.join(here, "transcribe.py")
    if not os.path.isfile(transcribe_py):
        print("Error: transcribe.py not found beside batch_transcribe.py", file=sys.stderr)
        return 1

    files = discover_files(input_dir, args.pattern)
    if not files:
        print("No input files matched.")
        return 0

    if args.skip_existing:
        before = len(files)
        files = [f for f in files if not os.path.exists(output_txt_path(f, out_dir_abs))]
        skipped = before - len(files)
        if skipped:
            print(f"Skipping {skipped} existing transcripts")

    total = len(files)
    print(f"Processing {total} files with concurrency={args.concurrency} -> {out_dir_abs}")

    errors: list[tuple[str, int, str]] = []
    if args.concurrency and args.concurrency > 1:
        # Parallel mode with a lightweight heartbeat on stderr to reassure the user
        heartbeat_enabled = (not args.no_progress) and sys.stderr.isatty()
        heartbeat_interval = 1.5  # seconds between updates
        start_ts = time.time()
        last_hb_ts = 0.0
        last_hb_len = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            fut_map: dict[concurrent.futures.Future[tuple[str, int, str]], str] = {}
            for f in files:
                fut = ex.submit(run_one, transcribe_py, f, out_dir_abs, args)
                fut_map[fut] = f

            done_count = 0
            pending: set[concurrent.futures.Future[tuple[str, int, str]]] = set(fut_map.keys())

            def _print_heartbeat() -> None:
                nonlocal last_hb_ts, last_hb_len
                if not heartbeat_enabled:
                    return
                now = time.time()
                if now - last_hb_ts < heartbeat_interval:
                    return
                elapsed = int(now - start_ts)
                remaining = total - done_count
                # Approximate running as min(concurrency, remaining) but cannot exceed current pending
                approx_running = min(args.concurrency or 1, remaining, len(pending))
                msg = (
                    f"Working… {done_count}/{total} done, remaining={remaining}, "
                    f"running≈{approx_running}, elapsed {elapsed}s"
                )
                # erase previous
                if last_hb_len:
                    sys.stderr.write("\r" + (" " * last_hb_len) + "\r")
                sys.stderr.write(msg)
                sys.stderr.flush()
                last_hb_len = len(msg)
                last_hb_ts = now

            while pending:
                done, not_done = concurrent.futures.wait(
                    pending, timeout=0.2, return_when=concurrent.futures.FIRST_COMPLETED
                )
                # Heartbeat during waits
                _print_heartbeat()

                for fut in done:
                    # Clear heartbeat line before printing a completion line
                    if heartbeat_enabled and last_hb_len:
                        sys.stderr.write("\r" + (" " * last_hb_len) + "\r")
                        sys.stderr.flush()

                    pending.discard(fut)
                    done_count += 1
                    try:
                        in_path, code, msg = fut.result()
                    except Exception as exc:
                        in_path = fut_map.get(fut, "<unknown>")
                        code = 1
                        msg = f"exception: {exc}"
                    name = os.path.basename(in_path)
                    status = "OK" if code == 0 else f"ERR({code})"
                    remaining = total - done_count
                    print(f"[{done_count}/{total}] {name}: {status} (remaining: {remaining})")
                    if code != 0:
                        errors.append((in_path, code, msg))

            # Final newline to move past heartbeat line if visible
            if heartbeat_enabled and last_hb_len:
                sys.stderr.write("\n")
                sys.stderr.flush()
    else:
        for i, f in enumerate(files, start=1):
            # Print a quick "starting" line for immediate feedback in sequential mode
            print(f"Starting [{i}/{total}] {os.path.basename(f)}…", flush=True)
            in_path, code, msg = run_one(transcribe_py, f, out_dir_abs, args)
            name = os.path.basename(in_path)
            status = "OK" if code == 0 else f"ERR({code})"
            print(f"[{i}/{total}] {name}: {status}")
            if code != 0:
                errors.append((in_path, code, msg))

    if errors:
        print("\nCompleted with errors:", file=sys.stderr)
        for in_path, code, msg in errors:
            print(f"- {os.path.basename(in_path)} -> exit {code}: {msg}", file=sys.stderr)
        return 2

    print("\nAll files processed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
