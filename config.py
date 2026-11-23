"""Shared paths, settings and helpers for the talking-image pipeline."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WORK_DIR = BASE_DIR / "input"      # per-run temp files
OUTPUT_DIR = BASE_DIR / "output"   # final videos

WAV2LIP_DIR = BASE_DIR / "Wav2Lip"
WAV2LIP_CHECKPOINT = WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pth"

GFPGAN_MODEL_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
)


class PipelineError(RuntimeError):
    """A pipeline step failed; the message is safe to show to the user."""


def ensure_dirs():
    WORK_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def ffmpeg_path():
    return shutil.which("ffmpeg")


def ffprobe_path():
    return shutil.which("ffprobe")


def wav2lip_python():
    """Python interpreter that has Wav2Lip's dependencies installed.

    Priority: WAV2LIP_PYTHON env var -> ./venv310 (Windows/Linux/Mac) -> current python.
    """
    env = os.environ.get("WAV2LIP_PYTHON")
    if env and Path(env).exists():
        return env
    for candidate in (
        BASE_DIR / "venv310" / "Scripts" / "python.exe",
        BASE_DIR / "venv310" / "bin" / "python",
    ):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def run_cmd(cmd, step, cwd=None):
    """Run a command, capture output, raise PipelineError with the log tail on failure."""
    cmd = [str(c) for c in cmd]
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
    except FileNotFoundError as e:
        raise PipelineError(f"{step}: '{cmd[0]}' was not found. Is it installed / on PATH?") from e
    if proc.returncode != 0:
        log = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise PipelineError(f"{step} failed (exit code {proc.returncode}):\n" + "\n".join(log[-15:]))
    return proc


def media_duration(path):
    """Duration in seconds via ffprobe (returns None if unavailable)."""
    probe = ffprobe_path()
    if not probe:
        return None
    try:
        out = run_cmd(
            [probe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            step="ffprobe",
        ).stdout.strip()
        return float(out)
    except (PipelineError, ValueError):
        return None


def preflight():
    """Return a list of (ok, message) checks for the sidebar."""
    return [
        (bool(ffmpeg_path()), "ffmpeg on PATH"),
        (WAV2LIP_DIR.exists(), "Wav2Lip folder"),
        (WAV2LIP_CHECKPOINT.exists(), "wav2lip_gan.pth checkpoint"),
    ]
