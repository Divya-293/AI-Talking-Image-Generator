import io

from PIL import Image, ImageOps

from config import (PipelineError, WAV2LIP_CHECKPOINT, WAV2LIP_DIR,
                    run_cmd, wav2lip_python)


def prepare_face_image(image_bytes, dest, max_side=1024):
    """Normalise an uploaded image: fix EXIF rotation, drop alpha, cap size, force even dims."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:
        raise PipelineError(f"Could not read the uploaded image: {e}") from e

    scale = min(1.0, max_side / max(img.size))
    w, h = int(img.width * scale) // 2 * 2, int(img.height * scale) // 2 * 2
    img = img.resize((max(w, 2), max(h, 2)), Image.LANCZOS)
    img.save(dest, "JPEG", quality=95)
    return dest


def run_wav2lip(face_path, audio_path, run_dir, pads=(0, 20, 0, 0), batch_size=16):
    """Run Wav2Lip inference and return the output video path."""
    if not WAV2LIP_CHECKPOINT.exists():
        raise PipelineError(f"Wav2Lip checkpoint missing: {WAV2LIP_CHECKPOINT}")

    out_path = (run_dir / "lipsync.mp4").resolve()
    (WAV2LIP_DIR / "temp").mkdir(exist_ok=True)   # Wav2Lip writes intermediates here

    cmd = [
        wav2lip_python(), "inference.py",
        "--checkpoint_path", WAV2LIP_CHECKPOINT.resolve(),
        "--face", face_path.resolve(),
        "--audio", audio_path.resolve(),
        "--pads", *map(str, pads),
        "--resize_factor", "1",
        "--face_det_batch_size", "1",
        "--wav2lip_batch_size", str(batch_size),
        "--nosmooth",
        "--outfile", out_path,
    ]

    try:
        run_cmd(cmd, step="Wav2Lip", cwd=WAV2LIP_DIR)
    except PipelineError as e:
        msg = str(e)
        if "Face not detected" in msg:
            msg += "\n\nTip: use a clear, front-facing photo with the whole face visible."
        raise PipelineError(msg) from e

    if not out_path.exists():
        raise PipelineError("Wav2Lip finished but produced no output file.")
    return out_path
