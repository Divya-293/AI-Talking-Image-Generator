"""Optional GFPGAN face restoration, applied frame by frame (the GFPGAN CLI only handles images)."""
from config import GFPGAN_MODEL_URL, PipelineError, ffmpeg_path, run_cmd

_restorer = None


def gfpgan_available():
    try:
        import gfpgan  # noqa: F401
        import cv2     # noqa: F401
        return True
    except Exception:
        return False


def _get_restorer():
    global _restorer
    if _restorer is None:
        from gfpgan import GFPGANer
        _restorer = GFPGANer(
            model_path=GFPGAN_MODEL_URL,   # auto-downloaded and cached on first use
            upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None,
        )
    return _restorer


def enhance_video(input_video, run_dir, weight=0.5, progress=None):
    """Restore faces in every frame, then re-attach the original audio.

    Returns (video_path, note). On any failure, returns the ORIGINAL video plus a note,
    so one optional step never kills the whole pipeline.
    """
    if not gfpgan_available():
        return input_video, "GFPGAN/OpenCV not installed - skipped enhancement."

    try:
        import cv2
        restorer = _get_restorer()

        cap = cv2.VideoCapture(str(input_video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        silent = run_dir / "enhanced_silent.mp4"
        writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _, _, restored = restorer.enhance(
                frame, has_aligned=False, only_center_face=True,
                paste_back=True, weight=weight,
            )
            out = restored if restored is not None else frame
            if out.shape[:2] != (h, w):
                out = cv2.resize(out, (w, h))
            writer.write(out)
            i += 1
            if progress:
                progress(min(i / total, 1.0))
        cap.release()
        writer.release()

        final = run_dir / "enhanced.mp4"
        run_cmd(
            [ffmpeg_path(), "-y", "-i", silent, "-i", input_video,
             "-map", "0:v:0", "-map", "1:a:0?",
             "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-shortest", final],
            step="Re-attaching audio",
        )
        return final, None

    except (PipelineError, Exception) as e:  # noqa: BLE001 - optional step, never fatal
        return input_video, f"Enhancement skipped ({type(e).__name__}: {e})"
