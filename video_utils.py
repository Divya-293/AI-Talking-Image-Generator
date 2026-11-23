from config import PipelineError, ffmpeg_path, media_duration, run_cmd

FPS = 30


def process_video(input_video, output_path, duration, orientation,
                  background="Blurred", fit_duration=True):
    """Place the clip on a 1080x1920 / 1920x1080 canvas with ffmpeg (no moviepy needed).

    - Scales to fit (never crops the face), centred on a black or blurred background.
    - Longer than `duration`  -> trimmed.
    - Shorter and fit_duration -> last frame held and audio padded with silence.
    """
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise PipelineError("ffmpeg not found on PATH.")
    if not input_video or not input_video.exists():
        raise PipelineError("No video to process.")

    W, H = (1080, 1920) if orientation == "Portrait" else (1920, 1080)
    src = media_duration(input_video) or duration
    pad = fit_duration and src < duration
    limit = duration if (pad or src > duration) else None

    pre = f"fps={FPS},"
    if pad:
        pre += f"tpad=stop_mode=clone:stop_duration={duration - src + 0.5:.2f},"

    if background == "Blurred":
        vf = (
            f"[0:v]{pre}split=2[a][b];"
            f"[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=30:5[bg];"
            f"[b]scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]"
        )
    else:
        vf = (
            f"[0:v]{pre}scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p[v]"
        )

    cmd = [ffmpeg, "-y", "-i", input_video, "-filter_complex", vf, "-map", "[v]", "-map", "0:a:0?"]
    if pad:
        cmd += ["-af", "apad"]
    if limit:
        cmd += ["-t", f"{limit}"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", output_path]

    run_cmd(cmd, step="Final video render")
    if not output_path.exists():
        raise PipelineError("Final render produced no file.")
    return output_path
