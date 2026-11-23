from gtts import gTTS

from config import PipelineError, ffmpeg_path, media_duration, run_cmd


def generate_voice(text, run_dir, lang="en", slow=False):
    """Text -> 16 kHz mono WAV. Returns (wav_path, duration_seconds)."""
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise PipelineError("ffmpeg not found on PATH.")

    mp3_path = run_dir / "voice.mp3"
    wav_path = run_dir / "voice.wav"

    try:
        gTTS(text=text, lang=lang, slow=slow).save(str(mp3_path))
    except Exception as e:  # gTTS raises several different exception types
        raise PipelineError(f"Voice generation failed (gTTS needs internet access): {e}") from e

    run_cmd([ffmpeg, "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", wav_path],
            step="Audio conversion")

    if not wav_path.exists():
        raise PipelineError("Audio conversion produced no file.")
    return wav_path, media_duration(wav_path)
