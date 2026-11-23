import shutil
import uuid

import streamlit as st

from config import OUTPUT_DIR, WORK_DIR, PipelineError, ensure_dirs, preflight
from lipsync import prepare_face_image, run_wav2lip
from tts import generate_voice
from video_utils import process_video

ensure_dirs()
st.set_page_config(page_title="AI Talking Image Generator", page_icon="🗣️")
st.title("🗣️ AI Talking Image Generator")

MAX_CHARS = 1000
LANGS = {"English": "en", "Hindi": "hi", "Tamil": "ta", "Spanish": "es", "French": "fr", "German": "de"}

# ---------------- sidebar: environment check ----------------
with st.sidebar:
    st.subheader("Setup check")
    for ok, label in preflight():
        st.write(("✅ " if ok else "❌ ") + label)

# ---------------- inputs ----------------
image = st.file_uploader("Upload a clear, front-facing photo", type=["jpg", "jpeg", "png"])
if image:
    st.image(image, width=220)

text = st.text_area("Enter script", max_chars=MAX_CHARS, height=140)
col1, col2 = st.columns(2)
language = col1.selectbox("Voice language", list(LANGS))
orientation = col2.radio("Orientation", ["Portrait", "Landscape"], horizontal=True)

duration = st.slider("Target duration (seconds)", 8, 15, 10)
col3, col4 = st.columns(2)
background = col3.selectbox("Background", ["Blurred", "Black"])
fit_duration = col4.checkbox("Pad short clips to exact duration", value=True)

# rough speech-rate hint (~15 characters per second)
if text.strip() and len(text) / 15 > duration:
    st.info(f"Your script is ~{len(text) / 15:.0f}s of speech; it will be trimmed to {duration}s. "
            "Shorten it or raise the duration to avoid cutting words off.")

# ---------------- pipeline ----------------
if st.button("Generate Video", type="primary"):
    if image is None or not text.strip():
        st.warning("Upload an image and enter a script first.")
    elif not all(ok for ok, _ in preflight()):
        st.error("Fix the ❌ items in the sidebar first.")
    else:
        st.session_state.pop("result", None)
        run_id = uuid.uuid4().hex[:8]
        run_dir = WORK_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            with st.status("Generating video...", expanded=True) as status:
                st.write("🖼️ Preparing image...")
                face = prepare_face_image(image.getvalue(), run_dir / "face.jpg")

                st.write("🔊 Generating voice...")
                audio, audio_len = generate_voice(text.strip(), run_dir, lang=LANGS[language])
                if audio_len:
                    st.write(f"   Audio length: {audio_len:.1f}s")

                st.write("👄 Running lip sync (this can take a few minutes)...")
                video = run_wav2lip(face, audio, run_dir)

                st.write("🎬 Rendering final video...")
                final = process_video(
                    video, OUTPUT_DIR / f"ai_video_{run_id}.mp4",
                    duration, orientation, background, fit_duration,
                )
                status.update(label="Done!", state="complete", expanded=False)

            st.session_state["result"] = str(final)
            shutil.rmtree(run_dir, ignore_errors=True)   # keep temp files only when something fails

        except PipelineError as e:
            st.error(str(e))
        except Exception as e:  # noqa: BLE001
            st.error(f"Unexpected error: {e}")

# Result lives in session_state so it survives the rerun triggered by the download button
if "result" in st.session_state:
    st.success("Your video is ready")
    st.video(st.session_state["result"])
    with open(st.session_state["result"], "rb") as f:
        st.download_button("Download MP4", f, file_name="ai_video.mp4", mime="video/mp4")
