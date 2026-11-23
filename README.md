# 🗣️ AI Talking Image Generator

Turn a single photo and a text script into a lip-synced talking video — from a simple Streamlit web app.

**Pipeline:** photo + script → text-to-speech → [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) lip-sync → ffmpeg rendering (portrait / landscape) → downloadable MP4.

▶️ [Watch the demo on YouTube](https://www.youtube.com/watch?v=rbVkRkYrMxs)

## Features

- Upload a face photo, type a script, pick a language, duration and orientation
- Voices in English, Hindi, Tamil, Spanish, French and German (Google TTS)
- Portrait (1080×1920) or landscape (1920×1080) output with a black or blurred background
- Output trimmed or padded to the chosen duration
- Setup check in the sidebar (ffmpeg, Wav2Lip, checkpoint)
- Clear error messages from every pipeline stage

## How it works

```
photo ──► prepare image (EXIF fix, RGB, ≤1024 px, even size)
script ─► gTTS (mp3) ─► ffmpeg ─► 16 kHz mono WAV
                          │
                          ▼
              Wav2Lip (wav2lip_gan.pth)  ──►  lipsync.mp4
                          │
                          ▼
         ffmpeg: fit to canvas, background, trim/pad  ──►  output/ai_video_<id>.mp4
```

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI and pipeline orchestration |
| `config.py` | Paths, subprocess helper, setup checks |
| `tts.py` | Text → speech (gTTS) → 16 kHz WAV |
| `lipsync.py` | Image preparation and Wav2Lip invocation |
| `video_utils.py` | Final render with ffmpeg (canvas, background, duration) |

## Requirements

- macOS / Linux / Windows with **Python 3.10**
- **ffmpeg** on your PATH
- Internet access (gTTS needs it)
- Runs on CPU. Expect a few minutes per 10-second clip on a laptop; a CUDA GPU is much faster.

## Setup (macOS)

```bash
# 1. system tools
brew install ffmpeg python@3.10 git

# 2. get the project and Wav2Lip
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
git clone https://github.com/Rudrabha/Wav2Lip.git

# 3. virtual environment (named venv310 so the app finds it automatically)
python3.10 -m venv venv310
source venv310/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Download the model files

**Lip-sync checkpoint** (`wav2lip_gan.pth`, ~146 MB):

```bash
pip install gdown
mkdir -p Wav2Lip/checkpoints
gdown 15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk -O Wav2Lip/checkpoints/wav2lip_gan.pth
```

> The official GAN checkpoint is a **TorchScript archive** of about 146 MB. If you have a ~436 MB `wav2lip.pth` instead, that is the *non-GAN* model, a different file.

**Face detector** (`s3fd.pth`):

```bash
curl -L -o Wav2Lip/face_detection/detection/sfd/s3fd.pth \
  https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth
```

The model weights are not stored in this repository. Check the [Wav2Lip README](https://github.com/Rudrabha/Wav2Lip) for their terms of use.

### Patch Wav2Lip for modern Python / PyTorch

The upstream Wav2Lip code predates PyTorch 2.6 and has a fragile ffmpeg call. Three small edits in `Wav2Lip/inference.py` are needed.

**1. Allow loading the checkpoint** (PyTorch ≥ 2.6 defaults to `weights_only=True`). In `_load`:

```python
def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path, weights_only=False)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage,
                                weights_only=False)
    return checkpoint
```

**2. Load the TorchScript GAN checkpoint.** Replace `load_model`:

```python
def load_model(path):
    print("Load checkpoint from: {}".format(path))
    try:
        # official wav2lip_gan.pth is a TorchScript archive
        model = torch.jit.load(path, map_location=device)
        return model.eval()
    except RuntimeError:
        # regular checkpoint (e.g. wav2lip.pth) with a state_dict
        model = Wav2Lip()
        checkpoint = _load(path)
        s = checkpoint["state_dict"]
        new_s = {k.replace('module.', ''): v for k, v in s.items()}
        model.load_state_dict(new_s)
        model = model.to(device)
        return model.eval()
```

**3. Pass ffmpeg arguments as a list instead of a shell string** (a space in the project path, e.g. `My Projects`, otherwise breaks the final audio merge, and the failure is silently ignored). Replace both `ffmpeg` calls with `subprocess.run([...], check=True)`:

```python
subprocess.run(['ffmpeg', '-y', '-i', args.audio, '-strict', '-2', 'temp/temp.wav'], check=True)
...
os.makedirs(os.path.dirname(os.path.abspath(args.outfile)), exist_ok=True)
subprocess.run(['ffmpeg', '-y', '-i', args.audio, '-i', 'temp/result.avi',
                '-strict', '-2', '-q:v', '1', args.outfile], check=True)
```

### Why these pinned versions?

| Pin | Reason |
|---|---|
| `numpy<2` | Several older dependencies break with NumPy 2 |
| `librosa==0.9.2` | Wav2Lip's `audio.py` calls librosa in a way that newer versions (0.10+) reject |
| `setuptools<70` | `librosa 0.9.2` imports `pkg_resources`, removed in newer setuptools. Pip may print a torch/setuptools conflict warning; it is harmless here |

## Run

```bash
source venv310/bin/activate
streamlit run app.py
```

Open http://localhost:8501. The sidebar should show three green checks. Upload a clear, front-facing photo, write a short script, and click **Generate Video**.

The finished video is saved to `output/ai_video_<run_id>.mp4` and can be downloaded from the page. Temporary files in `input/<run_id>/` are deleted after a successful run and kept if a run fails, so you can debug it.

## Limitations

- Needs a clear, roughly front-facing face; side profiles, occlusions, cartoons and multiple faces give poor or failed results
- Wav2Lip generates the mouth at low resolution, so the mouth can look softer than the rest of the face
- gTTS voices are robotic, and speech length is not controlled: long scripts are trimmed to the target duration
- Slow on CPU

## Responsible use

This tool can make a real person's photo appear to say anything. Only use photos of people who have given consent, never to impersonate or deceive, and label the results as AI-generated.

## Credits and licences

- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip), by Prajwal et al., *"A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild"* (ACM Multimedia 2020). Its pretrained models are for **research and personal use only**; check its terms before any commercial use.
- [gTTS](https://github.com/pndurette/gTTS), [Streamlit](https://streamlit.io), [FFmpeg](https://ffmpeg.org), [OpenCV](https://opencv.org)

The code in this repository is released under the [MIT licence](https://github.com/Divya-293/wave2lip/blob/main/LICENSE). This does not extend to Wav2Lip or its weights.
