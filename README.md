# Voice2LLM

**Turn 2-hour podcasts and meeting recordings into clean, token-efficient text for ChatGPT and Claude — locally, with a single click.**

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-BSD%203--Clause-3DA639">
  <img alt="Platforms" src="https://img.shields.io/badge/platforms-Windows%20%7C%20macOS%20%7C%20Linux-A855F7">
  <img alt="Engine" src="https://img.shields.io/badge/engine-faster--whisper%20%2B%20CTranslate2-FF6B6B">
  <img alt="No FFmpeg" src="https://img.shields.io/badge/external%20FFmpeg-not%20required-22C55E">
</p>

<p align="center">
  <a href="https://github.com/Mesa-afk/Voice2LLM/releases/latest"><img alt="Download" src="https://img.shields.io/badge/⬇%20Download-Windows%20binary-A855F7?style=for-the-badge"></a>
  &nbsp;
  <a href="https://github.com/Mesa-afk/Voice2LLM/actions/workflows/build-release.yml"><img alt="Build status" src="https://github.com/Mesa-afk/Voice2LLM/actions/workflows/build-release.yml/badge.svg"></a>
</p>

---

## The Problem

Whisper is an outstanding speech recognition model, but the *raw output* is a poor fit for a language model:

- **It is fragmented.** Whisper emits hundreds of short segments, often 2–6 seconds each. The result reads like a broken screenplay rather than a transcript, and every line break costs context.
- **It wastes your context window.** Timestamps such as `[00:42:13 --> 00:42:19]` are pure noise for ChatGPT or Claude. On a two-hour recording they can easily consume thousands of tokens before the model sees a single useful sentence.
- **It is a pain to install.** Most Whisper front-ends require a system-wide FFmpeg installation, a CUDA toolkit and Python package surgery. That is a non-starter for anyone who simply wants their podcast transcribed.
- **It falls over on long files.** Naive wrappers load a 200 MB file, freeze the UI, show no progress, and cannot be cancelled. If something goes wrong in hour two, everything is lost.

Voice2LLM fixes all of this: it decodes audio with the PyAV build that ships inside `faster-whisper` (no system FFmpeg), filters silence with Silero VAD, merges fragments into readable paragraphs, and streams the result into a dark, responsive desktop app.

## Features

- **100% local & private.** Audio never leaves your computer. No API keys, no accounts, no telemetry.
- **No FFmpeg required.** MP3, WAV, M4A, FLAC, OGG and more are decoded by the bundled PyAV libraries, both from source and from the packaged binary.
- **Silero VAD silence filtering.** Silence is skipped (`min_silence_duration_ms=500`), which removes hallucinated phrases and speeds up long recordings.
- **LLM-ready output.** Timecodes are stripped, phrases are merged into paragraphs, and an optional system prompt is prepended, so the text is ready to paste into ChatGPT or Claude.
- **Two output modes.**
  - **LLM Ready** — clean paragraphs, no timecodes, optional Russian system prompt:
    > Ниже представлена транскрибация аудиозаписи. Проанализируй текст, выдели ключевые тезисы, инсайты и список задач (Action Items):
  - **With timecodes** — `[HH:MM:SS -> HH:MM:SS] Text`, ideal for subtitles and citations.
- **Automatic GPU acceleration.** CUDA + `float16` is selected when a compatible GPU is found; otherwise the app falls back to CPU `int8`, and even retries on CPU if CUDA fails mid-run.
- **Live feedback.** Streaming segments, percentage, elapsed/remaining time, word count and an estimated token count (`words × 1.33`).
- **Cancellable and thread-safe.** Recognition runs in a dedicated `threading.Thread`; the GUI is updated only through a `queue.Queue`, and a `threading.Event` cancels gracefully while keeping partial results.
- **Modern dark UI.** `customtkinter` interface with a `#0B0B0D` canvas, `#16161A` panels, `#A855F7` accents, auto-scroll, one-click copy and `.txt` / `.md` export.
- **One-folder Windows build.** `--onedir` packaging means models load instantly from the `models/` folder next to the executable, without unpacking to a temporary directory.

## Quick Start (For Users)

### Windows — download and run

1. Open the **[latest release](https://github.com/Mesa-afk/Voice2LLM/releases/latest)**.
2. Download **`Voice2LLM-windows-x64.zip`**.
3. Extract the archive anywhere (for example `C:\Tools\Voice2LLM`).
4. Run **`Voice2LLM.exe`**.
5. Pick an audio file, choose a model, press **Начать транскрибацию**.

> **First launch:** the selected Whisper model is downloaded from Hugging Face into the `models/` folder next to the executable. `base` is a few hundred MB; `large-v3-turbo` is about 1.6 GB. Later launches start instantly and work fully offline.

> **Windows SmartScreen:** the binary is unsigned, so Windows may show “Windows protected your PC”. Choose *More info → Run anyway*.

### macOS / Linux

There are no prebuilt binaries yet — build from source with the instructions below. The application is pure Python and works wherever `customtkinter`, `faster-whisper` and Tk are available.

## Choosing a Model

| Model | Parameters | Disk | Quality | Suggested hardware |
| --- | --- | --- | --- | --- |
| `base` | 74 M | ~145 MB | Draft quality, good for clear speech | Any CPU, very fast |
| `small` | 244 M | ~485 MB | Good balance (default) | Modern CPU, ~10× faster on GPU |
| `medium` | 769 M | ~1.5 GB | High quality, strong on accents | GPU recommended (4 GB+ VRAM) |
| `large-v3-turbo` | 809 M | ~1.6 GB | Best quality per second | GPU strongly recommended (6 GB+ VRAM) |

A two-hour podcast on a mid-range GPU with `large-v3-turbo` typically takes a few minutes; the same job on CPU with `small` is considerably slower but still works.

## How It Works

```
audio.mp3 ──► PyAV decoder ──► 16 kHz mono PCM ──► Silero VAD ──► CTranslate2
                                                                    │
                                             streaming segments ◄───┘
                                                   │
                                    TranscriptPostProcessor
                                  (paragraphs / timecodes / prompt)
                                                   │
                                        queue.Queue ──► customtkinter UI
```

- **Decoding:** PyAV (bundled with `faster-whisper`) demuxes and resamples MP3/WAV/M4A without external tools.
- **VAD:** Silero VAD drops silence so the model does not hallucinate over pauses.
- **Inference:** CTranslate2 runs the Whisper model with `beam_size=5`. The worker keeps audio duration from the container header, so progress and ETA are available before the first segment arrives.
- **Post-processing:** paragraphs close when the gap between segments exceeds **2.0 seconds**, after **6–8 sentences**, or when a paragraph grows beyond 900 characters. Sentence-ending punctuation is normalised so the result is coherent prose.

### Example output

**LLM Ready**

```
Ниже представлена транскрибация аудиозаписи. Проанализируй текст, выдели ключевые тезисы, инсайты и список задач (Action Items):

Сегодня мы разберём, как небольшие команды запускают продукты быстрее крупных корпораций. Главный тезис прост: скорость обучения важнее скорости написания кода. Мы обсудили три привычки, которые меняют процесс разработки, и то, как они влияют на мотивацию инженеров.

...
```

**With timecodes**

```
[00:00:03 -> 00:00:09] Сегодня мы разберём, как небольшие команды запускают продукты быстрее крупных корпораций.
[00:00:09 -> 00:00:15] Главный тезис прост: скорость обучения важнее скорости написания кода.
```

## For Developers

### Requirements

- Python **3.10+**
- A working Tk installation (bundled with the official Windows and macOS installers)
- Optional: NVIDIA GPU with CUDA for hardware acceleration (the app works on CPU out of the box)

### Run from source

```bash
git clone https://github.com/Mesa-afk/Voice2LLM.git
cd Voice2LLM

python -m venv venv
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python main.py
```

Model weights are cached in `models/` next to `main.py` (git-ignored). If that directory is not writable, the app uses `~/.cache/voice2llm/models`.

### Build a standalone binary

```bash
python build.py           # dist/Voice2LLM/ (one folder)
python build.py --zip     # also creates a platform archive
python build.py --console # keep a console for debugging
```

The build script:

- validates Python and the installed dependencies;
- cleans previous `build/`, `dist/Voice2LLM` and `Voice2LLM.spec` artefacts;
- configures `--noconsole`, `--onedir`, `--noupx` and a Windows `VERSIONINFO` resource;
- bundles `customtkinter`, `faster_whisper`, `ctranslate2`, `tokenizers`, `onnxruntime`, `av` and `huggingface_hub` with `--collect-all`;
- excludes heavy unused packages such as `torch` and `pandas`;
- verifies that the executable exists and prints the output size.

Place `assets/icon.ico` (Windows), `assets/icon.icns` (macOS) or `assets/icon.png` (Linux) in the repository to brand the binary.

### Project structure

```
Voice2LLM/
├── main.py                     # entry point, Windows DPI / taskbar setup
├── ui.py                       # customtkinter dark-mode interface
├── transcriber.py              # worker thread, model loading, post-processing
├── build.py                    # PyInstaller build pipeline
├── requirements.txt            # pinned dependencies
├── .github/workflows/
│   └── build-release.yml       # Windows CI, archives and publishes releases
└── models/                     # downloaded weights (git-ignored)
```

### Releasing

Pushing a version tag triggers the Windows workflow, which builds the binary, zips `dist/Voice2LLM` and attaches the archive to a GitHub Release:

```bash
git tag v1.0.0
git push origin v1.0.0
```

The workflow can also be started manually (`workflow_dispatch`); manual runs keep the archive as a workflow artifact without publishing a release.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| “GPU-ускорение недоступно… Переключаюсь на CPU” | CUDA/cuDNN DLLs are missing. The app already fell back to CPU; install the NVIDIA runtime if you want GPU speed. |
| Very slow first transcription | The model is downloading. Watch the status line; subsequent runs load from `models/` instantly. |
| “Не удалось определить длительность аудиофайла” | The file is corrupt or is not an audio stream. Re-export it as MP3/WAV/M4A. |
| SmartScreen / antivirus warning | Expected for an unsigned PyInstaller binary. Verify the release checksum or build from source. |
| Copy button does nothing on Linux | Install a clipboard backend for `pyperclip` (`xclip`, `xsel` or `wl-clipboard`). |
| `ModuleNotFoundError: customtkinter` | Activate the virtual environment and run `pip install -r requirements.txt`. |

## Privacy

Voice2LLM performs every operation locally: no network calls except the one-time model download from Hugging Face, no analytics, no crash reporting. Your audio and transcripts stay on your disk.

## License

Released under the [BSD 3-Clause License](LICENSE).

Whisper weights are distributed by their respective publishers; `faster-whisper`, CTranslate2, PyAV, `customtkinter` and Silero VAD each carry their own licenses.
