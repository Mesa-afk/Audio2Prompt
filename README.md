# Audio2Prompt

**Turn 2-hour podcasts and meeting recordings into clean, token-efficient text for ChatGPT and Claude — locally, with a single click.**

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-BSD%203--Clause-3DA639">
  <img alt="Platforms" src="https://img.shields.io/badge/platforms-Windows%20%7C%20macOS%20%7C%20Linux-A855F7">
  <img alt="Engine" src="https://img.shields.io/badge/engine-faster--whisper%20%2B%20CTranslate2-FF6B6B">
  <img alt="No FFmpeg" src="https://img.shields.io/badge/external%20FFmpeg-not%20required-22C55E">
</p>

<p align="center">
  <a href="https://github.com/Mesa-afk/Audio2Prompt/releases/latest"><img alt="Download" src="https://img.shields.io/badge/⬇%20Download-Windows%20binary-A855F7?style=for-the-badge"></a>
  &nbsp;
  <a href="https://github.com/Mesa-afk/Audio2Prompt/actions/workflows/build-release.yml"><img alt="Build status" src="https://github.com/Mesa-afk/Audio2Prompt/actions/workflows/build-release.yml/badge.svg"></a>
</p>

<p align="center">
  <b>English</b> · <a href="README.ru.md">Русский</a>
</p>

---

## The Problem

Whisper is an outstanding speech recognition model, but the *raw output* is a poor fit for a language model:

- **It is fragmented.** Whisper emits hundreds of short segments, often 2–6 seconds each. The result reads like a broken screenplay rather than a transcript, and every line break costs context.
- **It wastes your context window.** Timestamps such as `[00:42:13 --> 00:42:19]` are pure noise for ChatGPT or Claude. On a two-hour recording they can easily consume thousands of tokens before the model sees a single useful sentence.
- **It is a pain to install.** Most Whisper front-ends require a system-wide FFmpeg installation, a CUDA toolkit and Python package surgery. That is a non-starter for anyone who simply wants their podcast transcribed.
- **It falls over on long files.** Naive wrappers load a 200 MB file, freeze the UI, show no progress, and cannot be cancelled. If something goes wrong in hour two, everything is lost.

Audio2Prompt fixes all of this: it decodes audio with the PyAV build that ships inside `faster-whisper` (no system FFmpeg), filters silence with Silero VAD, merges fragments into readable paragraphs, and streams the result into a dark, responsive desktop app.

## Features

- **100% local & private.** Audio never leaves your computer. No API keys, no accounts, no telemetry.
- **No FFmpeg required.** MP3, WAV, M4A, FLAC, OGG and more are decoded by the bundled PyAV libraries, both from source and from the packaged binary.
- **Silero VAD silence filtering.** Silence is skipped (`min_silence_duration_ms=500`), which removes hallucinated phrases and speeds up long recordings.
- **LLM-ready output.** Timecodes are stripped, phrases are merged into paragraphs, and an optional system prompt is prepended, so the text is ready to paste into ChatGPT or Claude.
- **Two output modes.**
  - **LLM Ready** — clean paragraphs, no timecodes, optional Russian system prompt:
    > Ниже представлена транскрибация аудиозаписи. Проанализируй текст, выдели ключевые тезисы, инсайты и список задач (Action Items):
  - **With timecodes** — `[HH:MM:SS -> HH:MM:SS] Text`, ideal for subtitles and citations.
- **Accuracy first.** The full `large-v3` model is available next to the fast `large-v3-turbo`, a dedicated **Accuracy mode** widens beam search to 10, uses unquantised `float32` CPU inference and adds a punctuation-rich prompt, and the **Context** field feeds names, brands and technical terms to Whisper as hotwords.
- **Automatic GPU acceleration.** CUDA + `float16` is selected when a compatible GPU is found; otherwise the app falls back to CPU (`int8`, or `float32` in accuracy mode), and even retries on CPU if CUDA fails mid-run. The status line always reports the device and compute type actually in use.
- **Live feedback.** Streaming segments, percentage, elapsed/remaining time, word count and an estimated token count (`words × 1.33`).
- **Cancellable and thread-safe.** Recognition runs in a dedicated `threading.Thread`; the GUI is updated only through a `queue.Queue`, and a `threading.Event` cancels gracefully while keeping partial results.
- **Minimal dark UI.** A compact single-screen `customtkinter` workspace with a `#0B0B0D` canvas, `#141417` panels and `#A855F7` accents. Only the core controls are visible; accuracy, context and the optional system prompt sit behind the **«Дополнительно»** toggle. The transcript auto-scrolls, and one click copies it or exports `.txt` / `.md`.
- **One-folder builds.** `--onedir` packaging means models load instantly from the `models/` folder next to the executable, without unpacking to a temporary directory.

## Quick Start (For Users)

| Platform | Ready-to-run binary | How to get it |
| --- | --- | --- |
| **Windows x64** | ✅ Yes | Download `Audio2Prompt-windows-x64.zip` from Releases |
| **Linux x64 / arm64** | ❌ No | Build locally — see **Linux** below |
| **macOS Intel / Apple Silicon** | ❌ No | Build locally — see **macOS** below |

### Windows — download and run

1. Open the **[latest release](https://github.com/Mesa-afk/Audio2Prompt/releases/latest)**.
2. Download **`Audio2Prompt-windows-x64.zip`**.
3. Extract the archive anywhere (for example `C:\Tools\Audio2Prompt`).
4. Run **`Audio2Prompt.exe`**.
5. Pick an audio file, choose a model, press **Начать**.

> **First launch:** the selected Whisper model is downloaded from Hugging Face into the `models/` folder next to the executable. `base` is a few hundred MB, `large-v3-turbo` is about 1.6 GB and `large-v3` is about 3.1 GB. Later launches start instantly and work fully offline.

> **For the most accurate transcript,** choose `large-v3`, keep **«Точный режим»** enabled and add names or terms to **«Контекст»** — see [Improving Accuracy](#improving-accuracy). This is slower, especially without a CUDA GPU.

> **Windows SmartScreen:** the binary is unsigned, so Windows may show “Windows protected your PC”. Choose *More info → Run anyway*.

### Linux — build and run from source

There is **no prebuilt Linux archive**: PyInstaller binaries are platform-specific, and a Linux build cannot be produced from Windows. You have to build it yourself on the Linux machine — it takes a few minutes.

#### 1. Install the system packages

`customtkinter` needs Tk, and `pyperclip` needs a clipboard utility:

```bash
# Debian / Ubuntu
sudo apt update
sudo apt install -y python3 python3-venv python3-pip python3-tk xclip

# Fedora
sudo dnf install -y python3 python3-tkinter xclip

# Arch
sudo pacman -S --needed python tk xclip
```

On Wayland install `wl-clipboard` (`sudo apt install wl-clipboard` on Debian/Ubuntu) instead of `xclip`.

#### 2. Clone the repository and install the Python dependencies

```bash
git clone https://github.com/Mesa-afk/Audio2Prompt.git
cd Audio2Prompt

python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Start the application

```bash
python main.py
```

#### 4. Build a portable Linux distribution (optional)

```bash
python build.py --zip
```

The result is a self-contained `dist/Audio2Prompt/` folder with the `Audio2Prompt` launcher and `dist/Audio2Prompt-linux-x64.zip` (the suffix is `linux-arm64` on ARM machines). Run the build **on the oldest distribution you plan to support**: the PyInstaller binary links against the build machine's glibc and will not start on older systems.

> **GPU (optional):** CUDA acceleration works with an NVIDIA GPU and a compatible CUDA/cuDNN runtime. Without it the app automatically uses CPU `int8` on all available cores.

### macOS — build and run from source

There is **no prebuilt macOS archive** yet. The application runs natively on Intel and Apple Silicon, but **CTranslate2 has no Metal/MPS backend on macOS**, so recognition is CPU-only — prefer `small` or `medium` for long recordings.

#### 1. Install Python 3.10+ with Tk

The python.org installer bundles Tk. With Homebrew, install the matching Tk formula explicitly (example for Python 3.12):

```bash
brew install python@3.12 python-tk@3.12
```

#### 2. Clone and install the Python dependencies

```bash
git clone https://github.com/Mesa-afk/Audio2Prompt.git
cd Audio2Prompt

python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python main.py
```

#### 3. Build a local one-folder distribution (optional)

```bash
python build.py --console
```

With the default `--noconsole` mode PyInstaller creates an `Audio2Prompt.app` bundle, while `build.py` verifies a plain one-folder layout; `--console` therefore produces the folder build at `dist/Audio2Prompt/`. A distributable `.app` should be signed and notarized; Gatekeeper warns about unsigned builds.

## Choosing a Model

| Model | Parameters | Disk | Quality | Suggested hardware |
| --- | --- | --- | --- | --- |
| `base` | 74 M | ~145 MB | Draft quality, good for clear speech | Any CPU, very fast |
| `small` | 244 M | ~485 MB | Good balance (default) | Modern CPU, ~10× faster on GPU |
| `medium` | 769 M | ~1.5 GB | High quality, strong on accents | GPU recommended (4 GB+ VRAM) |
| `large-v3-turbo` | 809 M | ~1.6 GB | Excellent quality at high speed | GPU strongly recommended (6 GB+ VRAM) |
| `large-v3` | 1.55 B | ~3.1 GB | **Maximum accuracy**, best Russian punctuation and rare words | NVIDIA GPU (10 GB+ VRAM); CPU works but is slow |

A two-hour podcast on a mid-range GPU with `large-v3-turbo` typically takes a few minutes; the same job on CPU with `small` is considerably slower but still works.

**Platform notes.** Windows and Linux can use an NVIDIA GPU through CUDA (`float16`); without a compatible GPU the app automatically selects CPU (`int8`, or `float32` in accuracy mode). macOS is CPU-only because CTranslate2 does not support Metal/MPS, so `large-v3-turbo` on a Mac is noticeably slower — `small` or `medium` is usually the better trade-off.

## Improving Accuracy

Whisper quality depends far more on the model and the audio than on any UI setting. If the transcript contains misheard words, missing phrases or broken punctuation, work through this list in order:

1. **Use `large-v3`.** It is the most accurate Whisper model for Russian and most other languages. `base` and `small` are meant for drafts, not for final transcripts.
2. **Keep “Точный режим” (Accuracy mode) on** (open **«Дополнительно»**). It raises `beam_size` from 5 to 10, uses `float32` instead of `int8` on CPU, and adds a punctuation-rich initial prompt. Expect CPU transcription to be noticeably slower (often 2–4×) — that is the intended speed/quality trade-off.
3. **Describe the recording in “Контекст”** (also under **«Дополнительно»**). Names, companies, product names, acronyms and domain terms are passed to Whisper as hotwords in every 30-second window. For example: `Иван Петров, Kubernetes, Audio2Prompt, выручка, CAC, LTV`.
4. **Select the language explicitly.** `ru` or `en` is more reliable than auto-detection, especially on recordings with music, silence or several speakers. Auto-detection samples the first three 30-second windows and cannot be corrected afterwards, so an explicit language is still safer.
5. **Start from the best source audio.** A 64 kbps re-encode loses consonants that no model can recover. Prefer the original WAV/M4A/MP3, remove background music where possible, and avoid recordings made with aggressive noise suppression.
6. **Check which device is active.** The status line shows `GPU · float16` or `CPU · float32`. A GPU does not change the model itself, but it makes `large-v3` fast enough to use; if the app reports CPU after saying “GPU недоступен”, the CUDA/cuDNN runtime is incomplete.

Accuracy mode is also used in the **With timecodes** output mode, so both formats benefit from the same decoder settings.

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
- **Inference:** CTranslate2 runs the Whisper model with `beam_size=5`, or `10` in accuracy mode; that mode also switches CPU inference to `float32` and supplies a punctuation prompt. The worker keeps audio duration from the container header, so progress and ETA are available before the first segment arrives.
- **Vocabulary:** names and terms from the context field are passed as Whisper hotwords, biasing every decoding window towards the expected spelling.
- **Post-processing:** paragraphs close when the gap between segments exceeds **2.0 seconds**, after **6–8 sentences**, or when a paragraph grows beyond **700 characters**. Sentence-ending punctuation is normalised and the first letter of every paragraph is capitalised so the result is coherent prose.

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
- A working Tk installation (bundled with the official Windows and macOS installers; on Linux install `python3-tk`)
- Optional: NVIDIA GPU with CUDA for hardware acceleration (the app works on CPU out of the box)

### Run from source

```bash
git clone https://github.com/Mesa-afk/Audio2Prompt.git
cd Audio2Prompt

python -m venv venv
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python main.py
```

Model weights are cached in `models/` next to `main.py` (git-ignored). If that directory is not writable, the app uses `~/.cache/audio2prompt/models`.

### Build a standalone binary

```bash
python build.py           # dist/Audio2Prompt/ (one folder)
python build.py --zip     # also creates a platform archive
python build.py --console # keep a console for debugging
```

The build script:

- validates Python and the installed dependencies;
- cleans previous `build/`, `dist/Audio2Prompt` and `Audio2Prompt.spec` artefacts;
- configures `--noconsole`, `--onedir`, `--noupx` and a Windows `VERSIONINFO` resource;
- bundles `customtkinter`, `faster_whisper`, `ctranslate2`, `tokenizers`, `onnxruntime`, `av` and `huggingface_hub` with `--collect-all`;
- excludes heavy unused packages such as `torch` and `pandas`;
- verifies that the executable exists and prints the output size.

> **Cross-platform note.** PyInstaller cannot cross-compile, so always build on the OS you want to target. Linux produces `dist/Audio2Prompt/Audio2Prompt`; Windows produces `dist/Audio2Prompt/Audio2Prompt.exe`; on macOS the default windowed mode creates an `Audio2Prompt.app` bundle, so use `python build.py --console` when you need a plain one-folder build. The GitHub Actions workflow currently builds Windows only.

Place `assets/icon.ico` (Windows), `assets/icon.icns` (macOS) or `assets/icon.png` (Linux) in the repository to brand the binary.

### Project structure

```
Audio2Prompt/
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

Pushing a version tag triggers the Windows workflow, which builds the binary, zips `dist/Audio2Prompt` and attaches the archive to a GitHub Release:

```bash
git tag v2.0.0
git push origin v2.0.0
```

The workflow is triggered only by version tags (`v*`), runs on `windows-latest`, and always publishes the Windows archive to a GitHub Release; the `contents: write` permission is what allows the file to be attached. It does not build Linux or macOS binaries — PyInstaller cannot cross-compile, so those must be built on the corresponding OS (see **Quick Start**).

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| “GPU недоступен… перезапуск на CPU” | CUDA/cuDNN DLLs are missing or the driver crashed. The app already restarted on CPU and reset the timer; install the NVIDIA runtime if you want GPU speed. |
| Very slow first transcription | The model is downloading. Watch the status line; subsequent runs load from `models/` instantly. |
| “Не удалось определить длительность аудиофайла” | The file is corrupt or is not an audio stream. Re-export it as MP3/WAV/M4A. |
| SmartScreen / antivirus warning | Expected for an unsigned PyInstaller binary. Verify the release checksum or build from source. |
| Copy button does nothing on Linux | Install a clipboard backend for `pyperclip` (`xclip`, `xsel` or `wl-clipboard`). |
| `ModuleNotFoundError: No module named 'tkinter'` | Install Tk: `sudo apt install python3-tk` (Debian/Ubuntu) or `sudo dnf install python3-tkinter` (Fedora). |
| `ModuleNotFoundError: customtkinter` | Activate the virtual environment and run `pip install -r requirements.txt`. |

## Privacy

Audio2Prompt performs every operation locally: no network calls except the one-time model download from Hugging Face, no analytics, no crash reporting. Your audio and transcripts stay on your disk.

## License

Released under the [BSD 3-Clause License](LICENSE).

Whisper weights are distributed by their respective publishers; `faster-whisper`, CTranslate2, PyAV, `customtkinter` and Silero VAD each carry their own licenses.
