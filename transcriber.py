"""Transcription engine for Voice2LLM.

This module contains the complete audio -> text pipeline used by the desktop
application:

* :class:`TranscriptPostProcessor` turns raw Whisper segments into either
  LLM-optimised prose (no timecodes, coherent paragraphs) or a timestamped
  transcript.
* :class:`TranscriberWorker` runs all heavy work (model download/loading and
  streaming inference) inside a dedicated :class:`threading.Thread` and reports
  progress to the GUI exclusively through a :class:`queue.Queue`.
* Cancellation is cooperative and driven by :class:`threading.Event`.
* Audio decoding is handled by PyAV (bundled with faster-whisper), therefore no
  system FFmpeg installation is required.

The worker never touches Tkinter widgets. Every message it produces is a plain
``dict`` with a ``"type"`` key; the GUI drains the queue on the Tk main loop.
"""

from __future__ import annotations

import gc
import os
import queue
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Whisper model sizes exposed in the GUI. ``large-v3-turbo`` is the fast
#: large model (809M parameters, ~8x faster than ``large-v3``).
MODEL_SIZES: Tuple[str, ...] = ("base", "small", "medium", "large-v3-turbo")

#: ``auto`` lets faster-whisper detect the spoken language.
SUPPORTED_LANGUAGES: Tuple[str, ...] = ("auto", "ru", "en")

#: Optional first line prepended to the transcript in "LLM Ready" mode.
SYSTEM_PROMPT: str = (
    "Ниже представлена транскрибация аудиозаписи. Проанализируй текст, "
    "выдели ключевые тезисы, инсайты и список задач (Action Items):"
)

#: A silence between two segments longer than this starts a new paragraph.
PAUSE_THRESHOLD_SECONDS: float = 2.0

#: Paragraphs are closed after 6, 7 or 8 sentences (cycling), so long
#: monologues never collapse into a single unreadable wall of text.
MIN_SENTENCES_PER_PARAGRAPH: int = 6
MAX_SENTENCES_PER_PARAGRAPH: int = 8

#: Safety valve for speeches that contain almost no punctuation at all.
MAX_PARAGRAPH_CHARACTERS: int = 900

#: Rough token/word ratio for English, Russian and most European languages.
TOKENS_PER_WORD: float = 1.33

#: Inference parameters.
BEAM_SIZE: int = 5
VAD_MIN_SILENCE_DURATION_MS: int = 500

#: Queue message types consumed by ``ui.py``.
MSG_STATUS = "status"
MSG_HEADER = "header"
MSG_CHUNK = "chunk"
MSG_PROGRESS = "progress"
MSG_RESET = "reset"
MSG_DONE = "done"
MSG_CANCELLED = "cancelled"
MSG_ERROR = "error"

# ---------------------------------------------------------------------------
# Compiled regular expressions
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([,.;:!?…])")
_SENTENCE_END_RE = re.compile(r"[.!?…]+(?=\s|$)")
_TIMECODE_RE = re.compile(
    r"\[\s*\d{1,3}:\d{2}(?::\d{2})?(?:[.,]\d+)?\s*"
    r"(?:-->|->|–|—)\s*"
    r"\d{1,3}:\d{2}(?::\d{2})?(?:[.,]\d+)?\s*\]"
)

# Exception text fragments that indicate a broken CUDA/cuDNN installation.
# faster-whisper ships CUDA kernels, but cuBLAS/cuDNN DLLs must exist on the
# user's machine; when they are missing we transparently fall back to CPU.
_CUDA_ERROR_MARKERS: Tuple[str, ...] = (
    "cuda",
    "cudnn",
    "cublas",
    "cudart",
    "gpu",
    "out of memory",
    "no kernel image",
    "dll load failed",
    "libcublas",
    "libcudnn",
)

#: Audio containers that PyAV can demux. The GUI opens a file dialog with the
#: same list, but the engine does not reject anything else.
SUPPORTED_AUDIO_EXTENSIONS: Tuple[str, ...] = (
    ".mp3",
    ".wav",
    ".m4a",
    ".mp4",
    ".aac",
    ".flac",
    ".ogg",
    ".oga",
    ".opus",
    ".wma",
    ".mka",
    ".mkv",
    ".webm",
    ".mov",
)


class _Cancelled(Exception):
    """Internal control-flow signal raised when the user cancels a job."""


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------


def format_timestamp(seconds: float) -> str:
    """Return ``seconds`` as ``HH:MM:SS`` (hours are not wrapped at 24h).

    Timestamps are floored, not rounded, so a segment that starts at 2.5 s is
    reported as ``00:00:02`` — the moment the speech actually begins.
    """
    total_seconds = max(0, int(float(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_duration(seconds: Optional[float]) -> str:
    """Return ``seconds`` as ``MM:SS`` or ``H:MM:SS`` for human display."""
    if seconds is None:
        return "--:--"
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def count_words(text: str) -> int:
    """Count whitespace-separated words in ``text``."""
    if not text:
        return 0
    stripped = text.strip()
    if not stripped:
        return 0
    return len(stripped.split())


def estimate_tokens(words: int) -> int:
    """Estimate LLM tokens from a word count (``words * 1.33``)."""
    if words <= 0:
        return 0
    return int(round(words * TOKENS_PER_WORD))


def normalize_segment_text(text: str) -> str:
    """Collapse whitespace and remove spaces that precede punctuation."""
    if not text:
        return ""
    normalized = _WHITESPACE_RE.sub(" ", text).strip()
    normalized = _SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", normalized)
    return normalized


def strip_timecodes(text: str) -> str:
    """Remove ``[00:01:02 -> 00:01:05]`` markers from ``text``.

    Whisper segments do not normally contain timecodes, but some upstream
    pipelines (and copy-pasted subtitles) do. They are pure token overhead for
    an LLM, so "LLM Ready" mode drops them and tidies the leftover spacing.
    """
    if not text or "[" not in text:
        return text
    without_timecodes = _TIMECODE_RE.sub(" ", text)
    return normalize_segment_text(without_timecodes)


def count_sentences(text: str) -> int:
    """Count sentence-ending punctuation groups in ``text``."""
    if not text:
        return 0
    return len(_SENTENCE_END_RE.findall(text))


def cuda_is_available() -> bool:
    """Return ``True`` when a CUDA device usable by CTranslate2 is present."""
    device_count = 0
    try:
        import ctranslate2  # type: ignore[import-not-found]

        device_count = int(ctranslate2.get_cuda_device_count())
    except Exception:
        # CTranslate2 is absent or cannot enumerate devices (driver issues).
        device_count = 0

    if device_count > 0:
        return True

    # Optional secondary probe: some environments expose CUDA through PyTorch
    # while CTranslate2 needs extra environment variables to see it.
    try:
        import torch  # type: ignore[import-not-found]

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def models_directory() -> Path:
    """Return the directory where Whisper weights are cached.

    Next to the executable/script when possible, otherwise in the user's cache
    (an installation inside ``Program Files`` is often read-only).
    """
    if getattr(sys, "frozen", False):
        base_directory = Path(sys.executable).resolve().parent
    else:
        base_directory = Path(__file__).resolve().parent

    preferred = base_directory / "models"
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    except OSError:
        fallback = Path.home() / ".cache" / "voice2llm" / "models"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def probe_duration(audio_path: Path) -> float:
    """Return the media duration in seconds, or ``0.0`` when unknown.

    PyAV reads the container header, which is essentially instant even for
    multi-hour podcasts. If the header lacks a duration, the full stream is
    decoded as a last resort so progress reporting still works.
    """
    header_duration = 0.0
    try:
        import av  # type: ignore[import-not-found]

        with av.open(str(audio_path)) as container:
            if container.duration is not None and container.duration > 0:
                # PyAV reports container duration in microseconds.
                header_duration = float(container.duration) / 1_000_000.0
            if header_duration <= 0:
                for stream in container.streams.audio:
                    if stream.duration is not None and stream.duration > 0:
                        time_base = stream.time_base or 0
                        stream_duration = float(stream.duration * time_base)
                        if stream_duration > 0:
                            header_duration = stream_duration
                            break
    except Exception:
        header_duration = 0.0

    if header_duration > 0:
        return header_duration

    try:
        from faster_whisper.audio import decode_audio  # type: ignore[import-not-found]

        samples = decode_audio(str(audio_path), sampling_rate=16000)
        return float(len(samples)) / 16000.0
    except Exception:
        return 0.0


def _cpu_thread_count() -> int:
    """Use all logical cores for CTranslate2 CPU inference (at least 1)."""
    return max(1, os.cpu_count() or 4)


def _truncate(text: str, limit: int = 220) -> str:
    """Shorten long error messages so they fit into a status line."""
    single_line = _WHITESPACE_RE.sub(" ", str(text)).strip()
    if len(single_line) <= limit:
        return single_line
    return single_line[: limit - 1].rstrip() + "…"


def _is_cuda_error(error: BaseException) -> bool:
    """Heuristically detect CUDA/cuDNN failures raised during inference."""
    message = f"{type(error).__name__}: {error}".lower()
    return any(marker in message for marker in _CUDA_ERROR_MARKERS)


def _friendly_error(error: BaseException) -> str:
    """Convert an exception into an actionable message for the end user."""
    if isinstance(error, ImportError):
        return (
            "Не удалось загрузить компоненты распознавания. "
            "Установите зависимости командой: pip install -r requirements.txt"
        )
    if isinstance(error, FileNotFoundError):
        return "Аудиофайл не найден. Возможно, он был перемещён или удалён."
    if isinstance(error, PermissionError):
        return "Нет доступа к файлу. Закройте программу, которая его использует, и повторите попытку."
    if isinstance(error, OSError) and "space" in str(error).lower():
        return "На диске недостаточно места для загрузки весов модели."
    return (
        f"{type(error).__name__}: {_truncate(error, 400)}\n\n"
        "Проверьте, что файл не повреждён и что модель поддерживается, затем повторите попытку."
    )


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


class TranscriptPostProcessor:
    """Convert streaming Whisper segments into LLM-friendly text.

    Two modes are supported:

    ``llm_mode=True``
        Timecodes are dropped and consecutive segments are merged into
        paragraphs. A paragraph is closed when the silence before the next
        segment exceeds :data:`PAUSE_THRESHOLD_SECONDS`, when 6-8 sentences
        have accumulated, or when the paragraph grows beyond
        :data:`MAX_PARAGRAPH_CHARACTERS` characters.

    ``llm_mode=False``
        Every segment becomes ``[HH:MM:SS -> HH:MM:SS] Text``.

    The processor is also a streaming renderer: :meth:`push` returns the text
    that should be appended to the GUI as soon as a paragraph (or a timestamped
    line) is complete, and :attr:`rendered_text` contains everything emitted so
    far. :meth:`finalize` flushes the last paragraph.
    """

    def __init__(self, llm_mode: bool = True, include_system_prompt: bool = False) -> None:
        self.llm_mode = bool(llm_mode)
        self.include_system_prompt = bool(include_system_prompt) and self.llm_mode
        self._buffer: List[str] = []
        self._rendered: List[str] = []
        self._sentence_count = 0
        self._paragraph_index = 0
        self._words = 0
        self._last_end: Optional[float] = None
        self._header_rendered = False
        self._finalized = False

    # -- introspection ------------------------------------------------------

    @property
    def word_count(self) -> int:
        """Number of transcript words seen so far (system prompt excluded)."""
        return self._words

    @property
    def rendered_text(self) -> str:
        """Complete transcript emitted so far, trailing blank lines removed."""
        return "".join(self._rendered).strip()

    @property
    def has_system_prompt(self) -> bool:
        """``True`` when the system prompt will be part of the transcript."""
        return self.include_system_prompt

    # -- streaming API ------------------------------------------------------

    def render_header(self) -> Optional[str]:
        """Return the system-prompt block once, or ``None`` when disabled."""
        if not self.include_system_prompt or self._header_rendered:
            return None
        self._header_rendered = True
        header = f"{SYSTEM_PROMPT}\n\n"
        self._rendered.append(header)
        return header

    def push(self, text: str, start: float, end: float) -> Optional[str]:
        """Feed one Whisper segment and return newly completed display text."""
        cleaned = normalize_segment_text(text)
        if not cleaned:
            return None

        if self.llm_mode:
            cleaned = strip_timecodes(cleaned)
            if not cleaned:
                return None

        self._words += count_words(cleaned)
        start_seconds = max(0.0, float(start))
        end_seconds = max(start_seconds, float(end))

        if not self.llm_mode:
            line = (
                f"[{format_timestamp(start_seconds)} -> "
                f"{format_timestamp(end_seconds)}] {cleaned}\n"
            )
            self._rendered.append(line)
            return line

        chunks: List[str] = []
        if (
            self._buffer
            and self._last_end is not None
            and start_seconds - self._last_end >= PAUSE_THRESHOLD_SECONDS
        ):
            paragraph = self._flush_paragraph()
            if paragraph:
                chunks.append(paragraph)

        self._buffer.append(cleaned)
        self._sentence_count += count_sentences(cleaned)
        self._last_end = end_seconds

        buffer_characters = sum(len(piece) for piece in self._buffer) + max(
            0, len(self._buffer) - 1
        )
        if (
            self._sentence_count >= self._sentence_target()
            or buffer_characters >= MAX_PARAGRAPH_CHARACTERS
        ):
            paragraph = self._flush_paragraph()
            if paragraph:
                chunks.append(paragraph)

        if not chunks:
            return None
        return "".join(chunks)

    def finalize(self) -> str:
        """Flush the trailing paragraph and return it (empty when nothing left)."""
        if self._finalized:
            return ""
        self._finalized = True
        if not self.llm_mode:
            return ""
        return self._flush_paragraph()

    # -- internals ----------------------------------------------------------

    def _sentence_target(self) -> int:
        """Cycle the paragraph length across 6, 7 and 8 sentences."""
        span = MAX_SENTENCES_PER_PARAGRAPH - MIN_SENTENCES_PER_PARAGRAPH + 1
        return MIN_SENTENCES_PER_PARAGRAPH + (self._paragraph_index % span)

    def _flush_paragraph(self) -> str:
        if not self._buffer:
            self._sentence_count = 0
            return ""

        raw_paragraph = normalize_segment_text(" ".join(self._buffer))
        self._buffer.clear()
        self._sentence_count = 0

        if not raw_paragraph:
            return ""

        if raw_paragraph[-1] not in ".!?…":
            raw_paragraph = raw_paragraph.rstrip(",;:—-– ") + "."

        self._paragraph_index += 1
        chunk = f"{raw_paragraph}\n\n"
        self._rendered.append(chunk)
        return chunk


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class TranscriberWorker:
    """Load a Whisper model and transcribe a file in a background thread.

    Parameters
    ----------
    audio_path:
        Path to the audio file (MP3, WAV, M4A, FLAC, OGG and other containers
        supported by PyAV).
    model_size:
        One of :data:`MODEL_SIZES`.
    language:
        ``"ru"``, ``"en"`` or ``"auto"``.
    llm_mode:
        Enable paragraph merging and drop timecodes.
    include_system_prompt:
        Prepend :data:`SYSTEM_PROMPT` (only meaningful with ``llm_mode``).
    message_queue:
        Optional queue to use; a fresh :class:`queue.Queue` is created when not
        provided. Poll it with :meth:`queue.Queue.get_nowait` from the GUI.
    models_dir:
        Optional override for the model cache directory.

    Typical usage::

        worker = TranscriberWorker("podcast.mp3", "small", "ru")
        worker.start()
        while worker.is_running or not worker.message_queue.empty():
            event = worker.message_queue.get()
            ...
    """

    def __init__(
        self,
        audio_path: str | os.PathLike[str],
        model_size: str = "small",
        language: str = "auto",
        llm_mode: bool = True,
        include_system_prompt: bool = False,
        message_queue: Optional[queue.Queue] = None,
        models_dir: Optional[str | os.PathLike[str]] = None,
    ) -> None:
        resolved_path = Path(audio_path).expanduser().resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Аудиофайл не найден: {resolved_path}")
        if model_size not in MODEL_SIZES:
            raise ValueError(
                f"Неизвестная модель «{model_size}». "
                f"Допустимые значения: {', '.join(MODEL_SIZES)}"
            )
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Неподдерживаемый язык «{language}». "
                f"Допустимые значения: {', '.join(SUPPORTED_LANGUAGES)}"
            )

        self.audio_path = resolved_path
        self.model_size = model_size
        self.language = language
        self.llm_mode = bool(llm_mode)
        self.include_system_prompt = bool(include_system_prompt) and self.llm_mode
        self.message_queue: queue.Queue = (
            message_queue if message_queue is not None else queue.Queue()
        )
        self.models_dir = (
            Path(models_dir).expanduser().resolve()
            if models_dir is not None
            else models_directory()
        )
        self.cancel_event = threading.Event()

        self._thread: Optional[threading.Thread] = None
        self._model: Any = None
        self._device = "cpu"
        self._compute_type = "int8"

    # -- public API ---------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """``True`` while the background thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def device(self) -> str:
        """Device used by the active (or most recent) model."""
        return self._device

    @property
    def compute_type(self) -> str:
        """CTranslate2 compute type of the active (or most recent) model."""
        return self._compute_type

    def start(self) -> threading.Thread:
        """Launch the worker thread and return it."""
        if self.is_running:
            raise RuntimeError("Транскрибация уже выполняется.")
        self.cancel_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"voice2llm-{self.audio_path.name}",
            daemon=True,
        )
        self._thread.start()
        return self._thread

    def cancel(self) -> None:
        """Request cooperative cancellation.

        The worker stops at the next segment boundary; partial results are sent
        as a :data:`MSG_CANCELLED` message.
        """
        self.cancel_event.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for the background thread to finish; ``True`` when it stopped."""
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    # -- queue plumbing -----------------------------------------------------

    def _emit(self, payload: Dict[str, Any]) -> None:
        """Put a message on the queue without blocking the worker thread."""
        self.message_queue.put(payload)

    def _emit_status(self, text: str) -> None:
        self._emit({"type": MSG_STATUS, "text": text})

    # -- thread body --------------------------------------------------------

    def _run(self) -> None:
        started = time.monotonic()
        processor = TranscriptPostProcessor(self.llm_mode, self.include_system_prompt)
        language_label = "автоопределение" if self.language == "auto" else self.language

        try:
            duration = probe_duration(self.audio_path)
            if duration <= 0:
                raise RuntimeError(
                    "Не удалось определить длительность аудиофайла. "
                    "Возможно, файл повреждён или это не аудио."
                )

            self._emit_status(
                f"Загрузка модели «{self.model_size}»… "
                "Первый запуск может занять несколько минут: скачиваются веса."
            )
            gpu_available = cuda_is_available()
            model, device, compute_type = self._load_model(prefer_cuda=gpu_available)
            self._model = model
            self._device = device
            self._compute_type = compute_type

            if self.cancel_event.is_set():
                raise _Cancelled()

            self._emit_status(
                f"Модель загружена · {self._device_label(device, compute_type)} · "
                f"язык: {language_label}"
            )

            header = processor.render_header()
            if header is not None:
                self._emit({"type": MSG_HEADER, "text": header})

            try:
                self._stream(
                    model=model,
                    duration=duration,
                    processor=processor,
                    started=started,
                    device=device,
                    compute_type=compute_type,
                )
            except _Cancelled:
                raise
            except Exception as inference_error:
                if (
                    device == "cuda"
                    and not self.cancel_event.is_set()
                    and _is_cuda_error(inference_error)
                ):
                    self._emit_status(
                        "GPU-ускорение недоступно во время распознавания "
                        f"({_truncate(inference_error, 140)}). "
                        "Переключаюсь на CPU (int8)…"
                    )
                    self._release_model()
                    processor = TranscriptPostProcessor(
                        self.llm_mode, self.include_system_prompt
                    )
                    self._emit({"type": MSG_RESET})
                    model, device, compute_type = self._load_model(prefer_cuda=False)
                    self._model = model
                    self._device = device
                    self._compute_type = compute_type
                    started = time.monotonic()
                    header = processor.render_header()
                    if header is not None:
                        self._emit({"type": MSG_HEADER, "text": header})
                    self._emit_status(
                        f"Распознавание на CPU · {self._device_label(device, compute_type)}"
                    )
                    self._stream(
                        model=model,
                        duration=duration,
                        processor=processor,
                        started=started,
                        device=device,
                        compute_type=compute_type,
                    )
                else:
                    raise

        except _Cancelled:
            words = processor.word_count
            self._emit(
                {
                    "type": MSG_CANCELLED,
                    "text": processor.rendered_text,
                    "words": words,
                    "tokens": estimate_tokens(words),
                    "elapsed": time.monotonic() - started,
                    "device": self._device,
                    "compute_type": self._compute_type,
                }
            )
        except Exception as error:
            self._emit(
                {
                    "type": MSG_ERROR,
                    "message": _friendly_error(error),
                    "details": traceback.format_exc(),
                }
            )
        finally:
            self._release_model()

    # -- inference ----------------------------------------------------------

    def _stream(
        self,
        model: Any,
        duration: float,
        processor: TranscriptPostProcessor,
        started: float,
        device: str,
        compute_type: str,
    ) -> None:
        """Consume the faster-whisper segment generator and emit queue events."""
        segments, info = model.transcribe(
            str(self.audio_path),
            language=None if self.language == "auto" else self.language,
            task="transcribe",
            beam_size=BEAM_SIZE,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": VAD_MIN_SILENCE_DURATION_MS},
            word_timestamps=False,
            # Disabled on purpose: conditioning on previous text is the main
            # cause of repetition loops and hallucinated phrases on long files.
            condition_on_previous_text=False,
        )

        info_duration = float(getattr(info, "duration", 0.0) or 0.0)
        effective_duration = info_duration if info_duration > 0 else duration
        detected_language = getattr(info, "language", None)
        language_probability = getattr(info, "language_probability", None)

        if detected_language:
            probability_text = (
                f" · уверенность {language_probability * 100:.0f}%"
                if isinstance(language_probability, (int, float))
                else ""
            )
            self._emit_status(
                f"Распознавание · {self._device_label(device, compute_type)} · "
                f"язык: {detected_language}{probability_text}"
            )

        self._emit(
            {
                "type": MSG_PROGRESS,
                "percent": 0.0,
                "elapsed": time.monotonic() - started,
                "eta": None,
                "words": processor.word_count,
                "tokens": estimate_tokens(processor.word_count),
            }
        )

        last_progress_emit = 0.0
        segment_count = 0

        for segment in segments:
            if self.cancel_event.is_set():
                self._close_generator(segments)
                raise _Cancelled()

            segment_text = getattr(segment, "text", "") or ""
            segment_start = float(getattr(segment, "start", 0.0) or 0.0)
            segment_end = float(getattr(segment, "end", segment_start) or segment_start)
            chunk = processor.push(segment_text, segment_start, segment_end)
            segment_count += 1

            percent, eta = self._progress_snapshot(
                segment_end, effective_duration, started
            )
            words = processor.word_count
            tokens = estimate_tokens(words)
            elapsed = time.monotonic() - started

            if chunk:
                self._emit(
                    {
                        "type": MSG_CHUNK,
                        "text": chunk,
                        "words": words,
                        "tokens": tokens,
                        "percent": percent,
                        "elapsed": elapsed,
                        "eta": eta,
                        "start": segment_start,
                        "end": segment_end,
                    }
                )

            now = time.monotonic()
            if now - last_progress_emit >= 0.2:
                self._emit(
                    {
                        "type": MSG_PROGRESS,
                        "percent": percent,
                        "elapsed": elapsed,
                        "eta": eta,
                        "words": words,
                        "tokens": tokens,
                    }
                )
                last_progress_emit = now

            if self.cancel_event.is_set():
                self._close_generator(segments)
                raise _Cancelled()

        if self.cancel_event.is_set():
            self._close_generator(segments)
            raise _Cancelled()

        tail = processor.finalize()
        if tail:
            self._emit(
                {
                    "type": MSG_CHUNK,
                    "text": tail,
                    "words": processor.word_count,
                    "tokens": estimate_tokens(processor.word_count),
                    "percent": 100.0,
                    "elapsed": time.monotonic() - started,
                    "eta": 0.0,
                }
            )

        elapsed = time.monotonic() - started
        words = processor.word_count
        self._emit(
            {
                "type": MSG_PROGRESS,
                "percent": 100.0,
                "elapsed": elapsed,
                "eta": 0.0,
                "words": words,
                "tokens": estimate_tokens(words),
            }
        )
        self._emit(
            {
                "type": MSG_DONE,
                "text": processor.rendered_text,
                "words": words,
                "tokens": estimate_tokens(words),
                "elapsed": elapsed,
                "segments": segment_count,
                "device": device,
                "compute_type": compute_type,
                "language": detected_language or self.language,
            }
        )
        self._close_generator(segments)

    @staticmethod
    def _close_generator(generator: Any) -> None:
        """Release the underlying decoder/transcription generator promptly."""
        close_method = getattr(generator, "close", None)
        if not callable(close_method):
            return
        try:
            close_method()
        except Exception:
            # Generators can already be exhausted or mid-decode; a garbage
            # collection run is the safest way to release native buffers.
            gc.collect()

    @staticmethod
    def _progress_snapshot(
        position: float, duration: float, started: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """Return ``(percent, eta_seconds)`` for the current stream position."""
        elapsed = max(0.0, time.monotonic() - started)
        if duration <= 0:
            return None, None
        fraction = min(max(position / duration, 0.0), 1.0)
        percent = min(fraction * 100.0, 99.5)
        if fraction < 0.02:
            return percent, None
        eta = max(0.0, elapsed / fraction - elapsed)
        return percent, eta

    # -- model management ---------------------------------------------------

    def _load_model(self, prefer_cuda: bool) -> Tuple[Any, str, str]:
        """Load the first working ``(model, device, compute_type)`` combination."""
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]

        candidates: List[Tuple[str, str]] = []
        if prefer_cuda:
            candidates.append(("cuda", "float16"))
            candidates.append(("cuda", "int8_float16"))
        candidates.append(("cpu", "int8"))
        candidates.append(("cpu", "float32"))

        failures: List[str] = []
        for device, compute_type in candidates:
            try:
                model = WhisperModel(
                    self.model_size,
                    device=device,
                    compute_type=compute_type,
                    download_root=str(self.models_dir),
                    cpu_threads=_cpu_thread_count(),
                    num_workers=1,
                )
                return model, device, compute_type
            except Exception as error:
                failures.append(f"{device}/{compute_type}: {_truncate(error)}")
                self._emit_status(
                    f"Вариант {device} ({compute_type}) недоступен · "
                    "пробую следующий…"
                )

        details = "\n".join(f"• {failure}" for failure in failures)
        raise RuntimeError(
            "Не удалось загрузить модель распознавания. "
            "Проверьте подключение к интернету (первый запуск скачивает веса) "
            "и свободное место на диске.\n\n"
            f"{details}"
        )

    def _release_model(self) -> None:
        """Drop the model so its memory is returned to the OS."""
        self._model = None
        gc.collect()

    @staticmethod
    def _device_label(device: str, compute_type: str) -> str:
        if device == "cuda":
            return f"GPU · CUDA · {compute_type}"
        return f"CPU · {compute_type}"
