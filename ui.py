"""Dark-mode desktop interface for Voice2LLM.

The GUI is built with :mod:`customtkinter` and follows a strict rule: widgets
are only ever touched from the Tk main loop. All work happens in
:class:`transcriber.TranscriberWorker`, which posts plain dictionaries into a
:class:`queue.Queue`; :meth:`Voice2LLMApp._poll_messages` drains that queue with
``after()`` callbacks.
"""

from __future__ import annotations

import queue
import time
from pathlib import Path
from tkinter import font as tkfont
from tkinter import filedialog, messagebox
from typing import Any, Dict, Optional

import customtkinter as ctk
import pyperclip

from transcriber import (
    MSG_CANCELLED,
    MSG_CHUNK,
    MSG_DONE,
    MSG_ERROR,
    MSG_HEADER,
    MSG_PROGRESS,
    MSG_RESET,
    MSG_STATUS,
    MODEL_SIZES,
    SUPPORTED_AUDIO_EXTENSIONS,
    TranscriberWorker,
    format_duration,
    models_directory,
)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

COLOR_BACKGROUND = "#0B0B0D"
COLOR_PANEL = "#16161A"
COLOR_PANEL_INNER = "#101013"
COLOR_BORDER = "#26262C"
COLOR_TEXT = "#F2F2F7"
COLOR_MUTED = "#8E8E96"
COLOR_ACCENT = "#A855F7"
COLOR_ACCENT_HOVER = "#9333EA"
COLOR_ACCENT_SOFT = "#C4B5FD"
COLOR_DANGER = "#EF4444"
COLOR_DANGER_HOVER = "#B91C1C"
COLOR_SUCCESS = "#22C55E"
COLOR_BUTTON = "#232329"
COLOR_BUTTON_HOVER = "#2E2E37"
COLOR_TRACK = "#26262C"

# ---------------------------------------------------------------------------
# Layout / timing constants
# ---------------------------------------------------------------------------

WINDOW_WIDTH = 1120
WINDOW_HEIGHT = 790
MINIMUM_WIDTH = 940
MINIMUM_HEIGHT = 660

QUEUE_POLL_INTERVAL_MS = 80
TIMER_INTERVAL_MS = 250
COPY_FEEDBACK_DURATION_MS = 2000
CANCEL_CLOSE_TIMEOUT_SECONDS = 10.0

PLACEHOLDER_TEXT = (
    "Здесь появится транскрибация.\n\n"
    "1. Выберите аудиофайл (MP3, WAV, M4A).\n"
    "2. Выберите модель и язык.\n"
    "3. Нажмите «Начать транскрибацию».\n\n"
    "Готовый текст можно скопировать в ChatGPT или Claude одной кнопкой."
)

LANGUAGE_LABELS: Dict[str, str] = {
    "Автоопределение": "auto",
    "Русский (ru)": "ru",
    "English (en)": "en",
}

MODEL_HINTS: Dict[str, str] = {
    "base": "Самая быстрая · низкая точность",
    "small": "Быстрая · хороший баланс",
    "medium": "Точная · требует больше ресурсов",
    "large-v3-turbo": "Максимальная точность · нужен GPU",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pluralize(value: int, one: str, few: str, many: str) -> str:
    """Pick the correct Russian plural form for ``value``."""
    number = abs(int(value))
    if number % 10 == 1 and number % 100 != 11:
        return one
    if 2 <= number % 10 <= 4 and not 12 <= number % 100 <= 14:
        return few
    return many


def _format_size(size_bytes: int) -> str:
    """Format a file size in megabytes without locale surprises."""
    megabytes = size_bytes / (1024 * 1024)
    if megabytes >= 100:
        return f"{megabytes:.0f} МБ"
    if megabytes >= 10:
        return f"{megabytes:.1f} МБ"
    return f"{megabytes:.2f} МБ"


def _shorten_filename(name: str, limit: int = 58) -> str:
    """Shorten a long filename while keeping its extension visible."""
    if len(name) <= limit:
        return name
    path = Path(name)
    extension = path.suffix
    stem_budget = max(8, limit - len(extension) - 1)
    return f"{path.stem[:stem_budget]}…{extension}"


class Voice2LLMApp(ctk.CTk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__(fg_color=COLOR_BACKGROUND)

        self.title("Voice2LLM — транскрибация для LLM")
        self.minsize(MINIMUM_WIDTH, MINIMUM_HEIGHT)
        self.configure(fg_color=COLOR_BACKGROUND)

        self._audio_path: Optional[Path] = None
        self._worker: Optional[TranscriberWorker] = None
        self._messages: Optional[queue.Queue] = None
        self._running = False
        self._closing = False
        self._close_when_done = False
        self._close_deadline = 0.0

        self._run_started_at = 0.0
        self._last_elapsed = 0.0
        self._percent = 0.0
        self._eta: Optional[float] = None
        self._words = 0
        self._tokens = 0

        self._placeholder_active = False
        self._copy_reset_job: Optional[str] = None

        self._build_fonts()
        self._build_layout()
        self._install_placeholder()
        self._set_running(False)
        self._set_status("Готов к работе · выберите аудиофайл")
        self._refresh_metrics(0.0)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(QUEUE_POLL_INTERVAL_MS, self._poll_messages)
        self.after(TIMER_INTERVAL_MS, self._tick_timer)
        self._center_window()

    # -- construction -------------------------------------------------------

    def _pick_font_family(self, candidates: tuple[str, ...], fallback: str) -> str:
        """Return the first installed font family from ``candidates``."""
        try:
            installed = {family.lower() for family in tkfont.families(self)}
        except Exception:
            return fallback
        for candidate in candidates:
            if candidate.lower() in installed:
                return candidate
        return fallback

    def _build_fonts(self) -> None:
        ui_family = self._pick_font_family(
            ("Segoe UI", "Inter", "SF Pro Display", "Ubuntu", "DejaVu Sans"),
            "Arial",
        )
        mono_family = self._pick_font_family(
            (
                "Cascadia Mono",
                "Consolas",
                "JetBrains Mono",
                "SF Mono",
                "DejaVu Sans Mono",
                "Courier New",
            ),
            "Courier New",
        )

        self.font_title = ctk.CTkFont(family=ui_family, size=28, weight="bold")
        self.font_subtitle = ctk.CTkFont(family=ui_family, size=13)
        self.font_section = ctk.CTkFont(family=ui_family, size=12, weight="bold")
        self.font_body = ctk.CTkFont(family=ui_family, size=13)
        self.font_small = ctk.CTkFont(family=ui_family, size=12)
        self.font_button = ctk.CTkFont(family=ui_family, size=14, weight="bold")
        self.font_status = ctk.CTkFont(family=ui_family, size=12)
        self.font_metrics = ctk.CTkFont(family=ui_family, size=12)
        self.font_text = ctk.CTkFont(family=mono_family, size=14)
        self.font_chip = ctk.CTkFont(family=ui_family, size=11, weight="bold")

    def _build_layout(self) -> None:
        container = ctk.CTkFrame(self, fg_color=COLOR_BACKGROUND)
        container.pack(fill="both", expand=True, padx=22, pady=18)
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(3, weight=1)

        self._build_header(container)
        self._build_options_card(container)
        self._build_controls(container)
        self._build_transcript_card(container)
        self._build_footer(container)

    def _build_header(self, parent: ctk.CTkFrame) -> None:
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        header.grid_columnconfigure(0, weight=1)

        title_block = ctk.CTkFrame(header, fg_color="transparent")
        title_block.grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            title_block,
            text="Voice2LLM",
            font=self.font_title,
            text_color=COLOR_TEXT,
            anchor="w",
        ).pack(anchor="w")

        ctk.CTkLabel(
            title_block,
            text="Длинные аудио → чистый текст, готовый для ChatGPT и Claude",
            font=self.font_subtitle,
            text_color=COLOR_MUTED,
            anchor="w",
        ).pack(anchor="w", pady=(2, 0))

        chip = ctk.CTkLabel(
            header,
            text="100% ЛОКАЛЬНО · БЕЗ FFMPEG",
            font=self.font_chip,
            text_color=COLOR_ACCENT_SOFT,
            fg_color=COLOR_PANEL,
            corner_radius=8,
            width=230,
            height=30,
        )
        chip.grid(row=0, column=1, sticky="e")

    def _build_options_card(self, parent: ctk.CTkFrame) -> None:
        card = ctk.CTkFrame(parent, fg_color=COLOR_PANEL, corner_radius=16)
        card.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        card.grid_columnconfigure(1, weight=1)

        self.choose_button = ctk.CTkButton(
            card,
            text="Выбрать аудиофайл",
            width=176,
            height=38,
            corner_radius=10,
            fg_color=COLOR_BUTTON,
            hover_color=COLOR_BUTTON_HOVER,
            text_color=COLOR_TEXT,
            font=self.font_button,
            command=self._choose_file,
        )
        self.choose_button.grid(row=0, column=0, padx=(18, 14), pady=(16, 10), sticky="w")

        self.file_label = ctk.CTkLabel(
            card,
            text="Файл не выбран",
            font=self.font_button,
            text_color=COLOR_TEXT,
            anchor="w",
        )
        self.file_label.grid(row=0, column=1, sticky="ew", pady=(16, 10))

        self.file_meta_label = ctk.CTkLabel(
            card,
            text="MP3 · WAV · M4A",
            font=self.font_small,
            text_color=COLOR_MUTED,
            anchor="e",
        )
        self.file_meta_label.grid(row=0, column=2, padx=(12, 18), pady=(16, 10), sticky="e")

        settings = ctk.CTkFrame(card, fg_color="transparent")
        settings.grid(row=1, column=0, columnspan=3, sticky="ew", padx=18, pady=(4, 18))
        settings.grid_columnconfigure(3, weight=1)

        model_group = ctk.CTkFrame(settings, fg_color="transparent")
        model_group.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            model_group,
            text="МОДЕЛЬ",
            font=self.font_section,
            text_color=COLOR_MUTED,
            anchor="w",
        ).pack(anchor="w")
        self.model_menu = ctk.CTkOptionMenu(
            model_group,
            values=list(MODEL_SIZES),
            width=190,
            height=34,
            corner_radius=9,
            fg_color=COLOR_BUTTON,
            button_color="#2C2C33",
            button_hover_color=COLOR_BUTTON_HOVER,
            text_color=COLOR_TEXT,
            dropdown_fg_color=COLOR_PANEL,
            dropdown_hover_color=COLOR_ACCENT,
            dropdown_text_color=COLOR_TEXT,
            font=self.font_body,
            dropdown_font=self.font_body,
            command=self._on_model_change,
        )
        self.model_menu.set("small")
        self.model_menu.pack(anchor="w", pady=(6, 0))

        language_group = ctk.CTkFrame(settings, fg_color="transparent")
        language_group.grid(row=0, column=1, sticky="w", padx=(22, 0))
        ctk.CTkLabel(
            language_group,
            text="ЯЗЫК",
            font=self.font_section,
            text_color=COLOR_MUTED,
            anchor="w",
        ).pack(anchor="w")
        self.language_menu = ctk.CTkOptionMenu(
            language_group,
            values=list(LANGUAGE_LABELS.keys()),
            width=170,
            height=34,
            corner_radius=9,
            fg_color=COLOR_BUTTON,
            button_color="#2C2C33",
            button_hover_color=COLOR_BUTTON_HOVER,
            text_color=COLOR_TEXT,
            dropdown_fg_color=COLOR_PANEL,
            dropdown_hover_color=COLOR_ACCENT,
            dropdown_text_color=COLOR_TEXT,
            font=self.font_body,
            dropdown_font=self.font_body,
        )
        self.language_menu.set("Автоопределение")
        self.language_menu.pack(anchor="w", pady=(6, 0))

        self.llm_switch = ctk.CTkSwitch(
            settings,
            text="Режим для LLM",
            font=self.font_body,
            text_color=COLOR_TEXT,
            progress_color=COLOR_ACCENT,
            button_color=COLOR_TEXT,
            button_hover_color="#FFFFFF",
            width=48,
            command=self._on_llm_toggle,
        )
        self.llm_switch.grid(row=0, column=2, sticky="sw", padx=(30, 0), pady=(0, 6))
        self.llm_switch.select()

        self.prompt_checkbox = ctk.CTkCheckBox(
            settings,
            text="Добавить системный промпт",
            font=self.font_body,
            text_color=COLOR_TEXT,
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
            border_color="#3F3F48",
            border_width=2,
            checkbox_width=20,
            checkbox_height=20,
            corner_radius=5,
            checkmark_color=COLOR_BACKGROUND,
        )
        self.prompt_checkbox.grid(row=0, column=3, sticky="sw", padx=(24, 0), pady=(0, 6))

        self.model_hint_label = ctk.CTkLabel(
            settings,
            text=MODEL_HINTS["small"],
            font=self.font_small,
            text_color=COLOR_MUTED,
            anchor="e",
        )
        self.model_hint_label.grid(row=1, column=0, columnspan=4, sticky="e", pady=(10, 0))

    def _build_controls(self, parent: ctk.CTkFrame) -> None:
        controls = ctk.CTkFrame(parent, fg_color="transparent")
        controls.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        controls.grid_columnconfigure(1, weight=1)

        self.primary_button = ctk.CTkButton(
            controls,
            text="Начать транскрибацию",
            width=236,
            height=52,
            corner_radius=12,
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
            text_color="#FFFFFF",
            font=self.font_button,
            command=self._on_primary_click,
        )
        self.primary_button.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 18))

        progress_row = ctk.CTkFrame(controls, fg_color="transparent")
        progress_row.grid(row=0, column=1, sticky="ew")
        progress_row.grid_columnconfigure(0, weight=1)

        self.progress_bar = ctk.CTkProgressBar(
            progress_row,
            height=10,
            corner_radius=5,
            fg_color=COLOR_TRACK,
            progress_color=COLOR_ACCENT,
        )
        self.progress_bar.grid(row=0, column=0, sticky="ew", pady=(8, 0))
        self.progress_bar.set(0)

        self.percent_label = ctk.CTkLabel(
            progress_row,
            text="0%",
            width=52,
            font=self.font_metrics,
            text_color=COLOR_TEXT,
            anchor="e",
        )
        self.percent_label.grid(row=0, column=1, sticky="e", padx=(12, 0), pady=(8, 0))

        status_row = ctk.CTkFrame(controls, fg_color="transparent")
        status_row.grid(row=1, column=1, sticky="ew", pady=(6, 0))
        status_row.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            status_row,
            text="",
            font=self.font_status,
            text_color=COLOR_MUTED,
            anchor="w",
        )
        self.status_label.grid(row=0, column=0, sticky="w")

        self.metrics_label = ctk.CTkLabel(
            status_row,
            text="",
            font=self.font_metrics,
            text_color=COLOR_MUTED,
            anchor="e",
        )
        self.metrics_label.grid(row=0, column=1, sticky="e")

    def _build_transcript_card(self, parent: ctk.CTkFrame) -> None:
        card = ctk.CTkFrame(parent, fg_color=COLOR_PANEL, corner_radius=16)
        card.grid(row=3, column=0, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(card, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 6))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="ТРАНСКРИБАЦИЯ",
            font=self.font_section,
            text_color=COLOR_MUTED,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            header,
            text="Автопрокрутка при поступлении новых фрагментов",
            font=self.font_small,
            text_color=COLOR_MUTED,
            anchor="e",
        ).grid(row=0, column=1, sticky="e")

        self.textbox = ctk.CTkTextbox(
            card,
            fg_color=COLOR_PANEL_INNER,
            border_width=1,
            border_color=COLOR_BORDER,
            corner_radius=12,
            border_spacing=12,
            text_color=COLOR_TEXT,
            scrollbar_button_color="#2C2C33",
            scrollbar_button_hover_color=COLOR_ACCENT,
            font=self.font_text,
            wrap="word",
        )
        self.textbox.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))

    def _build_footer(self, parent: ctk.CTkFrame) -> None:
        footer = ctk.CTkFrame(parent, fg_color="transparent")
        footer.grid(row=4, column=0, sticky="ew", pady=(14, 0))
        footer.grid_columnconfigure(1, weight=1)

        self.copy_button = ctk.CTkButton(
            footer,
            text="Скопировать всё в буфер",
            width=232,
            height=42,
            corner_radius=11,
            fg_color=COLOR_BUTTON,
            hover_color=COLOR_BUTTON_HOVER,
            text_color=COLOR_TEXT,
            font=self.font_button,
            command=self._copy_all,
        )
        self.copy_button.grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            footer,
            text="Аудио не покидает ваш компьютер",
            font=self.font_small,
            text_color=COLOR_MUTED,
            anchor="center",
        ).grid(row=0, column=1, sticky="ew", padx=16)

        self.save_button = ctk.CTkButton(
            footer,
            text="Сохранить в .txt / .md",
            width=214,
            height=42,
            corner_radius=11,
            fg_color=COLOR_PANEL,
            hover_color=COLOR_BUTTON_HOVER,
            border_width=1,
            border_color=COLOR_ACCENT,
            text_color=COLOR_ACCENT_SOFT,
            font=self.font_button,
            command=self._save_text,
        )
        self.save_button.grid(row=0, column=2, sticky="e")

    def _center_window(self) -> None:
        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x_position = max(0, (screen_width - WINDOW_WIDTH) // 2)
        y_position = max(0, (screen_height - WINDOW_HEIGHT) // 2 - 20)
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x_position}+{y_position}")

    # -- placeholder & metrics ---------------------------------------------

    def _install_placeholder(self) -> None:
        self.textbox.delete("1.0", "end")
        self.textbox.insert("1.0", PLACEHOLDER_TEXT)
        self._placeholder_active = True
        text_widget = getattr(self.textbox, "_textbox", None)
        if text_widget is not None:
            try:
                text_widget.tag_configure("placeholder", foreground=COLOR_MUTED)
                text_widget.tag_add("placeholder", "1.0", "end")
            except Exception:
                # The palette text colour is still readable; only the subdued
                # tone is lost if the internal Text widget changes.
                self._placeholder_active = True

    def _clear_placeholder(self) -> None:
        if not self._placeholder_active:
            return
        self.textbox.delete("1.0", "end")
        self._placeholder_active = False

    def _refresh_metrics(self, elapsed: Optional[float] = None) -> None:
        if elapsed is None:
            if self._running:
                elapsed = max(0.0, time.monotonic() - self._run_started_at)
            else:
                elapsed = self._last_elapsed
        eta_text = (
            f"~{format_duration(self._eta)}"
            if isinstance(self._eta, (int, float)) and self._eta >= 1.0
            else "--:--"
        )
        words = _pluralize(self._words, "слово", "слова", "слов")
        tokens = _pluralize(self._tokens, "токен", "токена", "токенов")
        self.metrics_label.configure(
            text=(
                f"⏱ {format_duration(elapsed)} / {eta_text}"
                f"   ·   {self._words} {words}"
                f"   ·   ~{self._tokens} {tokens}"
            )
        )

    def _set_status(self, text: str, error: bool = False) -> None:
        single_line = " ".join(str(text).split())
        if len(single_line) > 120:
            single_line = single_line[:119].rstrip() + "…"
        self.status_label.configure(
            text=single_line,
            text_color=COLOR_DANGER if error else COLOR_MUTED,
        )

    # -- file & options -----------------------------------------------------

    def _choose_file(self) -> None:
        patterns = " ".join(f"*{extension}" for extension in SUPPORTED_AUDIO_EXTENSIONS)
        selected = filedialog.askopenfilename(
            title="Выберите аудиофайл",
            filetypes=[
                ("Аудио и видео", patterns),
                ("Все файлы", "*.*"),
            ],
        )
        if not selected:
            return

        candidate = Path(selected)
        try:
            size = candidate.stat().st_size
        except OSError as error:
            messagebox.showerror("Не удалось открыть файл", str(error))
            return

        if size <= 0:
            messagebox.showerror(
                "Пустой файл",
                "Выбранный файл имеет нулевой размер. Выберите другой файл.",
            )
            return

        self._audio_path = candidate
        self.file_label.configure(text=_shorten_filename(candidate.name))
        self.file_meta_label.configure(text=_format_size(size))
        self._set_status(
            f"Файл выбран · длительность будет определена при запуске · "
            f"модель {self.model_menu.get()}"
        )

    def _on_model_change(self, selected_model: str) -> None:
        hint = MODEL_HINTS.get(selected_model, "")
        self.model_hint_label.configure(text=hint)

    def _on_llm_toggle(self) -> None:
        llm_enabled = bool(self.llm_switch.get())
        if not llm_enabled:
            self.prompt_checkbox.deselect()
        self.prompt_checkbox.configure(
            state="normal" if llm_enabled and not self._running else "disabled"
        )

    # -- running state ------------------------------------------------------

    def _set_running(self, running: bool) -> None:
        self._running = running
        if running:
            self.primary_button.configure(
                text="Остановить",
                fg_color=COLOR_DANGER,
                hover_color=COLOR_DANGER_HOVER,
                state="normal",
            )
        else:
            self.primary_button.configure(
                text="Начать транскрибацию",
                fg_color=COLOR_ACCENT,
                hover_color=COLOR_ACCENT_HOVER,
                state="normal",
            )

        options_state = "disabled" if running else "normal"
        self.choose_button.configure(state=options_state)
        self.model_menu.configure(state=options_state)
        self.language_menu.configure(state=options_state)
        self.llm_switch.configure(state=options_state)

        prompt_state = (
            "normal" if bool(self.llm_switch.get()) and not running else "disabled"
        )
        self.prompt_checkbox.configure(state=prompt_state)

    # -- primary action -----------------------------------------------------

    def _on_primary_click(self) -> None:
        if self._running:
            self._request_cancel()
        else:
            self._start_transcription()

    def _start_transcription(self) -> None:
        if self._audio_path is None or not self._audio_path.is_file():
            messagebox.showwarning(
                "Файл не выбран",
                "Сначала выберите аудиофайл (MP3, WAV или M4A).",
            )
            return

        llm_mode = bool(self.llm_switch.get())
        include_prompt = bool(self.prompt_checkbox.get()) and llm_mode
        language = LANGUAGE_LABELS.get(self.language_menu.get(), "auto")
        model_size = self.model_menu.get()

        self._messages = queue.Queue()
        try:
            worker = TranscriberWorker(
                audio_path=self._audio_path,
                model_size=model_size,
                language=language,
                llm_mode=llm_mode,
                include_system_prompt=include_prompt,
                message_queue=self._messages,
                models_dir=models_directory(),
            )
        except Exception as error:
            messagebox.showerror("Не удалось запустить", str(error))
            return

        self._clear_transcript()
        self._run_started_at = time.monotonic()
        self._last_elapsed = 0.0
        self._percent = 0.0
        self._eta = None
        self._words = 0
        self._tokens = 0
        self.progress_bar.set(0)
        self.percent_label.configure(text="0%")
        self._refresh_metrics(0.0)
        self._set_status("Подготовка · загрузка модели…")
        self._set_running(True)
        self._worker = worker

        try:
            worker.start()
        except Exception as error:
            self._worker = None
            self._set_running(False)
            self._set_status("Не удалось запустить транскрибацию", error=True)
            messagebox.showerror("Не удалось запустить", str(error))

    def _clear_transcript(self) -> None:
        self.textbox.delete("1.0", "end")
        self._placeholder_active = False

    def _request_cancel(self) -> None:
        if self._worker is None:
            self._set_running(False)
            return
        self._worker.cancel()
        self.primary_button.configure(text="Останавливаю…", state="disabled")
        self._set_status("Останавливаю транскрибацию…")

    # -- queue handling -----------------------------------------------------

    def _poll_messages(self) -> None:
        try:
            if self._messages is not None:
                while True:
                    try:
                        message = self._messages.get_nowait()
                    except queue.Empty:
                        break
                    self._handle_message(message)
        finally:
            if not self._closing:
                self.after(QUEUE_POLL_INTERVAL_MS, self._poll_messages)

    def _handle_message(self, message: Dict[str, Any]) -> None:
        message_type = message.get("type")
        if message_type == MSG_STATUS:
            self._set_status(str(message.get("text", "")))
            return
        if message_type == MSG_HEADER:
            self._append_text(str(message.get("text") or ""))
            return
        if message_type == MSG_CHUNK:
            self._apply_metrics(message)
            self._append_text(str(message.get("text") or ""))
            return
        if message_type == MSG_PROGRESS:
            self._apply_metrics(message)
            return
        if message_type == MSG_RESET:
            self._reset_after_device_fallback()
            return
        if message_type == MSG_DONE:
            self._finish(message, cancelled=False)
            return
        if message_type == MSG_CANCELLED:
            self._finish(message, cancelled=True)
            return
        if message_type == MSG_ERROR:
            self._fail(message)

    def _apply_metrics(self, message: Dict[str, Any]) -> None:
        percent = message.get("percent")
        if isinstance(percent, (int, float)):
            self._percent = max(0.0, min(100.0, float(percent)))
            self.progress_bar.set(self._percent / 100.0)
            self.percent_label.configure(text=f"{self._percent:.0f}%")

        words = message.get("words")
        if isinstance(words, int):
            self._words = words
        tokens = message.get("tokens")
        if isinstance(tokens, int):
            self._tokens = tokens

        eta = message.get("eta")
        self._eta = float(eta) if isinstance(eta, (int, float)) else None
        elapsed = message.get("elapsed")
        if isinstance(elapsed, (int, float)):
            self._last_elapsed = float(elapsed)
        self._refresh_metrics()

    def _append_text(self, text: str) -> None:
        if not text:
            return
        self._clear_placeholder()
        self.textbox.insert("end", text)
        self._scroll_to_end()

    def _scroll_to_end(self) -> None:
        text_widget = getattr(self.textbox, "_textbox", None)
        if text_widget is not None:
            text_widget.see("end")

    def _reset_after_device_fallback(self) -> None:
        self._clear_transcript()
        self._percent = 0.0
        self._eta = None
        self._words = 0
        self._tokens = 0
        self._last_elapsed = 0.0
        self.progress_bar.set(0)
        self.percent_label.configure(text="0%")
        self._refresh_metrics(0.0)

    # -- completion ---------------------------------------------------------

    def _finish(self, message: Dict[str, Any], cancelled: bool) -> None:
        final_text = str(message.get("text") or "")
        if final_text:
            self.textbox.delete("1.0", "end")
            self.textbox.insert("1.0", final_text)
            self._placeholder_active = False

        words = message.get("words")
        if isinstance(words, int):
            self._words = words
        tokens = message.get("tokens")
        if isinstance(tokens, int):
            self._tokens = tokens
        elapsed = message.get("elapsed")
        if isinstance(elapsed, (int, float)):
            self._last_elapsed = float(elapsed)

        self._set_running(False)
        self._eta = None

        device = str(message.get("device") or "")
        device_label = "GPU (CUDA)" if device == "cuda" else "CPU"
        language = str(message.get("language") or "")
        language_text = f" · язык: {language}" if language else ""

        if cancelled:
            self._set_status(
                f"Остановлено · обработано {format_duration(self._last_elapsed)} · "
                f"{self._words} {_pluralize(self._words, 'слово', 'слова', 'слов')}"
            )
        else:
            self._percent = 100.0
            self.progress_bar.set(1.0)
            self.percent_label.configure(text="100%")
            self._set_status(
                f"Готово · {self._words} "
                f"{_pluralize(self._words, 'слово', 'слова', 'слов')} · "
                f"~{self._tokens} "
                f"{_pluralize(self._tokens, 'токен', 'токена', 'токенов')} · "
                f"{device_label}{language_text}"
            )
            self.bell()

        self._refresh_metrics(self._last_elapsed)
        self._worker = None

        if self._close_when_done:
            self._destroy_now()

    def _fail(self, message: Dict[str, Any]) -> None:
        self._set_running(False)
        self._worker = None
        self._eta = None
        self._refresh_metrics(self._last_elapsed)

        error_text = str(message.get("message") or "Неизвестная ошибка")
        self._set_status(f"Ошибка: {error_text.splitlines()[0]}", error=True)
        details = str(message.get("details") or "")
        if details:
            print(details)
        messagebox.showerror("Ошибка транскрибации", error_text)

        if self._close_when_done:
            self._destroy_now()

    # -- timer --------------------------------------------------------------

    def _tick_timer(self) -> None:
        try:
            if self._running:
                self._refresh_metrics(time.monotonic() - self._run_started_at)
            if self._close_when_done and (
                self._worker is None
                or not self._worker.is_running
                or time.monotonic() >= self._close_deadline
            ):
                self._destroy_now()
                return
        finally:
            if not self._closing:
                self.after(TIMER_INTERVAL_MS, self._tick_timer)

    # -- clipboard & saving -------------------------------------------------

    def _current_text(self) -> str:
        if self._placeholder_active:
            return ""
        return self.textbox.get("1.0", "end").strip()

    def _copy_all(self) -> None:
        text = self._current_text()
        if not text:
            self._set_status("Нечего копировать · текст ещё пуст")
            return

        try:
            pyperclip.copy(text)
        except Exception:
            try:
                self.clipboard_clear()
                self.clipboard_append(text)
                self.update_idletasks()
            except Exception as error:
                messagebox.showerror("Буфер обмена недоступен", str(error))
                return

        self.copy_button.configure(
            text="Скопировано!",
            fg_color=COLOR_SUCCESS,
            hover_color="#16A34A",
            text_color=COLOR_BACKGROUND,
        )
        if self._copy_reset_job is not None:
            self.after_cancel(self._copy_reset_job)
        self._copy_reset_job = self.after(
            COPY_FEEDBACK_DURATION_MS, self._reset_copy_button
        )
        self._set_status(
            f"Скопировано в буфер обмена · {len(text)} "
            f"{_pluralize(len(text), 'символ', 'символа', 'символов')}"
        )

    def _reset_copy_button(self) -> None:
        self._copy_reset_job = None
        self.copy_button.configure(
            text="Скопировать всё в буфер",
            fg_color=COLOR_BUTTON,
            hover_color=COLOR_BUTTON_HOVER,
            text_color=COLOR_TEXT,
        )

    def _save_text(self) -> None:
        text = self._current_text()
        if not text:
            messagebox.showwarning(
                "Нечего сохранять",
                "Сначала выполните транскрибацию — текст пока пуст.",
            )
            return

        default_name = "voice2llm_transcript.txt"
        if self._audio_path is not None:
            default_name = f"{self._audio_path.stem}_transcript.txt"

        target = filedialog.asksaveasfilename(
            title="Сохранить транскрибацию",
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[
                ("Текстовый файл", "*.txt"),
                ("Markdown", "*.md"),
                ("Все файлы", "*.*"),
            ],
        )
        if not target:
            return

        try:
            Path(target).write_text(text + "\n", encoding="utf-8")
        except OSError as error:
            messagebox.showerror("Не удалось сохранить файл", str(error))
            return

        self._set_status(f"Сохранено: {target}")

    # -- lifecycle ----------------------------------------------------------

    def _on_close(self) -> None:
        if self._closing:
            return

        worker_running = self._worker is not None and self._worker.is_running
        if not worker_running:
            self._destroy_now()
            return

        should_close = messagebox.askyesno(
            "Выход",
            "Транскрибация ещё выполняется.\n"
            "Остановить обработку и закрыть программу?",
        )
        if not should_close:
            return

        if self._worker is not None:
            self._worker.cancel()
        self._close_when_done = True
        self._close_deadline = time.monotonic() + CANCEL_CLOSE_TIMEOUT_SECONDS
        self._set_status("Останавливаю и закрываю…")
        self.primary_button.configure(state="disabled")

    def _destroy_now(self) -> None:
        self._closing = True
        self.destroy()
