"""Minimal dark-mode desktop interface for Voice2LLM.

The GUI is built with :mod:`customtkinter` and follows a strict rule: widgets
are only ever touched from the Tk main loop. All work happens in
:class:`transcriber.TranscriberWorker`, which posts plain dictionaries into a
:class:`queue.Queue`; :meth:`Voice2LLMApp._poll_messages` drains that queue with
``after()`` callbacks.

The window is intentionally reduced to the core workflow: choose a file, pick a
model and language, start transcription, then copy or save the result. Accuracy
controls and the optional system prompt live in a collapsible "Дополнительно"
panel so they never clutter the main screen.
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
COLOR_PANEL = "#141417"
COLOR_PANEL_INNER = "#0F0F12"
COLOR_BORDER = "#25252B"
COLOR_TEXT = "#EDEDF2"
COLOR_MUTED = "#85858E"
COLOR_ACCENT = "#A855F7"
COLOR_ACCENT_HOVER = "#9333EA"
COLOR_DANGER = "#E5484D"
COLOR_DANGER_HOVER = "#B93A3F"
COLOR_SUCCESS = "#34C759"
COLOR_BUTTON = "#1D1D22"
COLOR_BUTTON_HOVER = "#28282F"
COLOR_TRACK = "#222228"

# ---------------------------------------------------------------------------
# Layout / timing constants
# ---------------------------------------------------------------------------

WINDOW_WIDTH = 960
WINDOW_HEIGHT = 700
MINIMUM_WIDTH = 820
MINIMUM_HEIGHT = 600

QUEUE_POLL_INTERVAL_MS = 80
TIMER_INTERVAL_MS = 250
COPY_FEEDBACK_DURATION_MS = 2000
CANCEL_CLOSE_TIMEOUT_SECONDS = 10.0

PLACEHOLDER_TEXT = "Выберите аудиофайл."

LANGUAGE_LABELS: Dict[str, str] = {
    "Автоопределение": "auto",
    "Русский (ru)": "ru",
    "English (en)": "en",
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

        self.title("Voice2LLM")
        self.minsize(MINIMUM_WIDTH, MINIMUM_HEIGHT)
        self.configure(fg_color=COLOR_BACKGROUND)

        self._audio_path: Optional[Path] = None
        self._worker: Optional[TranscriberWorker] = None
        self._messages: Optional[queue.Queue] = None
        self._running = False
        self._closing = False
        self._close_when_done = False
        self._close_deadline = 0.0
        self._advanced_visible = False

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
        self._set_status("Готов")
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

        self.font_title = ctk.CTkFont(family=ui_family, size=22, weight="bold")
        self.font_body = ctk.CTkFont(family=ui_family, size=13)
        self.font_small = ctk.CTkFont(family=ui_family, size=11)
        self.font_button = ctk.CTkFont(family=ui_family, size=13, weight="bold")
        self.font_status = ctk.CTkFont(family=ui_family, size=12)
        self.font_metrics = ctk.CTkFont(family=mono_family, size=12)
        self.font_text = ctk.CTkFont(family=ui_family, size=14)

    def _build_layout(self) -> None:
        container = ctk.CTkFrame(self, fg_color=COLOR_BACKGROUND)
        container.pack(fill="both", expand=True, padx=24, pady=20)
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(4, weight=1)

        self._build_header(container)
        self._build_main_card(container)
        self._build_advanced_panel(container)
        self._build_controls(container)
        self._build_transcript(container)
        self._build_footer(container)

    def _build_header(self, parent: ctk.CTkFrame) -> None:
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="Voice2LLM",
            font=self.font_title,
            text_color=COLOR_TEXT,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

    def _build_main_card(self, parent: ctk.CTkFrame) -> None:
        card = ctk.CTkFrame(parent, fg_color=COLOR_PANEL, corner_radius=14)
        card.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        card.grid_columnconfigure(1, weight=1)

        self.choose_button = ctk.CTkButton(
            card,
            text="Выбрать файл",
            width=132,
            height=34,
            corner_radius=9,
            fg_color=COLOR_BUTTON,
            hover_color=COLOR_BUTTON_HOVER,
            text_color=COLOR_TEXT,
            font=self.font_body,
            command=self._choose_file,
        )
        self.choose_button.grid(
            row=0, column=0, padx=(16, 12), pady=(14, 8), sticky="w"
        )

        self.file_label = ctk.CTkLabel(
            card,
            text="Файл не выбран",
            font=self.font_body,
            text_color=COLOR_TEXT,
            anchor="w",
        )
        self.file_label.grid(row=0, column=1, sticky="ew", pady=(14, 8))

        self.file_meta_label = ctk.CTkLabel(
            card,
            text="MP3 · WAV · M4A",
            font=self.font_small,
            text_color=COLOR_MUTED,
            anchor="e",
        )
        self.file_meta_label.grid(
            row=0, column=2, padx=(12, 16), pady=(14, 8), sticky="e"
        )

        options = ctk.CTkFrame(card, fg_color="transparent")
        options.grid(row=1, column=0, columnspan=3, sticky="ew", padx=16, pady=(0, 14))
        options.grid_columnconfigure(2, weight=1)

        model_group = ctk.CTkFrame(options, fg_color="transparent")
        model_group.grid(row=0, column=0, sticky="w")
        model_group.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            model_group,
            text="Модель",
            font=self.font_small,
            text_color=COLOR_MUTED,
            anchor="w",
        ).grid(row=0, column=0, padx=(0, 8))

        self.model_menu = ctk.CTkOptionMenu(
            model_group,
            values=list(MODEL_SIZES),
            width=150,
            height=32,
            corner_radius=8,
            fg_color=COLOR_BUTTON,
            button_color=COLOR_BUTTON,
            button_hover_color=COLOR_BUTTON_HOVER,
            text_color=COLOR_TEXT,
            dropdown_fg_color=COLOR_PANEL,
            dropdown_hover_color=COLOR_ACCENT,
            dropdown_text_color=COLOR_TEXT,
            font=self.font_body,
            dropdown_font=self.font_body,
        )
        self.model_menu.set("small")
        self.model_menu.grid(row=0, column=1, sticky="w")

        language_group = ctk.CTkFrame(options, fg_color="transparent")
        language_group.grid(row=0, column=1, sticky="w", padx=(22, 0))
        language_group.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            language_group,
            text="Язык",
            font=self.font_small,
            text_color=COLOR_MUTED,
            anchor="w",
        ).grid(row=0, column=0, padx=(0, 8))

        self.language_menu = ctk.CTkOptionMenu(
            language_group,
            values=list(LANGUAGE_LABELS.keys()),
            width=150,
            height=32,
            corner_radius=8,
            fg_color=COLOR_BUTTON,
            button_color=COLOR_BUTTON,
            button_hover_color=COLOR_BUTTON_HOVER,
            text_color=COLOR_TEXT,
            dropdown_fg_color=COLOR_PANEL,
            dropdown_hover_color=COLOR_ACCENT,
            dropdown_text_color=COLOR_TEXT,
            font=self.font_body,
            dropdown_font=self.font_body,
        )
        self.language_menu.set("Автоопределение")
        self.language_menu.grid(row=0, column=1, sticky="w")

        self.llm_switch = ctk.CTkSwitch(
            options,
            text="LLM-режим",
            font=self.font_body,
            text_color=COLOR_TEXT,
            progress_color=COLOR_ACCENT,
            button_color=COLOR_TEXT,
            button_hover_color="#FFFFFF",
            width=42,
            command=self._on_llm_toggle,
        )
        self.llm_switch.grid(row=0, column=3, padx=(16, 14), sticky="e")
        self.llm_switch.select()

        self.advanced_button = ctk.CTkButton(
            options,
            text="Дополнительно",
            width=118,
            height=28,
            corner_radius=7,
            fg_color="transparent",
            hover_color=COLOR_BUTTON,
            text_color=COLOR_MUTED,
            font=self.font_small,
            command=self._toggle_advanced,
        )
        self.advanced_button.grid(row=0, column=4, sticky="e")

    def _build_advanced_panel(self, parent: ctk.CTkFrame) -> None:
        panel = ctk.CTkFrame(parent, fg_color=COLOR_PANEL, corner_radius=12)
        panel.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        panel.grid_columnconfigure(1, weight=1)

        self.accuracy_switch = ctk.CTkSwitch(
            panel,
            text="Точный режим",
            font=self.font_body,
            text_color=COLOR_TEXT,
            progress_color=COLOR_ACCENT,
            button_color=COLOR_TEXT,
            button_hover_color="#FFFFFF",
            width=42,
        )
        self.accuracy_switch.grid(
            row=0, column=0, padx=(16, 12), pady=(12, 0), sticky="w"
        )
        self.accuracy_switch.select()

        self.prompt_checkbox = ctk.CTkCheckBox(
            panel,
            text="Системный промпт",
            font=self.font_body,
            text_color=COLOR_TEXT,
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
            border_color=COLOR_BORDER,
            border_width=2,
            checkbox_width=18,
            checkbox_height=18,
            corner_radius=4,
            checkmark_color=COLOR_BACKGROUND,
        )
        self.prompt_checkbox.grid(
            row=0, column=1, padx=(12, 16), pady=(12, 0), sticky="e"
        )

        self.context_entry = ctk.CTkEntry(
            panel,
            placeholder_text="Имена, термины, контекст",
            height=34,
            corner_radius=8,
            border_width=1,
            border_color=COLOR_BORDER,
            fg_color=COLOR_PANEL_INNER,
            text_color=COLOR_TEXT,
            placeholder_text_color=COLOR_MUTED,
            font=self.font_body,
        )
        self.context_entry.grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=16, pady=(10, 12)
        )

        self.advanced_panel = panel
        panel.grid_remove()

    def _build_controls(self, parent: ctk.CTkFrame) -> None:
        controls = ctk.CTkFrame(parent, fg_color="transparent")
        controls.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        controls.grid_columnconfigure(1, weight=1)

        self.primary_button = ctk.CTkButton(
            controls,
            text="Начать",
            width=150,
            height=42,
            corner_radius=10,
            fg_color=COLOR_ACCENT,
            hover_color=COLOR_ACCENT_HOVER,
            text_color="#FFFFFF",
            font=self.font_button,
            command=self._on_primary_click,
        )
        self.primary_button.grid(
            row=0, column=0, rowspan=2, sticky="w", padx=(0, 16)
        )

        self.progress_bar = ctk.CTkProgressBar(
            controls,
            height=8,
            corner_radius=4,
            fg_color=COLOR_TRACK,
            progress_color=COLOR_ACCENT,
        )
        self.progress_bar.grid(row=0, column=1, sticky="ew", pady=(6, 0))
        self.progress_bar.set(0)

        self.percent_label = ctk.CTkLabel(
            controls,
            text="0%",
            width=44,
            font=self.font_metrics,
            text_color=COLOR_TEXT,
            anchor="e",
        )
        self.percent_label.grid(
            row=0, column=2, sticky="e", padx=(10, 0), pady=(6, 0)
        )

        info_row = ctk.CTkFrame(controls, fg_color="transparent")
        info_row.grid(
            row=1, column=1, columnspan=2, sticky="ew", pady=(7, 0)
        )
        info_row.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            info_row,
            text="",
            font=self.font_status,
            text_color=COLOR_MUTED,
            anchor="w",
        )
        self.status_label.grid(row=0, column=0, sticky="w")

        self.metrics_label = ctk.CTkLabel(
            info_row,
            text="",
            font=self.font_metrics,
            text_color=COLOR_MUTED,
            anchor="e",
        )
        self.metrics_label.grid(row=0, column=1, sticky="e", padx=(10, 0))

    def _build_transcript(self, parent: ctk.CTkFrame) -> None:
        self.textbox = ctk.CTkTextbox(
            parent,
            fg_color=COLOR_PANEL_INNER,
            border_width=1,
            border_color=COLOR_BORDER,
            corner_radius=12,
            border_spacing=14,
            text_color=COLOR_TEXT,
            scrollbar_button_color=COLOR_BUTTON,
            scrollbar_button_hover_color=COLOR_ACCENT,
            font=self.font_text,
            wrap="word",
        )
        self.textbox.grid(row=4, column=0, sticky="nsew")

    def _build_footer(self, parent: ctk.CTkFrame) -> None:
        footer = ctk.CTkFrame(parent, fg_color="transparent")
        footer.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        footer.grid_columnconfigure(0, weight=1)

        self.copy_button = ctk.CTkButton(
            footer,
            text="Копировать",
            width=140,
            height=36,
            corner_radius=9,
            fg_color=COLOR_BUTTON,
            hover_color=COLOR_BUTTON_HOVER,
            text_color=COLOR_TEXT,
            font=self.font_body,
            command=self._copy_all,
        )
        self.copy_button.grid(row=0, column=1, sticky="e", padx=(0, 8))

        self.save_button = ctk.CTkButton(
            footer,
            text="Сохранить",
            width=130,
            height=36,
            corner_radius=9,
            fg_color=COLOR_BUTTON,
            hover_color=COLOR_BUTTON_HOVER,
            text_color=COLOR_TEXT,
            font=self.font_body,
            command=self._save_text,
        )
        self.save_button.grid(row=0, column=2, sticky="e")

    def _toggle_advanced(self) -> None:
        self._advanced_visible = not self._advanced_visible
        if self._advanced_visible:
            self.advanced_panel.grid()
            self.advanced_button.configure(text="Свернуть")
        else:
            self.advanced_panel.grid_remove()
            self.advanced_button.configure(text="Дополнительно")

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
        self.metrics_label.configure(
            text=(
                f"{format_duration(elapsed)} / {eta_text}"
                f" · {self._words} {words}"
                f" · ~{self._tokens} ток."
            )
        )

    def _set_status(self, text: str, error: bool = False) -> None:
        single_line = " ".join(str(text).split())
        if len(single_line) > 52:
            single_line = single_line[:51].rstrip() + "…"
        self.status_label.configure(
            text=single_line,
            text_color=COLOR_DANGER if error else COLOR_MUTED,
        )

    # -- file & options -----------------------------------------------------

    def _choose_file(self) -> None:
        patterns = " ".join(f"*{extension}" for extension in SUPPORTED_AUDIO_EXTENSIONS)
        selected = filedialog.askopenfilename(
            title="Аудиофайл",
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
                "Выбранный файл имеет нулевой размер.",
            )
            return

        self._audio_path = candidate
        self.file_label.configure(text=_shorten_filename(candidate.name))
        self.file_meta_label.configure(text=_format_size(size))
        self._set_status("Готов")

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
                text="Начать",
                fg_color=COLOR_ACCENT,
                hover_color=COLOR_ACCENT_HOVER,
                state="normal",
            )

        options_state = "disabled" if running else "normal"
        self.choose_button.configure(state=options_state)
        self.model_menu.configure(state=options_state)
        self.language_menu.configure(state=options_state)
        self.llm_switch.configure(state=options_state)
        self.accuracy_switch.configure(state=options_state)
        self.context_entry.configure(state=options_state)
        self.advanced_button.configure(state=options_state)

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
                "Сначала выберите аудиофайл.",
            )
            return

        llm_mode = bool(self.llm_switch.get())
        include_prompt = bool(self.prompt_checkbox.get()) and llm_mode
        language = LANGUAGE_LABELS.get(self.language_menu.get(), "auto")
        model_size = self.model_menu.get()
        accuracy_mode = bool(self.accuracy_switch.get())
        hotwords = self.context_entry.get().strip() or None

        self._messages = queue.Queue()
        try:
            worker = TranscriberWorker(
                audio_path=self._audio_path,
                model_size=model_size,
                language=language,
                llm_mode=llm_mode,
                include_system_prompt=include_prompt,
                accuracy_mode=accuracy_mode,
                hotwords=hotwords,
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
        self._set_status("Загрузка модели…")
        self._set_running(True)
        self._worker = worker

        try:
            worker.start()
        except Exception as error:
            self._worker = None
            self._set_running(False)
            self._set_status("Не удалось запустить", error=True)
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
        self._set_status("Останавливаю…")

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
        self._run_started_at = time.monotonic()
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
        compute_type = str(message.get("compute_type") or "")
        device_label = "GPU" if device == "cuda" else "CPU"
        if compute_type:
            device_label = f"{device_label} · {compute_type}"

        if cancelled:
            self._set_status(
                f"Остановлено · {self._words} "
                f"{_pluralize(self._words, 'слово', 'слова', 'слов')}"
            )
        else:
            self._percent = 100.0
            self.progress_bar.set(1.0)
            self.percent_label.configure(text="100%")
            self._set_status(
                f"Готово · {self._words} "
                f"{_pluralize(self._words, 'слово', 'слова', 'слов')} · "
                f"{device_label}"
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
            self._set_status("Текст пуст")
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
            text="Скопировано",
            fg_color=COLOR_SUCCESS,
            hover_color=COLOR_SUCCESS,
            text_color=COLOR_BACKGROUND,
        )
        if self._copy_reset_job is not None:
            self.after_cancel(self._copy_reset_job)
        self._copy_reset_job = self.after(
            COPY_FEEDBACK_DURATION_MS, self._reset_copy_button
        )
        self._set_status("Скопировано")

    def _reset_copy_button(self) -> None:
        self._copy_reset_job = None
        self.copy_button.configure(
            text="Копировать",
            fg_color=COLOR_BUTTON,
            hover_color=COLOR_BUTTON_HOVER,
            text_color=COLOR_TEXT,
        )

    def _save_text(self) -> None:
        text = self._current_text()
        if not text:
            messagebox.showwarning(
                "Нечего сохранять",
                "Сначала выполните транскрибацию.",
            )
            return

        default_name = "voice2llm_transcript.txt"
        if self._audio_path is not None:
            default_name = f"{self._audio_path.stem}_transcript.txt"

        target = filedialog.asksaveasfilename(
            title="Сохранить текст",
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

        self._set_status(f"Сохранено: {Path(target).name}")

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
            "Остановить обработку и закрыть программу?",
        )
        if not should_close:
            return

        if self._worker is not None:
            self._worker.cancel()
        self._close_when_done = True
        self._close_deadline = time.monotonic() + CANCEL_CLOSE_TIMEOUT_SECONDS
        self._set_status("Закрываю…")
        self.primary_button.configure(state="disabled")

    def _destroy_now(self) -> None:
        self._closing = True
        self.destroy()
