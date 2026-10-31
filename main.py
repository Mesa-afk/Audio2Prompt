"""Audio2Prompt application entry point.

Run with ``python main.py`` for development, or build a standalone binary with
``python build.py``. The module keeps all Windows-specific setup (DPI awareness
and the taskbar AppUserModelID) isolated from the actual application code.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import tempfile
import traceback
from pathlib import Path

MINIMUM_PYTHON = (3, 10)


def _ensure_standard_streams() -> None:
    """Guarantee that ``print()`` works in console-less PyInstaller builds.

    On Windows a ``--noconsole`` executable starts with ``sys.stdout`` and
    ``sys.stderr`` set to ``None``. Anything that prints would then raise, so
    stdout is redirected to ``os.devnull`` and stderr to a log file in the
    temporary directory.
    """
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")

    if sys.stderr is None:
        log_path = Path(tempfile.gettempdir()) / "audio2prompt.log"
        try:
            sys.stderr = open(log_path, "a", encoding="utf-8", buffering=1)
        except OSError:
            sys.stderr = open(os.devnull, "w", encoding="utf-8")


def _show_fatal_error(message: str) -> None:
    """Report a startup failure without relying on a console window."""
    print(f"Audio2Prompt: {message}", file=sys.stderr)
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(
            0,
            message,
            "Audio2Prompt — критическая ошибка",
            0x00000010,  # MB_ICONERROR
        )
    except Exception as dialog_error:
        print(f"Unable to display the error dialog: {dialog_error}", file=sys.stderr)


def _enable_windows_integration() -> None:
    """Enable crisp text rendering and a proper taskbar identity on Windows."""
    if sys.platform != "win32":
        return

    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception as shcore_error:
        print(f"Per-monitor DPI awareness unavailable: {shcore_error}", file=sys.stderr)
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception as user32_error:
            print(f"System DPI awareness unavailable: {user32_error}", file=sys.stderr)

    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "MesaAFK.Audio2Prompt.Desktop.1"
        )
    except Exception as shell_error:
        print(f"Taskbar AppUserModelID unavailable: {shell_error}", file=sys.stderr)


def main() -> int:
    """Create the window and run the Tk event loop."""
    multiprocessing.freeze_support()
    _ensure_standard_streams()
    _enable_windows_integration()

    if sys.version_info < MINIMUM_PYTHON:
        required = ".".join(str(part) for part in MINIMUM_PYTHON)
        _show_fatal_error(
            f"Требуется Python {required} или новее.\n"
            f"Текущая версия: {sys.version.split()[0]}."
        )
        return 1

    try:
        import customtkinter as ctk

        from ui import Audio2PromptApp

        ctk.set_appearance_mode("dark")
        app = Audio2PromptApp()
    except Exception:
        _show_fatal_error(
            "Не удалось запустить приложение.\n\n" + traceback.format_exc()
        )
        return 1

    try:
        app.mainloop()
    except KeyboardInterrupt:
        return 0
    except Exception:
        _show_fatal_error(
            "Приложение было остановлено из-за ошибки.\n\n" + traceback.format_exc()
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
