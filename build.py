"""Build the Voice2LLM desktop application with PyInstaller.

The script produces a one-folder (``--onedir``) Windows distribution in
``dist/Voice2LLM``. A one-folder build is intentional: CTranslate2 loads model
weights directly from the ``models/`` directory next to ``Voice2LLM.exe`` and
never unpacks hundreds of megabytes into ``%TEMP%`` on every launch, unlike
``--onefile``.

Usage
-----
    python build.py             # default Windows build (no console window)
    python build.py --zip       # also create a distributable archive
    python build.py --console   # keep the console attached (debugging)

All heavy native dependencies (``faster-whisper``, CTranslate2, PyAV,
onnxruntime) and ``customtkinter`` assets are collected with ``--collect-all``,
otherwise PyInstaller's static analysis misses DLLs, ``.pyd`` modules, theme
JSON files and the bundled Silero VAD model, and the produced executable fails
at runtime.

The process exits with a non-zero status code when PyInstaller fails or when
the expected executable is missing, so GitHub Actions stops immediately.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path
from typing import List, Optional

APP_NAME = "Voice2LLM"
APP_VERSION = "1.1.0"
APP_PUBLISHER = "Mesa-afk"

ROOT = Path(__file__).resolve().parent
ENTRY_POINT = ROOT / "main.py"
BUILD_DIR = ROOT / "build"
DIST_DIR = ROOT / "dist"
OUTPUT_DIR = DIST_DIR / APP_NAME
SPEC_PATH = ROOT / f"{APP_NAME}.spec"
VERSION_INFO_PREFIX = "voice2llm-version-"
ARCHIVE_PREFIX = APP_NAME

REQUIRED_MODULES = ("faster_whisper", "customtkinter", "pyperclip", "PyInstaller")

#: Packages whose entire contents must be bundled: compiled extension modules,
#: DLLs, metadata, theme JSON files and other package data.
REQUIRED_COLLECT_ALL = (
    "customtkinter",
    "faster_whisper",
    "ctranslate2",
    "tokenizers",
)

#: Additional runtime dependencies of faster-whisper that also ship native
#: libraries or data files (PyAV bundles FFmpeg, onnxruntime bundles the VAD
#: engine, huggingface_hub downloads model weights).
RUNTIME_COLLECT_ALL = (
    "onnxruntime",
    "av",
    "huggingface_hub",
)

#: Modules imported dynamically that static analysis cannot see.
HIDDEN_IMPORTS = (
    "faster_whisper",
    "ctranslate2",
    "tokenizers",
    "onnxruntime",
    "av",
    "av.audio",
    "av.audio.codeccontext",
    "av.audio.resampler",
    "av.container",
    "huggingface_hub",
    "pyperclip",
    "numpy",
    "tkinter",
    "_tkinter",
    "queue",
)

#: Heavy packages that are never imported at runtime; excluding them keeps the
#: one-folder distribution significantly smaller.
EXCLUDES = (
    "torch",
    "torchaudio",
    "torchvision",
    "tensorflow",
    "jax",
    "matplotlib",
    "pandas",
    "scipy",
    "pytest",
    "IPython",
    "notebook",
    "sklearn",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "wx",
)


class BuildError(RuntimeError):
    """Raised when the build cannot start or produces no executable."""


def log(message: str) -> None:
    """Print a timestamped build message and flush it for CI logs."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def human_size(size_bytes: int) -> str:
    """Format a byte count as megabytes or gigabytes."""
    megabytes = size_bytes / (1024 * 1024)
    if megabytes >= 1024:
        return f"{megabytes / 1024:.2f} GB"
    return f"{megabytes:.1f} MB"


def platform_tag() -> str:
    """Return ``windows-x64``, ``macos-arm64``, ``linux-x64`` and similar."""
    machine = platform.machine().lower() or "unknown"
    architecture = {
        "amd64": "x64",
        "x86_64": "x64",
        "arm64": "arm64",
        "aarch64": "arm64",
        "i386": "x86",
        "i686": "x86",
    }.get(machine, machine)
    if sys.platform == "win32":
        family = "windows"
    elif sys.platform == "darwin":
        family = "macos"
    else:
        family = "linux"
    return f"{family}-{architecture}"


def check_environment() -> None:
    """Validate the interpreter, the entry point and installed dependencies."""
    if sys.version_info < (3, 10):
        raise BuildError(
            "Voice2LLM requires Python 3.10 or newer to build; "
            f"the current interpreter is {sys.version.split()[0]}."
        )
    if not ENTRY_POINT.is_file():
        raise BuildError(f"Entry point not found: {ENTRY_POINT}")

    missing = [
        module
        for module in REQUIRED_MODULES
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise BuildError(
            "Missing build dependencies: "
            + ", ".join(missing)
            + ". Install them with: python -m pip install -r requirements.txt"
        )

    log(f"Python {sys.version.split()[0]} · target {platform_tag()}")


def clean_previous_output() -> None:
    """Remove artefacts from earlier builds so no stale file is shipped."""
    for path in (OUTPUT_DIR, BUILD_DIR):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            log(f"Removed {path.relative_to(ROOT)}")

    if SPEC_PATH.exists():
        SPEC_PATH.unlink()
        log(f"Removed {SPEC_PATH.name}")

    DIST_DIR.mkdir(parents=True, exist_ok=True)


def find_icon() -> Optional[Path]:
    """Return the platform icon when the repository provides one."""
    if sys.platform == "win32":
        candidates = (ROOT / "assets" / "icon.ico", ROOT / "icon.ico")
    elif sys.platform == "darwin":
        candidates = (ROOT / "assets" / "icon.icns", ROOT / "icon.icns")
    else:
        candidates = (ROOT / "assets" / "icon.png", ROOT / "icon.png")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def write_windows_version_info() -> Optional[Path]:
    """Generate a Windows VERSIONINFO resource for the executable properties."""
    if sys.platform != "win32":
        return None

    numeric_parts = [
        int(part) if part.isdigit() else 0 for part in APP_VERSION.split(".")
    ]
    while len(numeric_parts) < 4:
        numeric_parts.append(0)
    file_version = tuple(numeric_parts[:4])

    content = f"""VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={file_version},
    prodvers={file_version},
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(
        '040904B0',
        [
          StringStruct('CompanyName', '{APP_PUBLISHER}'),
          StringStruct('FileDescription', 'Voice2LLM - local audio transcription for LLMs'),
          StringStruct('FileVersion', '{APP_VERSION}'),
          StringStruct('InternalName', '{APP_NAME}'),
          StringStruct('LegalCopyright', 'BSD 3-Clause License'),
          StringStruct('OriginalFilename', '{APP_NAME}.exe'),
          StringStruct('ProductName', '{APP_NAME}'),
          StringStruct('ProductVersion', '{APP_VERSION}')
        ]
      )
    ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""

    # The resource lives in the system temp directory: PyInstaller's ``--clean``
    # wipes the workpath, and the file must survive until it is read.
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        prefix=VERSION_INFO_PREFIX,
        encoding="utf-8",
        delete=False,
    )
    with handle:
        handle.write(content)

    log(f"Generated Windows version resource for v{APP_VERSION}")
    return Path(handle.name)


def build_arguments(console: bool) -> List[str]:
    """Assemble the strict PyInstaller argument list."""
    arguments: List[str] = [
        str(ENTRY_POINT),
        f"--name={APP_NAME}",
        "--onedir",
        "--clean",
        "--noconfirm",
        "--noupx",
        "--log-level=INFO",
        f"--distpath={DIST_DIR}",
        f"--workpath={BUILD_DIR}",
        f"--specpath={ROOT}",
    ]

    if not console:
        arguments.append("--noconsole")

    for module in REQUIRED_COLLECT_ALL + RUNTIME_COLLECT_ALL:
        arguments.append(f"--collect-all={module}")

    for module in HIDDEN_IMPORTS:
        arguments.append(f"--hidden-import={module}")

    for module in EXCLUDES:
        arguments.append(f"--exclude-module={module}")

    icon = find_icon()
    if icon is not None:
        arguments.append(f"--icon={icon}")
        log(f"Using icon {icon.relative_to(ROOT)}")
    else:
        log("No application icon found (assets/icon.ico) — using the default")

    version_file = write_windows_version_info()
    if version_file is not None:
        arguments.append(f"--version-file={version_file}")

    return arguments


def run_pyinstaller(arguments: List[str]) -> int:
    """Run PyInstaller in-process and return its exit status.

    ``PyInstaller.__main__.run`` may raise :class:`SystemExit` on analysis or
    build errors. Converting it to an integer lets the caller fail the GitHub
    Actions job deterministically instead of printing a traceback.
    """
    try:
        from PyInstaller.__main__ import run as pyinstaller_run
    except ImportError as error:
        raise BuildError(
            "PyInstaller is not installed. "
            "Run: python -m pip install 'pyinstaller>=6.0.0'"
        ) from error

    os.chdir(ROOT)
    log("Running PyInstaller (this can take a few minutes)…")

    try:
        pyinstaller_run(arguments)
    except SystemExit as error:
        code = error.code if isinstance(error.code, int) else 1
        return code
    except KeyboardInterrupt:
        log("Build interrupted by the user.")
        return 1

    return 0


def verify_output() -> Path:
    """Ensure the executable and its support files were produced."""
    executable_name = f"{APP_NAME}.exe" if sys.platform == "win32" else APP_NAME
    executable = OUTPUT_DIR / executable_name
    if not executable.is_file():
        raise BuildError(
            "PyInstaller finished but the executable is missing. "
            f"Expected: {executable}"
        )

    total_size = sum(
        file.stat().st_size for file in OUTPUT_DIR.rglob("*") if file.is_file()
    )
    log(f"Built {executable.relative_to(ROOT)}")
    log(f"Output size: {human_size(total_size)}")
    return executable


def create_archive() -> Path:
    """Zip the one-folder build, preserving the ``Voice2LLM`` top-level folder."""
    archive = DIST_DIR / f"{ARCHIVE_PREFIX}-{platform_tag()}.zip"
    if archive.exists():
        archive.unlink()

    log(f"Creating {archive.name}…")
    with zipfile.ZipFile(
        archive,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as zip_file:
        for path in sorted(OUTPUT_DIR.rglob("*")):
            if not path.is_file():
                continue
            arcname = Path(APP_NAME) / path.relative_to(OUTPUT_DIR)
            zip_file.write(path, arcname=str(arcname))

    archive_size = archive.stat().st_size
    if archive_size <= 0:
        raise BuildError(f"Archive {archive.name} is empty.")

    log(f"Archive created: {archive.relative_to(ROOT)} ({human_size(archive_size)})")
    return archive


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for local builds."""
    parser = argparse.ArgumentParser(
        description="Build the Voice2LLM desktop application with PyInstaller."
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="create a distributable .zip archive next to the build",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="keep a console window attached (useful for debugging)",
    )
    return parser.parse_args()


def main() -> int:
    """Execute the build pipeline and return a process exit code."""
    arguments = parse_arguments()
    started = time.monotonic()

    try:
        check_environment()
        clean_previous_output()

        exit_code = run_pyinstaller(build_arguments(arguments.console))
        if exit_code != 0:
            log(f"PyInstaller failed with exit code {exit_code}.")
            return 1

        executable = verify_output()
        archive = create_archive() if arguments.zip else None
    except BuildError as error:
        log(f"BUILD FAILED: {error}")
        return 1
    except Exception:
        traceback.print_exc()
        log("BUILD FAILED: unexpected error (see the traceback above).")
        return 1

    log(f"Done in {time.monotonic() - started:.1f}s")
    log(f"Executable: {executable}")
    if archive is not None:
        log(f"Archive:    {archive}")
    return 0


if __name__ == "__main__":
    exit_status = main()
    if exit_status != 0:
        log("Build failed. See the log above for details.")
        sys.exit(1)
    sys.exit(0)
