"""Build a standalone Voice2LLM binary with PyInstaller.

Usage
-----
    python build.py            # one-folder build in dist/Voice2LLM
    python build.py --zip      # also create a distributable archive
    python build.py --console  # keep the console window (debugging)

The one-folder (``--onedir``) layout is intentional: CTranslate2 loads model
weights straight from ``models/`` next to the executable, so long recordings do
not pay a one-time extraction cost into a temporary directory.

Optional local assets
---------------------
* ``assets/icon.ico`` (Windows) / ``assets/icon.icns`` (macOS) are used as the
  application icon when present.
"""

from __future__ import annotations

import argparse
import importlib.util
import platform
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import List, Optional

APP_NAME = "Voice2LLM"
APP_VERSION = "1.0.0"
APP_PUBLISHER = "Mesa-afk"

ROOT = Path(__file__).resolve().parent
ENTRY_POINT = ROOT / "main.py"
BUILD_DIR = ROOT / "build"
DIST_DIR = ROOT / "dist"
OUTPUT_DIR = DIST_DIR / APP_NAME
SPEC_PATH = ROOT / f"{APP_NAME}.spec"

REQUIRED_MODULES = ("faster_whisper", "customtkinter", "pyperclip", "PyInstaller")

#: Whole packages that must ship with all submodules, data files and binaries.
COLLECT_ALL = (
    "customtkinter",
    "faster_whisper",
    "ctranslate2",
    "tokenizers",
    "onnxruntime",
    "av",
    "huggingface_hub",
)

#: Modules imported dynamically (plugins, backends, optional codecs).
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

#: Heavy libraries that are never used at runtime; excluding them keeps the
#: archive substantially smaller.
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


def log(message: str) -> None:
    """Print a timestamped build message."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def human_size(size_bytes: int) -> str:
    """Format a byte count as MB or GB for build summaries."""
    megabytes = size_bytes / (1024 * 1024)
    if megabytes >= 1024:
        return f"{megabytes / 1024:.2f} GB"
    return f"{megabytes:.1f} MB"


def platform_tag() -> str:
    """Return an archive suffix describing the current platform and CPU."""
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
    """Abort early with an actionable message when the toolchain is missing."""
    if sys.version_info < (3, 10):
        raise SystemExit(
            "Voice2LLM requires Python 3.10 or newer to build. "
            f"Current interpreter: {sys.version.split()[0]}"
        )
    if not ENTRY_POINT.is_file():
        raise SystemExit(f"Entry point not found: {ENTRY_POINT}")

    missing = [
        module
        for module in REQUIRED_MODULES
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise SystemExit(
            "Missing build dependencies: "
            + ", ".join(missing)
            + "\nInstall them with:  pip install -r requirements.txt"
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
    """Generate the Windows VERSIONINFO resource (file properties dialog)."""
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
    # Written to the system temp directory: PyInstaller's ``--clean`` wipes
    # the workpath, and this file must still exist when it is read.
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        prefix="voice2llm-version-",
        encoding="utf-8",
        delete=False,
    )
    with handle:
        handle.write(content)
    log(f"Generated Windows version resource for v{APP_VERSION}")
    return Path(handle.name)


def build_arguments(console: bool) -> List[str]:
    """Assemble the PyInstaller command line."""
    arguments: List[str] = [
        str(ENTRY_POINT),
        "--name",
        APP_NAME,
        "--onedir",
        "--noconfirm",
        "--clean",
        "--noupx",
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(BUILD_DIR),
        "--specpath",
        str(ROOT),
        "--log-level",
        "INFO",
    ]

    # ``--noconsole`` is only meaningful on Windows and macOS. On Linux the
    # flag is accepted but has no effect, so it is skipped for clarity.
    if not console and sys.platform in ("win32", "darwin"):
        arguments.append("--noconsole")

    icon = find_icon()
    if icon is not None:
        arguments.extend(["--icon", str(icon)])
        log(f"Using icon {icon.relative_to(ROOT)}")
    else:
        log("No application icon found (assets/icon.ico) — using the default")

    version_file = write_windows_version_info()
    if version_file is not None:
        arguments.extend(["--version-file", str(version_file)])

    for module in COLLECT_ALL:
        arguments.extend(["--collect-all", module])
    for module in HIDDEN_IMPORTS:
        arguments.extend(["--hidden-import", module])
    for module in EXCLUDES:
        arguments.extend(["--exclude-module", module])

    return arguments


def run_pyinstaller(arguments: List[str]) -> None:
    """Invoke PyInstaller in-process and stream its log output."""
    from PyInstaller.__main__ import run as pyinstaller_run

    log("Running PyInstaller (this can take a few minutes)…")
    pyinstaller_run(arguments)


def verify_output() -> Path:
    """Ensure the expected executable and its data files were produced."""
    executable_name = f"{APP_NAME}.exe" if sys.platform == "win32" else APP_NAME
    executable = OUTPUT_DIR / executable_name
    if not executable.is_file():
        raise SystemExit(
            "PyInstaller finished but the executable is missing:\n"
            f"  expected: {executable}"
        )

    total_size = sum(
        file.stat().st_size for file in OUTPUT_DIR.rglob("*") if file.is_file()
    )
    log(f"Built {executable.relative_to(ROOT)}")
    log(f"Output size: {human_size(total_size)}")
    return executable


def create_archive() -> Path:
    """Zip the one-folder build preserving the ``Voice2LLM`` top-level folder."""
    archive = DIST_DIR / f"{APP_NAME}-{platform_tag()}.zip"
    if archive.exists():
        archive.unlink()

    log(f"Creating {archive.name}…")
    with zipfile.ZipFile(
        archive, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as zip_file:
        for path in sorted(OUTPUT_DIR.rglob("*")):
            if not path.is_file():
                continue
            arcname = Path(APP_NAME) / path.relative_to(OUTPUT_DIR)
            zip_file.write(path, arcname=str(arcname))

    log(f"Archive created: {archive.relative_to(ROOT)} ({human_size(archive.stat().st_size)})")
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
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="reuse previous build/dist artefacts instead of removing them",
    )
    return parser.parse_args()


def main() -> int:
    """Run the full build pipeline and report a summary."""
    arguments = parse_arguments()
    started = time.monotonic()

    check_environment()
    if arguments.no_clean:
        DIST_DIR.mkdir(parents=True, exist_ok=True)
        BUILD_DIR.mkdir(parents=True, exist_ok=True)
        log("Keeping previous build artefacts (--no-clean)")
    else:
        clean_previous_output()

    DIST_DIR.mkdir(parents=True, exist_ok=True)
    run_pyinstaller(build_arguments(arguments.console))
    executable = verify_output()

    archive: Optional[Path] = None
    if arguments.zip:
        archive = create_archive()

    elapsed = time.monotonic() - started
    log(f"Done in {elapsed:.1f}s")
    log(f"Executable: {executable}")
    if archive is not None:
        log(f"Archive:    {archive}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
