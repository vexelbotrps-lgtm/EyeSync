"""
app_paths.py - where EyeSync's files are, run from source or as EyeSync.exe.
===========================================================================

From source (python live_inference.py, EyeSync.pyw, the zip + install.bat)
everything is in this folder, as always.

As EyeSync.exe (build_exe.py; a single file that installs itself into
%LocalAppData%\\Programs\\EyeSync, see installer.py):
  * APP_DIR - the program's own read-only files (face model, README), as
    unpacked while it runs (RUNTIME_DIR);
  * DATA_DIR - the user's files (settings and calibration, API key, logs):
    next to EyeSync.exe, i.e. the same files a zip install kept there, so
    switching from the zip install to the exe keeps them.

Standard library only: the always-running standby imports it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))
SOURCE_DIR = Path(__file__).resolve().parent
APP_DIR = Path(getattr(sys, "_MEIPASS", SOURCE_DIR)) if FROZEN else SOURCE_DIR
DATA_DIR = Path(sys.executable).resolve().parent if FROZEN else SOURCE_DIR
# EyeSync.exe is one file: while it runs, it unpacks itself into a _MEI*
# folder here (build_exe.py's --runtime-tmpdir), removed again when it ends.
# Kept apart from the install folder, so an update never touches the files
# the running installer uses.
RUNTIME_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "EyeSync" / "runtime"
RUNTIME_TMPDIR_OPTION = r"%LOCALAPPDATA%\EyeSync\runtime"     # the same, for PyInstaller


def engine_command(args: list[str]) -> list[str]:
    """The command that runs the EyeSync engine (live_inference) with ``args``."""
    if FROZEN:
        return [sys.executable, "--engine", *args]
    exe = Path(sys.executable)
    pythonw = exe.with_name("pythonw.exe")       # no console window
    return [str(pythonw if pythonw.exists() else exe), str(SOURCE_DIR / "live_inference.py"), *args]


def blink_test_command() -> list[str]:
    """The command that runs the blink test (tools/record_blinks.py, a window of its own)."""
    if FROZEN:
        return [sys.executable, "--blink-test"]
    return [sys.executable, str(SOURCE_DIR / "tools" / "record_blinks.py")]
