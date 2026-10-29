"""
EyeSync.pyw - Start EyeSync from source. Double-click this (or the EyeSync shortcut).
=====================================================================================

Opens the EyeSync window; with --standby (Windows login) it waits for the
EyeSync website instead. All of it is in launcher.py; EyeSync.exe runs the
same code.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import launcher  # noqa: E402  (after sys.path tweak)

if __name__ == "__main__":
    sys.exit(launcher.main())
