"""Build the single-file EyeSync setup executable from the portable ZIP."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from PyInstaller.__main__ import run

ROOT = Path(__file__).resolve().parent
VERSION = "1.0.7"
PAYLOAD = ROOT / f"EyeSync-Standalone-Windows-v{VERSION}.zip"
WORK = ROOT / "setup_build_standalone_v107"
OUTPUT = ROOT / f"EyeSync-Setup-v{VERSION}.exe"


def main() -> None:
    if sys.platform != "win32":
        raise RuntimeError("Build EyeSync setup on Windows.")
    if not PAYLOAD.is_file():
        raise FileNotFoundError(PAYLOAD)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    run([
        str(ROOT / "standalone_setup.py"),
        "--name", "EyeSync-Setup",
        "--onefile", "--windowed", "--clean", "--noconfirm", "--noupx",
        "--icon", str(ROOT / "EyeSync" / "models" / "eyesync.ico"),
        "--distpath", str(WORK / "dist"),
        "--workpath", str(WORK / "work"),
        "--specpath", str(WORK),
        "--add-data", f"{PAYLOAD}{os.pathsep}.",
    ])
    (WORK / "dist" / "EyeSync-Setup.exe").replace(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
