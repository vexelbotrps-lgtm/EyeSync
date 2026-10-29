"""Build a Windows ZIP that runs without Python or an installer.

Run from this folder with .venv\\Scripts\\python.exe build_portable.py.
The output is EyeSync-Portable-Windows-PUBLIC.zip in this folder.
"""
from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "EyeSync"
WORK = ROOT / "portable_build_public"
DIST = WORK / "dist"
PACKAGE = ROOT / "EyeSync-Portable-Windows-PUBLIC.zip"


def main() -> int:
    if sys.platform != "win32":
        raise RuntimeError("Build the Windows package on Windows.")
    if PACKAGE.exists():
        raise FileExistsError(f"Move or rename the existing release first: {PACKAGE}")
    from PyInstaller.__main__ import run

    options = [
        str(SOURCE / "eyesync_app.py"), "--name", "EyeSync", "--onedir",
        "--windowed", "--clean", "--noconfirm", "--noupx",
        "--distpath", str(DIST), "--workpath", str(WORK / "work"),
        "--specpath", str(WORK),
        "--paths", str(SOURCE / "control_backend"),
        "--paths", str(SOURCE / "tools"),
        "--hidden-import", "live_inference",
        "--hidden-import", "main_window",
        "--hidden-import", "control_backend.calibrate",
        "--hidden-import", "control_backend.wispr_command_panel",
        "--hidden-import", "control_backend.voice_commands",
        "--hidden-import", "gaze_core",
        "--collect-all", "mediapipe",
        "--collect-all", "pywhispercpp",
        "--collect-all", "sounddevice",
        "--exclude-module", "matplotlib",
    ]
    for name in ("face_landmarker.task", "ggml-base.en-q8_0.bin",
                 "ggml-silero-v6.2.0.bin"):
        options += ["--add-data", f"{SOURCE / 'models' / name}{os.pathsep}models"]
    options += ["--add-data", f"{SOURCE / 'control_backend' / 'voice_commands.ps1'}"
                f"{os.pathsep}control_backend"]
    run(options)

    app = DIST / "EyeSync"
    (app / "Start EyeSync.bat").write_text(
        '@echo off\r\ncd /d "%~dp0"\r\n'
        'if not exist "gaze_model.joblib" (\r\n'
        '  "EyeSync.exe" --calibrate-portable\r\n'
        '  if errorlevel 1 exit /b 1\r\n'
        ')\r\nstart "" "%~dp0EyeSync.exe" --portable\r\n', encoding="utf-8")
    (app / "Calibrate Gaze.bat").write_text(
        '@echo off\r\ncd /d "%~dp0"\r\n'
        '"EyeSync.exe" --calibrate-portable\r\n'
        'if errorlevel 1 exit /b 1\r\n'
        'start "" "%~dp0EyeSync.exe" --portable\r\n', encoding="utf-8")
    (app / "README.txt").write_text(
        'EyeSync for Windows 10/11, 64-bit.\n\n'
        'Extract the whole ZIP, then double-click Start EyeSync.bat. '
        'The first launch shows a nine-dot calibration. Point your nose at '
        'each dot. The app then opens. Later, use Calibrate in the app to '
        'retrain it.\n\n'
        'A webcam and microphone are required. The command panel opens with '
        'the app. Close both eyes for 1 second to toggle voice typing. '
        'Smile for left click; open your mouth for right click. All gestures '
        'work only while the app button says Pause.\n\n'
        'Keep the _internal folder beside EyeSync.exe. Personal settings and '
        'the gaze model are stored beside EyeSync.exe.\n', encoding="utf-8")
    with zipfile.ZipFile(PACKAGE, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for path in sorted(app.rglob("*")):
            if path.is_file():
                archive.write(path, Path("EyeSync") / path.relative_to(app))
    print(PACKAGE)
    print(f"ZIP size: {PACKAGE.stat().st_size / 1024**2:.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
