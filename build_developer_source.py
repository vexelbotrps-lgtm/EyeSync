"""Package the public application source and runtime models for developers."""
from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VERSION = "1.0.5"
OUTPUT = ROOT / f"EyeSync-Developer-Source-v{VERSION}.zip"
PREFIX = Path(f"EyeSync-Developer-Source-v{VERSION}")
TOP_LEVEL = {
    "README.md", "LICENSE", "BRAHMASTRA_README.md", "Run-EyeSync.bat",
    "build_portable.py", "build_setup.py", "standalone_setup.py",
    "build_developer_source.py",
}
MODELS = (
    "face_landmarker.task", "ggml-base.en-q8_0.bin", "ggml-silero-v6.2.0.bin",
)
INSTRUCTIONS = """EyeSync Standalone developer source, version 1.0.5

This ZIP includes application source and the three runtime model files. It
does not include a personal gaze calibration or a Python environment.

On Windows with Python 3.15.2 installed, from this extracted folder:

  py -3.15 -m venv .venv
  .venv\\Scripts\\python.exe -m pip install -r EyeSync\\requirements.txt
  .venv\\Scripts\\python.exe EyeSync\\standalone_app.py --calibrate-portable
  .venv\\Scripts\\python.exe EyeSync\\standalone_app.py --standalone

The application source is MIT-licensed. See LICENSE and
EyeSync/THIRD_PARTY_NOTICES.md for component notices.
"""


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0")
    files = []
    for raw in tracked:
        if not raw:
            continue
        relative = Path(raw.decode("utf-8"))
        if relative.parts[0] == "EyeSync" or str(relative) in TOP_LEVEL:
            files.append(relative)
    files.extend(Path("EyeSync/models") / name for name in MODELS)
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in sorted(set(files)):
            path = ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            archive.write(path, PREFIX / relative)
        archive.writestr(str(PREFIX / "DEVELOPER_README.txt"), INSTRUCTIONS)
    print(f"{OUTPUT} ({OUTPUT.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
