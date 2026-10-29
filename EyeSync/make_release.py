"""
make_release.py - build the EyeSync package you give to other people.
=====================================================================

    python make_release.py            -> E:\\EyeSync-Release\\EyeSync-<version>.zip

(The Python version. The main download, EyeSync-Setup.exe, is built by
build_exe.py into the same release folder next to the source folder;
EYESYNC_RELEASE_DIR chooses another.)

Contains only the program: the code, the installer, the README, the face
model, the XIAO firmware and the self-tests. Never your personal data:
settings, log or API key - the build stops if anything like that would be
included. Every user gets their own API key and settings the first time
EyeSync runs.

Then it unpacks the zip into a temporary folder and runs the self-tests
there, so a missing file shows up now and not on someone else's PC.

Recipients: unzip anywhere (for example Documents\\EyeSync) and double-click
install.bat.
"""

from __future__ import annotations

import fnmatch
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RELEASE = Path(os.environ.get("EYESYNC_RELEASE_DIR", HERE.parent / "EyeSync-Release"))

INCLUDE = [
    "EyeSync.pyw", "install.bat", "uninstall.bat", "requirements.txt", "README.md", "WEBSITE.md",
    "THIRD_PARTY_NOTICES.md",
    "live_inference.py", "launcher.py", "app_paths.py", "installer.py", "eyesync_app.py",
    "main_window.py", "face_engine.py", "gaze_engine.py", "desktop.py", "software_mouse.py",
    "site_access.py", "autostart.py", "voice_typing.py", "make_release.py",
    "models/face_landmarker.task", "models/ggml-base.en-q8_0.bin", "models/ggml-silero-v6.2.0.bin",
    "firmware/xiao_esp32s3_mouse/xiao_esp32s3_mouse.ino",
    "tests/test_eyesync.py", "tests/data/voice_phrases.wav",
]
# Personal or machine-specific - must never be shipped.
FORBIDDEN = ["datasets/*", "*.jpg", "*.png", "*.npz", "*.joblib", "*.log", "*.log.*",
             "eyesync_settings.json", "eyesync_api_key.txt", "*.tmp", ".venv/*", "*__pycache__*"]


def version() -> str:
    import site_access
    return site_access.VERSION


def build() -> Path:
    ver = version()
    for rel in INCLUDE:
        bad = [p for p in FORBIDDEN if fnmatch.fnmatch(rel, p)]
        if bad:
            raise SystemExit(f"refusing to ship {rel} (matches {bad})")
        if not (HERE / rel).is_file():
            raise SystemExit(f"missing {rel}")
    out = RELEASE / f"EyeSync-{ver}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    for old in RELEASE.glob("EyeSync-*.zip"):       # only the newest zip is given out
        if old != out:
            old.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in INCLUDE:
            src = HERE / rel
            if rel.endswith((".bat", ".ino", ".py", ".pyw", ".md", ".txt")):
                data = src.read_bytes()
                if rel.endswith(".bat"):            # cmd.exe needs CRLF line endings
                    data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
                z.writestr(f"EyeSync/{rel}", data)
            else:
                z.write(src, f"EyeSync/{rel}")
    with zipfile.ZipFile(out) as z:                 # double-check what actually went in
        for name in z.namelist():
            rel = name.split("/", 1)[1]
            if any(fnmatch.fnmatch(rel, p) for p in FORBIDDEN):
                out.unlink()
                raise SystemExit(f"personal file {rel} ended up in the package - deleted it")
    return out


def self_test(package: Path) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(package) as z:
            z.extractall(tmp)
        root = Path(tmp) / "EyeSync"
        env = dict(os.environ, EYESYNC_LOG=str(Path(tmp) / "test.log"))
        r = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"],
                           cwd=root, env=env, capture_output=True, text=True)
        print(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.stdout[-300:])
        return r.returncode == 0


def write_latest(package: Path) -> Path:
    """
    latest.json in the release folder, for the website (see WEBSITE.md).
    If build_exe.py already wrote it for this version, the setup stays the
    main download ("file") and the zip is added as "zip".
    """
    import hashlib
    import json
    import time
    out = package.parent / "latest.json"
    try:
        info = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        info = {}
    if info.get("version") == version() and str(info.get("file", "")).endswith(".exe"):
        info["zip"] = package.name
    else:
        info = {"version": version(), "file": package.name, "size": package.stat().st_size,
                "sha256": hashlib.sha256(package.read_bytes()).hexdigest(),
                "date": time.strftime("%Y-%m-%d")}
    out.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> int:
    package = build()
    size = package.stat().st_size / 1e6
    print(f"Built {package} ({size:.1f} MB)")
    print("Testing the unpacked package ...")
    if not self_test(package):
        package.unlink()                           # never leave a failed package to publish
        print("Self-tests FAILED in the unpacked package - deleted it.")
        return 1
    print(f"Wrote {write_latest(package)}")
    print(f"OK - it's in {RELEASE}. Users extract the zip and double-click install.bat "
          "(also to update).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
