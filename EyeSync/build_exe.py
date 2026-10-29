"""Build the single-file Windows EyeSync.exe and verify the actual artifact.

Run on Windows with the requirements installed::

    py -3.14 build_exe.py

The output goes to ``..\\EyeSync-Release`` (or EYESYNC_RELEASE_DIR). The
build signs the exe when EYESYNC_SIGN is set, runs the bundled regression
tests inside the exe, installs a throw-away copy without registering it,
and tests that copy too. No camera, registration or personal settings are
needed for these checks.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import app_paths

HERE = Path(__file__).resolve().parent
RELEASE = Path(os.environ.get("EYESYNC_RELEASE_DIR", HERE.parent / "EyeSync-Release"))
BUILD = Path(os.environ.get("EYESYNC_BUILD_DIR", HERE / "build"))
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def version() -> str:
    import site_access
    return site_access.VERSION


def version_file(ver: str) -> Path:
    """Write PyInstaller's Windows Properties > Details version resource."""
    parts = [int(p) for p in ver.split(".")]
    if len(parts) != 3 or any(p < 0 or p > 65535 for p in parts):
        raise ValueError(f"not a three-part Windows version: {ver}")
    numbers = (*parts, 0)
    BUILD.mkdir(parents=True, exist_ok=True)
    out = BUILD / "version_info.txt"
    out.write_text(f"""# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={numbers},
    prodvers={numbers},
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
        [StringStruct('CompanyName', 'EyeSync'),
         StringStruct('FileDescription', 'EyeSync hands-free mouse'),
         StringStruct('FileVersion', '{ver}'),
         StringStruct('InternalName', 'EyeSync'),
         StringStruct('LegalCopyright', 'EyeSync'),
         StringStruct('OriginalFilename', 'EyeSync.exe'),
         StringStruct('ProductName', 'EyeSync'),
         StringStruct('ProductVersion', '{ver}')])
    ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
""", encoding="utf-8")
    return out


def find_signtool() -> str | None:
    configured = os.environ.get("EYESYNC_SIGNTOOL")
    if configured:
        return configured
    found = shutil.which("signtool.exe") or shutil.which("signtool")
    if found:
        return found
    roots = [Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Windows Kits" / "10" / "bin",
             Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Windows Kits" / "10" / "bin"]
    candidates = [p for root in roots if root.is_dir() for p in root.glob("*/*/signtool.exe")]
    return str(max(candidates, key=lambda p: p.stat().st_mtime)) if candidates else None


def sign_commands(signtool: str, args: str, exe: Path) -> tuple[str, str]:
    """Return the sign and verify commands; kept separate for regression tests."""
    return (f'"{signtool}" sign {args} "{exe}"',
            f'"{signtool}" verify /pa "{exe}"')


def sign(exe: Path) -> bool:
    """Sign only when requested. A requested but impossible signing is fatal."""
    args = os.environ.get("EYESYNC_SIGN", "").strip()
    if not args:
        print("Code signing not requested (EYESYNC_SIGN is empty).")
        return True
    tool = find_signtool()
    if not tool:
        print("EYESYNC_SIGN is set, but signtool.exe was not found.", file=sys.stderr)
        return False
    sign_cmd, verify_cmd = sign_commands(tool, args, exe)
    for command in (sign_cmd, verify_cmd):
        print(command)
        if subprocess.run(command, shell=True).returncode:
            return False
    return True


def _add_data(args: list[str], source: Path, destination: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    args.extend(["--add-data", f"{source}{os.pathsep}{destination}"])


def pyinstaller_args(ver: str) -> list[str]:
    """An allow-list bundle: no settings, logs, recordings or API keys."""
    args = [
        str(HERE / "eyesync_app.py"),
        "--name", "EyeSync",
        "--onefile",
        "--windowed",
        "--clean",
        "--noconfirm",
        "--noupx",
        "--distpath", str(RELEASE),
        "--workpath", str(BUILD / "pyinstaller"),
        "--specpath", str(BUILD),
        "--runtime-tmpdir", app_paths.RUNTIME_TMPDIR_OPTION,
        "--version-file", str(version_file(ver)),
        # The window's Blink test button runs EyeSync.exe --blink-test, which
        # imports tools/record_blinks.py (1.5.8 and 1.5.9 were built without it).
        "--paths", str(HERE / "tools"),
        "--hidden-import", "record_blinks",
        # Only MediaPipe's drawing helpers import matplotlib, and face_engine
        # stands in for it (_defer_matplotlib): a smaller, faster exe (1.5.12).
        "--exclude-module", "matplotlib",
        # Native/runtime data used by the vision, audio and local speech stack.
        "--collect-all", "mediapipe",
        "--collect-all", "pywhispercpp",
        "--collect-all", "sounddevice",
    ]
    for name in ("README.md", "WEBSITE.md", "THIRD_PARTY_NOTICES.md", "requirements.txt"):
        _add_data(args, HERE / name, ".")
    for name in ("face_landmarker.task", "ggml-base.en-q8_0.bin", "ggml-silero-v6.2.0.bin"):
        _add_data(args, HERE / "models" / name, "models")
    _add_data(args, HERE / "firmware" / "xiao_esp32s3_mouse" / "xiao_esp32s3_mouse.ino",
              "firmware/xiao_esp32s3_mouse")
    _add_data(args, HERE / "tests" / "test_eyesync.py", "tests")
    _add_data(args, HERE / "tests" / "data" / "voice_phrases.wav", "tests/data")
    return args


def build(ver: str) -> Path:
    if os.name != "nt":
        raise RuntimeError("EyeSync.exe must be built on Windows.")
    from PyInstaller.__main__ import run as pyinstaller

    RELEASE.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    pyinstaller(pyinstaller_args(ver))
    exe = RELEASE / "EyeSync.exe"
    if not exe.is_file() or exe.stat().st_size < 1_000_000:
        raise RuntimeError("PyInstaller did not produce a plausible EyeSync.exe")
    return exe


def self_test(exe: Path, out: Path) -> None:
    """Run the tests from the files embedded in this exact executable."""
    out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, EYESYNC_LOG=str(out.with_suffix(".log")))
    result = subprocess.run([str(exe), "--selftest", str(out)], env=env,
                            creationflags=NO_WINDOW, timeout=900)
    text = out.read_text(encoding="utf-8") if out.is_file() else "no self-test output"
    print(text)
    if result.returncode or "\nOK\n" not in f"\n{text}\n":
        raise RuntimeError(f"self-test failed for {exe}; see {out}")


def test_install(exe: Path) -> None:
    """Copy-install without shortcuts/registry changes, then test that copy."""
    with tempfile.TemporaryDirectory(prefix="eyesync-install-test-") as tmp:
        dest = Path(tmp) / "installed"
        subprocess.run([str(exe), "--install", "--quiet", "--no-register", "--dest", str(dest)],
                       check=True, creationflags=NO_WINDOW, timeout=300)
        installed = dest / "EyeSync.exe"
        if not installed.is_file():
            raise RuntimeError("the test installer did not copy EyeSync.exe")
        if hashlib.sha256(installed.read_bytes()).digest() != hashlib.sha256(exe.read_bytes()).digest():
            raise RuntimeError("the installed exe differs from the tested release exe")
        self_test(installed, Path(tmp) / "installed-selftest.txt")


def write_latest(exe: Path) -> Path:
    info = {
        "version": version(),
        "file": exe.name,
        "size": exe.stat().st_size,
        "sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
        "date": time.strftime("%Y-%m-%d"),
    }
    package = RELEASE / f"EyeSync-{version()}.zip"
    if package.is_file():
        info["zip"] = package.name
    out = RELEASE / "latest.json"
    out.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return out


def write_readme(exe: Path) -> None:
    (RELEASE / "READ ME FIRST.txt").write_text(f"""EyeSync {version()} for Windows 10/11
===================================

EyeSync.exe is the main download. Open it and choose Install; no administrator
rights are needed. Your settings, calibration and API key stay when updating.

EyeSync-{version()}.zip is the optional Python-source package, when present.
Extract it and run install.bat instead. It needs internet access during setup.

latest.json contains the website's version, byte size and SHA-256 metadata.
WEBSITE.md is the maintainer's release/upload guide.
selftest.txt records the tests run inside this exact EyeSync.exe.

The build also installed a throw-away copy without shortcuts or registration
and ran the same self-tests there. No personal files are included.
""", encoding="utf-8")
    shutil.copy2(HERE / "WEBSITE.md", RELEASE / "WEBSITE.md")


def main() -> int:
    ver = version()
    print(f"Building EyeSync {ver} for Windows...")
    exe = build(ver)
    if not sign(exe):
        raise RuntimeError("code signing was requested but failed")
    print("Testing the built executable...")
    self_test(exe, RELEASE / "selftest.txt")
    print("Testing a clean, unregistered install...")
    test_install(exe)
    write_readme(exe)
    print(f"Wrote {write_latest(exe)}")
    print(f"OK: {exe} ({exe.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"BUILD FAILED: {exc}", file=sys.stderr)
        raise
