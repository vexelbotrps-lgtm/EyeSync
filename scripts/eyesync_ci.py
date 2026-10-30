"""Prepare and verify EyeSync Windows release artifacts in CI.

This helper deliberately uses only the Python standard library so it can run
before the application's dependencies are installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("face_landmarker.task", "ggml-small.en-q8_0.bin", "ggml-silero-v6.2.0.bin")
MODEL_SHA256 = {
    "face_landmarker.task": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    "ggml-small.en-q8_0.bin": "67a179f608ea6114bd3fdb9060e762b588a3fb3bd00c4387971be4d177958067",
    "ggml-silero-v6.2.0.bin": "2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987",
}
MODEL_URLS = {
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "ggml-small.en-q8_0.bin":
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en-q8_0.bin",
    "ggml-silero-v6.2.0.bin":
        "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_models() -> None:
    """Fetch pinned runtime models from their official repositories."""
    destination = ROOT / "EyeSync" / "models"
    destination.mkdir(parents=True, exist_ok=True)
    for name, url in MODEL_URLS.items():
        target = destination / name
        if target.is_file() and sha256(target) == MODEL_SHA256[name]:
            print(f"verified {target.relative_to(ROOT)} ({target.stat().st_size:,} bytes)")
            continue
        partial = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url) as source, partial.open("wb") as output:
            for block in iter(lambda: source.read(4 << 20), b""):
                output.write(block)
        if sha256(partial) != MODEL_SHA256[name]:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"downloaded model checksum is wrong: {name}")
        partial.replace(target)
        print(f"downloaded {target.relative_to(ROOT)} ({target.stat().st_size:,} bytes)")


def local_info(release_dir: Path) -> tuple[dict, Path]:
    latest_path = release_dir / "latest.json"
    info = json.loads(latest_path.read_text(encoding="utf-8"))
    version = info["version"]
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise RuntimeError(f"invalid release version: {version!r}")
    source_version = (ROOT / "EyeSync" / "site_access.py").read_text(encoding="utf-8")
    if f'VERSION = "{version}"' not in source_version:
        raise RuntimeError("latest.json and site_access.VERSION disagree")
    if info.get("file") != "EyeSync.exe":
        raise RuntimeError("the website's main download must be EyeSync.exe")
    exe = release_dir / "EyeSync.exe"
    if not exe.is_file():
        raise RuntimeError("EyeSync.exe is missing")
    if info.get("size") != exe.stat().st_size:
        raise RuntimeError("latest.json has the wrong executable size")
    if info.get("sha256") != sha256(exe):
        raise RuntimeError("latest.json has the wrong executable SHA-256")
    package = release_dir / info["zip"]
    if not package.is_file() or package.stat().st_size < 1_000_000:
        raise RuntimeError("the source package named by latest.json is missing")
    selftest = (release_dir / "selftest.txt").read_text(encoding="utf-8")
    if "\nOK\n" not in f"\n{selftest}\n":
        raise RuntimeError("the executable has no successful self-test record")
    print(f"verified local EyeSync {version}: {exe.stat().st_size:,} bytes, sha256={info['sha256']}")
    return info, exe


def verify_local(release_dir: Path) -> None:
    local_info(release_dir)


def verify_upload(tag: str, release_dir: Path) -> None:
    info, exe = local_info(release_dir)
    expected_tag = f"v{info['version']}"
    if tag != expected_tag:
        raise RuntimeError(f"release tag {tag} does not match {expected_tag}")
    repo = os.environ.get("GITHUB_REPOSITORY") or subprocess.run(
        ["gh", "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    raw = subprocess.run(
        ["gh", "api", f"repos/{repo}/releases/tags/{tag}"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    release = json.loads(raw)
    if release["tag_name"] != tag:
        raise RuntimeError("GitHub returned the wrong release")
    assets = {asset["name"]: asset for asset in release["assets"]}
    package = release_dir / info["zip"]
    for name, path in ((exe.name, exe), (info["zip"], package)):
        if name not in assets:
            raise RuntimeError(f"GitHub Release {tag} is missing {name}")
        uploaded = assets[name]
        if uploaded.get("state") != "uploaded" or uploaded.get("size") != path.stat().st_size:
            raise RuntimeError(f"the uploaded {name} size or state differs from the tested file")
        digest = uploaded.get("digest") or ""
        if digest and digest != f"sha256:{sha256(path)}":
            raise RuntimeError(f"the uploaded {name} digest differs from the tested file")
    print(f"verified uploaded release {tag}: {', '.join(sorted(assets))}")


def update_site(release_dir: Path) -> None:
    info, _ = local_info(release_dir)
    # Write only the public metadata fields, in the website's established order.
    public = {key: info[key] for key in ("version", "file", "size", "sha256", "date", "zip")}
    target = ROOT / "downloads" / "latest.json"
    target.write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    print(f"updated {target.relative_to(ROOT)} to EyeSync {info['version']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-models")
    for name in ("verify-local", "verify-upload", "update-site"):
        p = sub.add_parser(name)
        p.add_argument("release_dir", type=Path)
        if name == "verify-upload":
            p.add_argument("tag")
    args = parser.parse_args()
    if args.command == "prepare-models":
        prepare_models()
    elif args.command == "verify-local":
        verify_local(args.release_dir)
    elif args.command == "verify-upload":
        verify_upload(args.tag, args.release_dir)
    elif args.command == "update-site":
        update_site(args.release_dir)


if __name__ == "__main__":
    main()
