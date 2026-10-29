"""Download and featurise the public GRID audiovisual corpus (CC BY 4.0).

Raw videos are temporary. Only the compact numerical NPZ and an attribution
sidecar are kept. Zenodo's published MD5 is verified before any video is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tools"))

import video_features  # noqa: E402

RECORD = "3625687"
SOURCE = f"https://zenodo.org/records/{RECORD}"
ATTRIBUTION = "Cooke, Barker, Cunningham & Shao (2006), The Grid Audio-Visual Speech Corpus"
LICENSE = "CC BY 4.0"
log = logging.getLogger("eyesync-grid")


def request(url: str, timeout: int = 90):
    return urllib.request.urlopen(urllib.request.Request(
        url, headers={"User-Agent": "EyeSync-public-dataset-evaluation/1.0"}), timeout=timeout)


def metadata() -> dict:
    with request(f"https://zenodo.org/api/records/{RECORD}") as response:
        return json.load(response)


def download(speaker: int, destination: Path, info: dict) -> tuple[Path, str]:
    key = f"s{speaker}.zip"
    entry = next((item for item in info["files"] if item["key"] == key), None)
    if entry is None:
        raise RuntimeError(f"GRID has no downloadable {key} (speaker 21 has no video)")
    expected = entry["checksum"].split(":", 1)[-1].lower()
    url = entry["links"]["self"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        digest = hashlib.md5()  # noqa: S324 - Zenodo's published integrity value, not security
        size = 0
        try:
            log.info(f"downloading {key} ({entry['size'] / 1e6:.0f} MB), attempt {attempt}")
            with request(url) as response, destination.open("wb") as stream:
                while block := response.read(4 << 20):
                    stream.write(block)
                    digest.update(block)
                    size += len(block)
            if size != entry["size"] or digest.hexdigest() != expected:
                raise RuntimeError(f"size/MD5 mismatch for {key}: {size} bytes, {digest.hexdigest()}")
            log.info(f"verified {key}: {size:,} bytes, md5={expected}")
            return destination, expected
        except Exception:
            destination.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(attempt * 5)
    raise AssertionError("unreachable")


def extract_selected(archive: Path, folder: Path, count: int) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        names = sorted(name for name in source.namelist()
                       if Path(name).suffix.lower() in video_features.VIDEO_SUFFIXES)
        if not names:
            raise RuntimeError(f"no supported videos in {archive}")
        if 0 < count < len(names):
            indices = np.linspace(0, len(names) - 1, count).round().astype(int)
            names = [names[i] for i in np.unique(indices)]
        for i, name in enumerate(names):
            # Basename plus an ordinal avoids traversal and keeps replay order.
            target = folder / f"{i:04d}_{Path(name).name}"
            with source.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, 4 << 20)
        log.info(f"extracted {len(names)} of {len(source.namelist())} archive entries")
        return len(names)


def annotate(path: Path, speaker: int, archive_md5: str) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        data, columns = archive["data"], archive["columns"]
        meta = json.loads(str(archive["meta"].item()))
    meta.update(dataset="GRID", speaker=speaker, source=SOURCE,
                attribution=ATTRIBUTION, license=LICENSE, archive_md5=archive_md5)
    tmp = path.with_suffix(".annotated.npz")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, data=data, columns=columns,
                            meta=np.array(json.dumps(meta, separators=(",", ":"))))
    tmp.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = {key: value for key, value in meta.items() if key != "videos"}
    sidecar["npz_sha256"] = digest
    sidecar_path = path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return sidecar


def run(speaker: int, count: int, output: Path, keep: Path | None = None) -> dict:
    info = metadata()
    temporary = None if keep else tempfile.TemporaryDirectory(prefix=f"eyesync-grid-s{speaker}-")
    work = keep or Path(temporary.name)
    try:
        archive, checksum = download(speaker, work / f"s{speaker}.zip", info)
        extract_selected(archive, work / "videos", count)
        archive.unlink()                    # raw video is not retained after use
        meta = video_features.extract([str(work / "videos")], output)
        log.info(f"features: {meta}")
        return annotate(output, speaker, checksum)
    finally:
        if temporary:
            temporary.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speaker", type=int, required=True)
    parser.add_argument("--count", type=int, default=500, help="evenly sampled sentences; 0 = all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-work", type=Path, help="keep raw downloads for local debugging")
    args = parser.parse_args()
    if not 1 <= args.speaker <= 34 or args.speaker == 21:
        parser.error("speaker must be 1..34 except 21 (GRID has no speaker-21 video)")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run(args.speaker, args.count, args.output, args.keep_work)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
