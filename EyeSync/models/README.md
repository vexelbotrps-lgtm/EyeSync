# Model files

The runtime models are intentionally not committed here. The release helper
downloads them from the official MediaPipe and whisper.cpp model repositories:

- `face_landmarker.task`
- `ggml-small.en-q8_0.bin`
- `ggml-silero-v6.2.0.bin`

From the repository root, restore them before running the full app or build:

```sh
python scripts/eyesync_ci.py prepare-models
```

The Windows release workflow runs this command automatically. Every model is
verified against its pinned SHA-256 before a release is built.
