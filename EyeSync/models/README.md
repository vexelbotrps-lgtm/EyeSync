# Model files

The three runtime models are intentionally not committed here a second time.
They are preserved in the repository's `EyeSync-1.5.7.zip` upload:

- `face_landmarker.task`
- `ggml-base.en-q8_0.bin`
- `ggml-silero-v6.2.0.bin`

From the repository root, restore them before running the full app or build:

```sh
python scripts/eyesync_ci.py prepare-models
```

The Windows release workflow runs this command automatically. The model bytes
are unchanged between EyeSync 1.5.7 and 1.5.9.
