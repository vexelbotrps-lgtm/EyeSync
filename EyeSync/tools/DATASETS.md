# Public dataset evaluation

## GRID audiovisual speech corpus

- **Use here:** false-positive evaluation and conservative blink-timing calibration on people reading sentences, with speaker-disjoint development and held-out test sets.
- **Authors:** Martin Cooke, Jon Barker, Stuart Cunningham and Xu Shao (2006).
- **Paper:** *An audio-visual corpus for speech perception and automatic speech recognition*, Journal of the Acoustical Society of America 120(5), 2421–2424.
- **Official record:** https://zenodo.org/records/3625687
- **License:** CC BY 4.0. Attribution and the official record are also embedded in every extracted NPZ sidecar.
- **Privacy/storage:** the workflows download public videos temporarily, verify Zenodo's published MD5, and retain compact numerical features long enough for evaluation. Raw videos are deleted. Temporary eye-region review sheets and the numerical cache are removed from the final source tree; only aggregate JSON results remain.

Readers in GRID are not operating EyeSync, so every click, face gesture or scroll produced by replay is a false positive. Sixteen speakers and 500 evenly spaced sentences per speaker supplied about 200 minutes of replay. Speakers 1–8 were development data; speakers 9–16 were reserved for the final frozen comparison.

Public eye-region review showed that many remaining false clicks were genuine, longer natural blinks—not open downcast eyes and not labelled deliberate intent. A deep image/intent model would therefore have unsupported labels. The selected change instead learns a safe tick time from three slow calibration closures: 90% of their median detected duration, bounded to 0.30–0.65 seconds. Uncalibrated behavior and an explicit `--min-blink` accessibility override remain unchanged.

At the frozen 0.60-second representative setting:

- Development: **150 → 17 false clicks** over 98.814 minutes; 482/482 real-appearance extended holds clicked.
- Held-out speakers 9–16: **147 → 16 false clicks** over 99.999 minutes (**89.1% reduction**, 1.470 → 0.160 per minute).
- Held-out real-appearance holds: **639/640 under both old and new timing**, so no measured recall loss; the single miss is a pre-existing limitation, not a new regression.
- Right-clicks, gestures and scroll notches did not increase. One smile gesture occurred under both policies; no right-clicks or scroll notches occurred.

The recall probes reuse real closed-eye appearance and extend their timing. They test sensitivity, not whether a model can infer intent. See `experiments/grid/README.md` and the aggregate JSON files for exact per-speaker results.

## Other datasets considered

- **Eyeblink8** (https://www.blinkingmatters.com/research): useful labelled natural video, but GPL licensing and a small four-person corpus make it a better independent comparison than the first training source.
- **RT-BENE** (https://zenodo.org/records/3685316): more than 200,000 labelled eye patches, but the patches lack the full-face/head-pose inputs used here and the dataset/code are CC BY-NC-SA 4.0. It is not bundled or used by the current calibration pipeline.

No dataset can prove that an accessibility controller is perfect for every person, camera, posture or lighting condition. GRID readers were not deliberately operating EyeSync, and offline public video is not real-camera validation. The verified goal is a substantial improvement on unseen public speakers without sacrificing the tested deliberate-blink behavior.
