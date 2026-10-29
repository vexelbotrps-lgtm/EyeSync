# GRID blink-timing evaluation

Public source: [GRID audiovisual speech corpus](https://zenodo.org/records/3625687), Cooke, Barker, Cunningham and Shao (2006), **CC BY 4.0**.

Sixteen speakers, 500 evenly spaced sentences per speaker, were extracted to temporary numerical features. Raw videos, face-review sheets and the 82 MB numerical cache were removed after evaluation; the GitHub workflows can regenerate them. Only small aggregate reports remain here.

## Preregistered split and candidate

- Development: speakers 1–8 (98.814 minutes).
- Final held-out evaluation: speakers 9–16 (99.999 minutes), not inspected while choosing the candidate.
- The public readers were **not operating EyeSync**. Their clicks are false positives, not examples of deliberate-click intent.
- The selected policy is calibration-derived timing: 90% of the median detected duration of three slow calibration closures, bounded to 0.30–0.65 seconds. The frozen representative candidate used a 0.60-second tick; the uncalibrated default remains 0.30 seconds.
- Recall probes extend real, clearly closed public blink appearances until 0.65 seconds after the detector recognises closure. They measure sensitivity to real eye appearance, **not intent recognition**.

## Results

| Evaluation | Old 0.30 s tick | Calibrated 0.60 s candidate |
|---|---:|---:|
| Development false clicks (speakers 1–8) | 150 | 17 |
| Development false clicks/minute | 1.518 | 0.172 |
| Development appearance holds | — | 482/482 |
| **Held-out false clicks (speakers 9–16)** | **147** | **16** |
| **Held-out false clicks/minute** | **1.470** | **0.160** |
| Held-out reduction | — | **89.1%** |
| Held-out appearance holds | 639/640 | 639/640 |

The same speaker-15 appearance probe fails under old and new timing, so the candidate caused no measured recall loss in this probe set. Right-clicks, gestures and scroll notches did not increase on held-out replay (one smile gesture occurred under both policies; there were no right-clicks or scroll notches).

Files:

- `summary.json`: baseline replay totals for all 16 speakers.
- `dev-timing-sweep.json`: development-only timing sweep.
- `dev-recall-final.json`: corrected development recall probe.
- `heldout-evaluation.json`: frozen held-out comparison and per-speaker results.

These are offline public-video results, not a guarantee for deliberate users, every camera, lighting condition or accessibility need. No deep intent model was trained: visual review showed that many remaining false clicks were genuine longer natural blinks, so a conservative, user-demonstrated timing calibration was the better-supported change.
