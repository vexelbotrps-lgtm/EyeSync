"""Entry point for the portable EyeSync edition; no installer or updater."""
from __future__ import annotations

import os
import sys

os.environ["EYESYNC_STANDALONE"] = "1"


def main() -> int:
    from window_identity import set_process_identity
    set_process_identity()
    first, rest = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if first == "--calibrate-portable":
        from control_backend.calibrate import main as calibrate
        return calibrate(rest)
    if first == "--command-panel":
        from control_backend.wispr_command_panel import main as panel
        sys.argv = [sys.argv[0], *rest]
        panel()
        return 0
    if first not in ("", "--standalone"):
        raise SystemExit(f"Unknown standalone option: {first}")
    sys.argv = [sys.argv[0], "--always-active", "--mouse", "software",
                "--stall-timeout", "45", "--site-origin", "http://localhost:8000",
                "--site-origin", "https://vexelbotrps-lgtm.github.io", *rest]
    import live_inference
    return live_inference.main()


if __name__ == "__main__":
    raise SystemExit(main())
