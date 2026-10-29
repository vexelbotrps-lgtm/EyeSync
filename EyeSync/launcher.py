"""
launcher.py - starts EyeSync and looks after it (EyeSync.pyw, EyeSync.exe).
==========================================================================

Opens the EyeSync window and keeps EyeSync running: it stays OFF until the
EyeSync website dashboard connects. If EyeSync is already running, its
window is brought to the front instead.

At Windows login (see autostart.py) it starts with --standby: nothing opens
until the EyeSync website dashboard is opened. EyeSync closes again 30 s
after the dashboard is closed, and waits for it once more.

If EyeSync crashes it is restarted after a short back-off. If it is quit on
purpose - "Quit EyeSync" in the window, Ctrl+Alt+Shift+F12, or the close
gesture (mouth wide open 3 s) - it closes, and this launcher waits in a tiny
standby (no camera, no AI): EyeSync starts again, minimised, only when the
website's Connect button is pressed (its automatic reconnects are ignored,
see site_access.wait_for_visit), or at once when the EyeSync shortcut is
opened.

Everything is logged to eyesync.log (app_paths.DATA_DIR).
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import app_paths
import desktop
import site_access

LOG_PATH = Path(os.environ.get("EYESYNC_LOG", app_paths.DATA_DIR / "eyesync.log"))
log = logging.getLogger("eyesync")


class _AppendHandler(logging.Handler):
    """Opens the log only to write a line, so EyeSync can still rotate it."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(self.format(record) + "\n")
        except OSError:
            pass


EXIT_QUIT, EXIT_DASHBOARD_CLOSED = 0, 3      # = live_inference.EXIT_*


def run_until_quit(args: list[str]) -> bool:
    """
    Run EyeSync; restart it after a crash. Returns when it closed on
    purpose: True if the user quit it, False if the dashboard was closed.
    """
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    backoff = 3.0
    while True:
        started = time.monotonic()
        code = subprocess.call(app_paths.engine_command(args), cwd=str(app_paths.DATA_DIR),
                               creationflags=flags)
        if code in (EXIT_QUIT, EXIT_DASHBOARD_CLOSED):
            return code == EXIT_QUIT
        if time.monotonic() - started > 60:
            backoff = 3.0                          # it ran fine for a while
        log.warning(f"[standby] EyeSync stopped unexpectedly after "
                    f"{time.monotonic() - started:.0f}s (exit code {code:#x}) - "
                    f"restarting in {backoff:.0f}s")
        time.sleep(backoff)
        backoff = min(backoff * 2, 60.0)


def parse_args(argv: list[str]) -> tuple[bool, list[str], bool]:
    """
    (wait in standby?, the engine's arguments, opened from the website?).
    --url LINK is the website's "Open EyeSync" button (an eyesync: link,
    autostart.register_url): start for the website at once, minimised, like
    a Connect click on the standby. The link's text itself is never used.
    """
    argv = list(argv)
    from_site = "--url" in argv
    if from_site:
        i = argv.index("--url")
        del argv[i:i + 2]
        if "--minimized" not in argv:
            argv.append("--minimized")
    standby = "--standby" in argv and not from_site
    return standby, [a for a in argv if a != "--standby"], from_site


def main() -> int:
    if not desktop.acquire_single_instance("EyeSync.Background"):
        # Already running: wake it from standby, or show its window.
        if desktop.is_running("EyeSync.LiveInference") or not desktop.OpenSignal.send():
            desktop.bring_window_to_front("EyeSync", "TkTopLevel")
        return 0

    handler = _AppendHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s",
                                           datefmt="%H:%M:%S"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False

    wake = desktop.OpenSignal()
    standby, base_args, from_site = parse_args(sys.argv[1:])
    if from_site:
        log.info("[standby] Opened from the website (eyesync: link) - starting EyeSync")
    args = base_args
    after_quit = False
    while True:
        if not standby:
            # After a quit only Connect restarts it; after the dashboard was
            # closed, the dashboard reopening (or reconnecting) starts it too.
            after_quit = run_until_quit(args)
        standby = False
        wake.clear()
        log.info("[standby] Waiting - EyeSync opens when the EyeSync website dashboard is "
                 "opened (or the EyeSync shortcut)")
        how = site_access.wait_for_visit(wake=wake.is_set, after_quit=after_quit)
        log.info(f"[standby] {'Website opened' if how == 'website' else 'Shortcut opened'}"
                 " - starting EyeSync")
        # From the website: start minimised, so the window doesn't cover the page.
        args = [a for a in base_args if a != "--minimized"]
        if how == "website":
            args.append("--minimized")


if __name__ == "__main__":
    sys.exit(main())
