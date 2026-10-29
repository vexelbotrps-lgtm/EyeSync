"""
eyesync_app.py - the program inside EyeSync.exe (built by build_exe.py).
=======================================================================

EyeSync.exe is a single file that is every part of EyeSync, chosen by its
command line. Opened from anywhere but its install folder (a download),
it installs or updates itself first (installer.py).

    EyeSync.exe                     open EyeSync (the EyeSync shortcut)
    EyeSync.exe --install [...]     install / update this copy (installer.py)
    EyeSync.exe --standby           wait for the website (Windows login)
    EyeSync.exe --url eyesync:...   opened by the website's eyesync: link
                                    (autostart.register_url): start for it
    EyeSync.exe --engine ARGS       the engine itself (live_inference.main);
                                    the two above start it
    EyeSync.exe --setup-finish      run by EyeSync-Setup.exe after copying the
                                    files: shortcuts, start at login, the entry
                                    in Settings > Apps, then wait for the website
    EyeSync.exe --uninstall         remove EyeSync (Settings > Apps > Uninstall)
    EyeSync.exe --autostart ACTION  as "python autostart.py ACTION"
    EyeSync.exe --selftest FILE     run the self-tests inside the exe and write
                                    the result to FILE (build_exe.py does this)
    EyeSync.exe --blink-test        the blink test (tools/record_blinks.py; the
                                    window's Blink test button starts it)

From source, the same parts are EyeSync.pyw, live_inference.py and
autostart.py; this file is only needed for the exe.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import app_paths

SITE = "https://aniruddh9140.github.io/face-control-website/"
# The user's own files in the install folder; everything else there is the
# program (or its logs), removed by an uninstall that keeps the settings.
USER_FILES = ("eyesync_settings.json", "eyesync_api_key.txt")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    first, rest = (argv[0], argv[1:]) if argv else ("", [])
    if first == "--engine":
        sys.argv = [sys.argv[0], *rest]           # what live_inference's options see
        import live_inference
        return live_inference.main()
    if first == "--portable":
        sys.argv = [sys.argv[0], "--always-active", "--mouse", "software",
                    "--stall-timeout", "45", "--site-origin", "http://localhost:8000",
                    "--site-origin", "https://aniruddh9140.github.io", *rest]
        import live_inference
        return live_inference.main()
    if first == "--command-panel":
        sys.argv = [sys.argv[0], *rest]
        from control_backend.wispr_command_panel import main as panel_main
        panel_main()
        return 0
    if first == "--calibrate-portable":
        from control_backend.calibrate import main as calibrate_main
        return calibrate_main(rest)
    if first == "--autostart":
        sys.argv = ["autostart.py", *rest]
        import autostart
        return autostart.main()
    if first == "--install":
        import installer
        return installer.main(rest)
    if first == "--setup-finish":
        return setup_finish()
    if first == "--uninstall":
        return uninstall()
    if first == "--blink-test":
        import record_blinks
        return record_blinks.main()
    if first == "--selftest":
        return selftest(Path(rest[0]) if rest else app_paths.DATA_DIR / "selftest.txt")
    if app_paths.FROZEN:
        import installer
        if not argv and not installer.is_installed_copy():
            return installer.main([])            # a downloaded EyeSync.exe: install it
        clean_runtime()
    import launcher
    return launcher.main()


def clean_runtime() -> None:
    """
    EyeSync.exe unpacks itself into app_paths.RUNTIME_DIR while it runs and
    removes that again when it ends - but not when it was force-closed. With
    no other EyeSync running, remove such leftovers.
    """
    import shutil

    import desktop
    runtime = app_paths.RUNTIME_DIR
    mine = Path(getattr(sys, "_MEIPASS", "")).resolve()
    if not runtime.is_dir() or any(desktop.is_running(n) for n in
                                   ("EyeSync.Background", "EyeSync.LiveInference")):
        return
    for d in runtime.glob("_MEI*"):
        if d.resolve() != mine:
            shutil.rmtree(d, ignore_errors=True)


def setup_finish() -> int:
    """After EyeSync-Setup.exe copied the files: register EyeSync, then wait for the website."""
    import autostart
    import site_access
    autostart.add_shortcuts()
    autostart.enable()                     # the standby: the website's Connect starts EyeSync
    autostart.register_url()               # and eyesync: links, when nothing is waiting
    autostart.register_app(site_access.VERSION)
    autostart.start(standby=True)
    return 0


def uninstall() -> int:
    """Settings > Apps > EyeSync > Uninstall."""
    import tkinter as tk
    from tkinter import messagebox

    import autostart
    root = tk.Tk()
    root.withdraw()
    if not messagebox.askyesno("Remove EyeSync", "Remove EyeSync from this computer?", parent=root):
        return 1
    keep = not messagebox.askyesno(
        "Remove EyeSync", "Also delete your EyeSync settings, calibration and API key?\n\n"
        "Choose No to keep them for a later reinstall.", default="no", parent=root)
    autostart.stop()
    autostart.disable()
    autostart.remove_shortcuts()
    autostart.unregister_url()
    autostart.unregister_app()
    folder = app_paths.DATA_DIR
    # This program is still running: a hidden script deletes the files once it has exited.
    lines = ["@echo off", "timeout /t 2 /nobreak >nul"]
    if keep:
        for p in folder.iterdir():
            if p.name not in USER_FILES:
                lines.append(f'rmdir /s /q "{p}"' if p.is_dir() else f'del /f /q "{p}"')
    else:
        lines.append(f'rmdir /s /q "{folder}"')
    lines.append(f'rmdir /s /q "{app_paths.RUNTIME_DIR}"')       # where EyeSync.exe unpacked itself
    lines.append(f'rmdir "{app_paths.RUNTIME_DIR.parent}" 2>nul')
    lines.append('del /f /q "%~f0"')                 # and the script itself
    script = Path(os.environ.get("TEMP", str(folder.parent))) / "eyesync-uninstall.cmd"
    script.write_text("\r\n".join(lines) + "\r\n", encoding="mbcs")
    subprocess.Popen(["cmd", "/c", str(script)], cwd=str(script.parent),
                     creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS)
    messagebox.showinfo("Remove EyeSync", "EyeSync was removed." + (
        f"\n\nYour settings were kept in {folder}." if keep else ""), parent=root)
    root.destroy()
    return 0


def selftest(out: Path) -> int:
    """The regression tests, run inside the exe (proves the bundle is complete)."""
    import io
    import tempfile
    import unittest
    os.environ.setdefault("EYESYNC_LOG", str(Path(tempfile.gettempdir()) / "eyesync_selftest.log"))
    tests = app_paths.APP_DIR / "tests"
    sys.path.insert(0, str(tests))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=1).run(
        unittest.defaultTestLoader.discover(str(tests)))
    lines = [f"EyeSync.exe self-test: {result.testsRun} tests, {len(result.failures)} failures, "
             f"{len(result.errors)} errors, {len(result.skipped)} skipped",
             "OK" if result.wasSuccessful() and result.testsRun else "FAILED", stream.getvalue()[-6000:]]
    out.write_text("\n".join(lines), encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
