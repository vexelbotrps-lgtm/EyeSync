"""
installer.py - EyeSync.exe installs (and updates) itself.
=========================================================

EyeSync.exe is ONE file: the whole program, and its own installer. Opened
from anywhere other than its install folder (e.g. the Downloads folder),
it shows a small window and:

  1. closes EyeSync if it is running (an update);
  2. removes the old program files from %LocalAppData%\\Programs\\EyeSync,
     keeping the user's settings, calibration, API key and logs (also those
     of the older zip install, which used the same folder);
  3. copies itself there;
  4. registers that copy: Desktop and Start menu shortcuts, start at login
     (waiting for the website), eyesync: links (the website's "Open
     EyeSync"), and the Settings > Apps entry with Uninstall;
  5. starts it in the background and opens the EyeSync website.

    EyeSync.exe --install [--quiet] [--dest DIR] [--no-register]
        the same without the window (--quiet), into another folder, or only
        copying (--no-register; build_exe.py's check)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path

import site_access

VERSION = site_access.VERSION
SITE = "https://aniruddh9140.github.io/face-control-website/"
DEFAULT_DEST = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "Programs" / "EyeSync"
LOG_PATH = Path(os.environ.get("TEMP", str(Path.home()))) / "EyeSync-install.log"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# What may be removed from the install folder: EyeSync's program files, of
# this version and of older ones (the zip install, test builds). Anything
# else is left alone.
PROGRAM_DIRS = {"_internal", ".venv", "__pycache__", "tests", "models", "firmware", "dist"}
PROGRAM_FILES = {"EyeSync.exe", "requirements.txt", "README.md", "WEBSITE.md", "WEBSITE_CHANGES.md",
                 "selftest.txt"}
PROGRAM_SUFFIXES = {".py", ".pyw", ".bat"}

log = logging.getLogger("eyesync-install")


def is_installed_copy(exe: Path | None = None) -> bool:
    """Is this EyeSync.exe the installed one (not a download being opened)?"""
    exe = (exe or Path(sys.executable)).resolve()
    return str(exe.parent).lower() == str(DEFAULT_DEST.resolve()).lower()


def is_user_file(name: str) -> bool:
    """Settings and calibration, the API key, and the logs - always kept."""
    return name in ("eyesync_settings.json", "eyesync_api_key.txt") or (
        name.startswith("eyesync") and ".log" in name)


def is_program_file(p: Path) -> bool:
    if is_user_file(p.name):
        return False
    if p.is_dir():
        return p.name in PROGRAM_DIRS
    return p.name in PROGRAM_FILES or p.suffix.lower() in PROGRAM_SUFFIXES


def _others_running() -> bool:
    """Another EyeSync.exe than this one (and the process that started it)."""
    me = {os.getpid(), os.getppid()}
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq EyeSync.exe", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True, creationflags=NO_WINDOW).stdout
    pids = {int(line.split('","')[1]) for line in out.splitlines() if line.count('","') >= 1
            and line.split('","')[1].isdigit()}
    return bool(pids - me)


def stop_eyesync() -> None:
    """Close a running EyeSync (EyeSync.exe, or the Python version) and wait for it."""
    import autostart
    autostart.stop()
    for _ in range(40):                          # files can't be replaced while in use
        if not _others_running():
            return
        time.sleep(0.25)


def remove_old(dest: Path) -> None:
    if (dest / "build_exe.py").exists():
        # A developer's source folder (build_exe.py isn't in any package): its
        # .py files are the source code, not an old install - never delete them.
        raise RuntimeError(f"{dest} is EyeSync's source folder, not an install folder.")
    for p in list(dest.iterdir()):
        if not is_program_file(p):
            continue
        for attempt in range(8):                 # a program that just closed may hold files a moment
            try:
                shutil.rmtree(p) if p.is_dir() else p.unlink()
                break
            except OSError:
                if attempt == 7:
                    raise
                time.sleep(0.5)
        log.info(f"removed {p.name}")


def copy_with_progress(src: Path, dst: Path, report) -> None:
    total, done = src.stat().st_size or 1, 0
    tmp = dst.with_suffix(".new")
    with open(src, "rb") as fin, open(tmp, "wb") as fout:
        while chunk := fin.read(4 << 20):
            fout.write(chunk)
            done += len(chunk)
            report("Copying EyeSync...", 0.15 + 0.7 * done / total)
    os.replace(tmp, dst)


def install(dest: Path, register: bool, report=lambda text, frac: None) -> Path:
    """Install or update EyeSync in ``dest``; ``report(text, 0..1)`` shows progress."""
    log.info(f"EyeSync {VERSION}: installing {sys.executable} into {dest} (register={register})")
    dest.mkdir(parents=True, exist_ok=True)
    if register:
        report("Closing EyeSync if it is running...", 0.03)
        stop_eyesync()
    report("Removing the old version (your settings stay)...", 0.08)
    remove_old(dest)
    target = dest / "EyeSync.exe"
    copy_with_progress(Path(sys.executable), target, report)
    log.info("copied")
    if register:
        import autostart
        report("Adding shortcuts and the start with Windows...", 0.9)
        autostart.EXE = target                   # register the installed copy, not this download
        autostart.add_shortcuts()
        autostart.enable()                       # the standby: the website's Connect starts EyeSync
        autostart.register_url()                 # eyesync: links, when nothing is waiting
        autostart.register_app(VERSION)
        autostart.start(standby=True)
    report("Done", 1.0)
    log.info("installed")
    return target


def run_window(dest: Path) -> int:
    import tkinter as tk
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title("EyeSync")
    root.resizable(False, False)
    frame = ttk.Frame(root, padding=22)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text=f"EyeSync {VERSION}", font=("Segoe UI", 16, "bold")).pack(anchor="w")
    updating = (dest / "EyeSync.exe").exists() or (dest / "live_inference.py").exists()
    info = tk.StringVar(value=(
        "Move the mouse cursor with your head, and click by blinking.\n\n"
        + ("EyeSync is already installed: this updates it, and your settings and "
           "calibration are kept.\n\n" if updating else "")
        + f"EyeSync will be installed in:\n{dest}\n\nNo administrator rights are needed."))
    ttk.Label(frame, textvariable=info, wraplength=430, justify="left").pack(anchor="w", pady=(10, 14))
    bar = ttk.Progressbar(frame, length=430, maximum=1.0)
    status = tk.StringVar(value="")
    status_label = ttk.Label(frame, textvariable=status)
    open_site = tk.BooleanVar(value=True)
    site_box = ttk.Checkbutton(frame, text="Open the EyeSync website now", variable=open_site)
    buttons = ttk.Frame(frame)
    buttons.pack(fill="x", side="bottom")
    result = {"code": 1}

    def finish():
        if open_site.get():
            webbrowser.open(SITE)
        result["code"] = 0
        root.destroy()

    def progress(text, frac):
        status.set(text)
        bar.configure(value=frac)

    def done(error: BaseException | None):
        if error is not None:
            messagebox.showerror("EyeSync", f"EyeSync could not be installed:\n\n{error}\n\n"
                                 f"Details are in {LOG_PATH}", parent=root)
            action_btn.configure(state="normal", text="Try again")
            cancel_btn.configure(state="normal")
            return
        bar.pack_forget()
        status_label.pack_forget()
        info.set("EyeSync is installed.\n\nIt starts in the background (the first time takes "
                 "about half a minute) and switches on when you press Connect on the EyeSync "
                 "website's Dashboard. Tip: press Calibrate in EyeSync (30 seconds) to fit it "
                 "to you.\n\nYou can delete the downloaded EyeSync.exe now.")
        site_box.pack(anchor="w", pady=(0, 10))
        action_btn.configure(text="Finish", state="normal", command=finish)
        cancel_btn.pack_forget()

    def start():
        action_btn.configure(state="disabled")
        cancel_btn.configure(state="disabled")
        bar.pack(anchor="w")
        status_label.pack(anchor="w", pady=(4, 10))

        def work():
            try:
                install(dest, True, lambda t, f: root.after(0, progress, t, f))
                root.after(0, done, None)
            except BaseException as exc:            # noqa: BLE001 - shown to the user
                log.error(traceback.format_exc())
                root.after(0, done, exc)
        threading.Thread(target=work, daemon=True).start()

    cancel_btn = ttk.Button(buttons, text="Cancel", command=root.destroy)
    cancel_btn.pack(side="right")
    action_btn = ttk.Button(buttons, text="Update" if updating else "Install", command=start)
    action_btn.pack(side="right", padx=(0, 8))
    action_btn.focus_set()
    root.bind("<Return>", lambda _e: action_btn.invoke())
    root.update_idletasks()
    w, h = max(root.winfo_reqwidth(), 480), max(root.winfo_reqheight(), 300)
    root.geometry(f"{w}x{h}+{(root.winfo_screenwidth() - w) // 2}+{(root.winfo_screenheight() - h) // 3}")
    root.mainloop()
    return result["code"]


def main(argv: list[str]) -> int:
    """EyeSync.exe opened from outside its install folder, or --install [options]."""
    import argparse
    p = argparse.ArgumentParser(prog="EyeSync.exe --install", description="Install or update EyeSync")
    p.add_argument("--quiet", action="store_true", help="no window, no questions, no website")
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="install folder")
    p.add_argument("--no-register", action="store_true",
                   help="only copy: no shortcuts, login start, links or Settings > Apps entry")
    a = p.parse_args(argv)
    logging.basicConfig(filename=LOG_PATH, level=logging.INFO, encoding="utf-8",
                        format="%(asctime)s %(levelname)s %(message)s")
    if not a.quiet:
        return run_window(a.dest)
    try:
        install(a.dest, not a.no_register)
        return 0
    except BaseException:                            # noqa: BLE001 - logged for the caller
        log.error(traceback.format_exc())
        return 1
