"""Install EyeSync from a bundled ZIP and create a shortcut.

The downloaded setup executable is built by build_setup.py. It installs only
this standalone edition and never touches the original EyeSync application.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

VERSION = "1.0.6"
PAYLOAD_NAME = f"EyeSync-Standalone-Windows-v{VERSION}.zip"
SHORTCUT_NAME = "EyeSync.lnk"
DEFAULT_DEST = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "Programs" / "EyeSync Standalone"


def desktop_folder() -> Path:
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
         "[Environment]::GetFolderPath('Desktop')"],
        capture_output=True, text=True, check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    folder = result.stdout.strip()
    if not folder:
        raise OSError("Windows did not report a Desktop folder for this user.")
    return Path(folder)


def bundled_payload() -> Path:
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return root / PAYLOAD_NAME


def install(payload: Path, dest: Path, shortcut_dir: Path) -> Path:
    """Extract app files while preserving a user's calibration beside them."""
    dest.mkdir(parents=True, exist_ok=True)
    base = dest.resolve()
    with zipfile.ZipFile(payload) as archive:
        for item in archive.infolist():
            parts = Path(item.filename).parts
            if not parts or parts[0] != "EyeSync":
                raise ValueError("The EyeSync package has an unexpected file path.")
            relative = Path(*parts[1:])
            if not relative.parts:
                continue
            target = (dest / relative).resolve()
            if not target.is_relative_to(base):
                raise ValueError("The EyeSync package contains an unsafe file path.")
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, target.open("wb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
    launcher = dest / "Start EyeSync.bat"
    if not launcher.is_file() or not (dest / "EyeSync.exe").is_file():
        raise ValueError("The EyeSync package is incomplete.")
    shortcut_dir.mkdir(parents=True, exist_ok=True)
    (shortcut_dir / "EyeSync Standalone.lnk").unlink(missing_ok=True)
    shortcut = shortcut_dir / SHORTCUT_NAME
    script = (
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:ES_LNK); "
        "$s.TargetPath = $env:ES_TARGET; "
        "$s.WorkingDirectory = $env:ES_DIR; "
        "$s.IconLocation = $env:ES_ICON; "
        "$s.Description = 'Launch EyeSync'; $s.Save()"
    )
    env = {**os.environ, "ES_LNK": str(shortcut), "ES_TARGET": str(launcher),
           "ES_DIR": str(dest), "ES_ICON": str(dest / "EyeSync.exe")}
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        env=env, capture_output=True, text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode or not shortcut.is_file():
        raise OSError(f"Windows could not create the EyeSync shortcut: {result.stderr.strip()}")
    return shortcut


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install EyeSync")
    parser.add_argument("--quiet", action="store_true", help="install without a window")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--shortcut-dir", type=Path)
    parser.add_argument("--payload", type=Path, default=bundled_payload())
    args = parser.parse_args(argv)
    if sys.platform != "win32":
        raise OSError("EyeSync setup requires Windows.")
    shortcut_dir = args.shortcut_dir or desktop_folder()
    if args.quiet:
        shortcut = install(args.payload, args.dest, shortcut_dir)
        print(shortcut)
        return 0

    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("Install EyeSync")
    root.geometry("560x340")
    root.resizable(False, False)
    frame = ttk.Frame(root, padding=24)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Install EyeSync", font=("Segoe UI", 19, "bold")).pack(anchor="w")
    ttk.Label(frame, text="A webcam, microphone, and Windows 10 or 11 are required.",
              wraplength=500).pack(anchor="w", pady=(8, 18))
    folder = tk.StringVar(value=str(shortcut_dir))
    ttk.Label(frame, text="Shortcut folder", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(16, 4))
    row = ttk.Frame(frame)
    row.pack(fill="x")
    ttk.Entry(row, textvariable=folder, state="readonly").pack(side="left", fill="x", expand=True)

    def browse() -> None:
        choice = filedialog.askdirectory(initialdir=folder.get(), title="Choose shortcut folder")
        if choice:
            folder.set(choice)

    browse_button = ttk.Button(row, text="Browse…", command=browse)
    browse_button.pack(side="left", padx=(8, 0))
    status = tk.StringVar(value="Ready to install. No Python installation or administrator access is needed.")
    ttk.Label(frame, textvariable=status, wraplength=500).pack(anchor="w", pady=(18, 10))

    def run_install() -> None:
        selected_shortcut_dir = Path(folder.get())
        install_button.state(["disabled"])
        browse_button.state(["disabled"])
        status.set("Installing EyeSync and creating your shortcut…")

        def worker() -> None:
            try:
                shortcut = install(args.payload, args.dest, selected_shortcut_dir)
            except Exception as exc:
                message = str(exc)
                root.after(0, lambda: failed(message))
            else:
                root.after(0, lambda: succeeded(shortcut))

        threading.Thread(target=worker, daemon=True).start()

    def failed(message: str) -> None:
        status.set("Installation failed. Close EyeSync if it is running, then try again.")
        install_button.state(["!disabled"])
        browse_button.state(["!disabled"])
        messagebox.showerror("EyeSync setup", message)

    def succeeded(shortcut: Path) -> None:
        status.set(f"Installed. Click the shortcut to start EyeSync:\n{shortcut}")
        install_button.configure(text="Finish", command=root.destroy)
        install_button.state(["!disabled"])

    install_button = ttk.Button(frame, text="Install", command=run_install)
    install_button.pack(anchor="e", pady=(8, 0))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
