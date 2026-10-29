"""
autostart.py - Make EyeSync wait for the website at every login.
=============================================================================
(EyeSync.exe: the same with ``EyeSync.exe --autostart <action>``; it also
lists EyeSync in Settings > Apps, see register_app.)

    python autostart.py install     start at every Windows login (and now, with --now)
    python autostart.py uninstall   stop starting at login
    python autostart.py status      is it installed / running?
    python autostart.py start       start EyeSync now (or show its window)
    python autostart.py stop        stop EyeSync now
    python autostart.py shortcuts   add "EyeSync" to the Desktop and Start menu
                                    (uninstall removes them again)

Installed, EyeSync waits invisibly at login (no window, no camera, no AI:
EyeSync.pyw --standby) and opens only when the EyeSync website dashboard
is opened; it closes again after the dashboard is closed. The window's
"Start with Windows"
checkbox does the same as install/uninstall. Installation writes one
value under HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Run
(current user only, no admin rights needed); uninstall removes it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import app_paths

HERE = app_paths.DATA_DIR        # the project folder, or where EyeSync.exe is installed
# EyeSync.exe to register: this program, unless the installer (a downloaded
# EyeSync.exe, see installer.py) sets the installed copy here first.
EXE: Path | None = None
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME = "EyeSync"
# Settings > Apps lists EyeSync (EyeSync.exe only), with its Uninstall button.
UNINSTALL_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\EyeSync"


def pythonw() -> Path:
    """Windowless interpreter, preferring the project's virtual environment."""
    venv = HERE / ".venv" / "Scripts" / "pythonw.exe"
    if venv.exists():
        return venv
    candidate = Path(sys.executable).with_name("pythonw.exe")
    return candidate if candidate.exists() else Path(sys.executable)


def exe() -> Path:
    """The EyeSync.exe being set up (frozen only)."""
    return EXE or Path(sys.executable)


def home() -> Path:
    """The folder EyeSync runs in: the project, or EyeSync.exe's install folder."""
    return exe().parent if app_paths.FROZEN else HERE


def launch_parts(standby: bool = False) -> tuple[str, str]:
    """(program, arguments) that start EyeSync - EyeSync.exe, or EyeSync.pyw."""
    if app_paths.FROZEN:
        return str(exe()), "--standby" if standby else ""
    return str(pythonw()), f'"{HERE / "EyeSync.pyw"}"' + (" --standby" if standby else "")


def command(standby: bool = False) -> str:
    program, arguments = launch_parts(standby)
    return f'"{program}"' + (f" {arguments}" if arguments else "")


# Windows' own on/off switch for login programs (Settings > Apps > Startup,
# Task Manager > Startup apps). It overrides the Run entry: a value whose
# first byte is odd (03) means "turned off".
APPROVED_KEY = r"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run"


def turned_off_in_windows() -> bool:
    """The user switched EyeSync off in Windows' Startup apps."""
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, APPROVED_KEY) as k:
            value = winreg.QueryValueEx(k, VALUE_NAME)[0]
    except OSError:
        return False
    return isinstance(value, bytes) and len(value) > 0 and value[0] & 1 == 1


def enabled() -> bool:
    """EyeSync really starts at login: installed, and not switched off in Windows."""
    return installed_command() is not None and not turned_off_in_windows()


def enable() -> None:
    """At every Windows login, wait for the website dashboard (see the module docstring)."""
    import winreg
    with _key(True) as k:
        winreg.SetValueEx(k, VALUE_NAME, 0, winreg.REG_SZ, command(standby=True))
    if turned_off_in_windows():
        # The user asked for it (EyeSync's checkbox, install.bat): switch it
        # back on in Windows too, as Task Manager's "Enable" does.
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, APPROVED_KEY, 0,
                                winreg.KEY_SET_VALUE) as k:
            winreg.SetValueEx(k, VALUE_NAME, 0, winreg.REG_BINARY, bytes([2] + [0] * 11))


def disable() -> None:
    import winreg
    try:
        with _key(True) as k:
            winreg.DeleteValue(k, VALUE_NAME)
    except FileNotFoundError:
        pass


def _key(write: bool):
    import winreg
    access = winreg.KEY_SET_VALUE | winreg.KEY_QUERY_VALUE if write else winreg.KEY_QUERY_VALUE
    return winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, access)


def installed_command() -> str | None:
    import winreg
    try:
        with _key(False) as k:
            return winreg.QueryValueEx(k, VALUE_NAME)[0]
    except FileNotFoundError:
        return None


def running() -> bool:
    """True if the live tracker (background or manual) is running."""
    import ctypes
    SYNCHRONIZE = 0x00100000
    h = ctypes.windll.kernel32.OpenMutexW(SYNCHRONIZE, False, "Local\\EyeSync.LiveInference")
    if h:
        ctypes.windll.kernel32.CloseHandle(h)
        return True
    return False


def start(standby: bool = False) -> None:
    """Open EyeSync now - or, with ``standby``, only start waiting for the dashboard."""
    if running():
        import desktop
        desktop.bring_window_to_front("EyeSync", "TkTopLevel")
        print("EyeSync is already running - its window was brought to the front.")
        return
    subprocess.Popen(command(standby), cwd=str(home()), creationflags=subprocess.DETACHED_PROCESS
                     | subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
    print("EyeSync is waiting - it opens when the EyeSync website dashboard is opened."
          if standby else "EyeSync started - its window opens in about 15 seconds.")


def stop() -> None:
    """
    End the background runner and the tracker, run from source (found by
    their command lines) or as EyeSync.exe - never the process asking.
    """
    ps = (f"$me = @({os.getpid()}, {os.getppid()}); "
          "Get-CimInstance Win32_Process | Where-Object { "
          "(($_.CommandLine -match 'EyeSync.pyw|live_inference.py' -and $_.Name -match 'python') "
          "-or $_.Name -eq 'EyeSync.exe') -and $me -notcontains $_.ProcessId } | "
          "ForEach-Object { Stop-Process -Id $_.ProcessId -Force; $_.ProcessId }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True,
                         text=True, creationflags=subprocess.CREATE_NO_WINDOW).stdout.split()
    print(f"Stopped {len(out)} EyeSync process(es)." if out else "EyeSync was not running.")


def register_app(version: str) -> None:
    """EyeSync.exe: list EyeSync in Settings > Apps, whose Uninstall runs EyeSync.exe --uninstall."""
    import winreg
    exe_path = exe()
    size_kb = exe_path.stat().st_size // 1024
    values = {"DisplayName": "EyeSync", "DisplayVersion": version, "Publisher": "EyeSync",
              "DisplayIcon": f"{exe_path},0", "InstallLocation": str(exe_path.parent),
              "UninstallString": f'"{exe_path}" --uninstall',
              "URLInfoAbout": "https://aniruddh9140.github.io/face-control-website/"}
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY, 0, winreg.KEY_SET_VALUE) as k:
        for name, value in values.items():
            winreg.SetValueEx(k, name, 0, winreg.REG_SZ, value)
        for name, value in (("NoModify", 1), ("NoRepair", 1), ("EstimatedSize", size_kb)):
            winreg.SetValueEx(k, name, 0, winreg.REG_DWORD, value)


URL_KEY = r"Software\Classes\eyesync"


def register_url() -> None:
    """
    eyesync: links open EyeSync - the website's "Open EyeSync" button when
    nothing is waiting in the background (e.g. the login start was switched
    off). The browser asks the user first; EyeSync then starts for the
    website (camera still off until the website connects).
    """
    import winreg
    program, arguments = launch_parts(standby=False)
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, URL_KEY, 0, winreg.KEY_SET_VALUE) as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, "URL:EyeSync")
        winreg.SetValueEx(k, "URL Protocol", 0, winreg.REG_SZ, "")
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, URL_KEY + r"\DefaultIcon", 0,
                            winreg.KEY_SET_VALUE) as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ, f'"{make_icon() or program}"')
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, URL_KEY + r"\shell\open\command", 0,
                            winreg.KEY_SET_VALUE) as k:
        winreg.SetValueEx(k, "", 0, winreg.REG_SZ,
                          f'"{program}"' + (f" {arguments}" if arguments else "") + ' --url "%1"')


def unregister_url() -> None:
    import winreg
    for sub in (r"\shell\open\command", r"\shell\open", r"\shell", r"\DefaultIcon", ""):
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, URL_KEY + sub)
        except FileNotFoundError:
            pass


def unregister_app() -> None:
    import winreg
    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, UNINSTALL_KEY)
    except FileNotFoundError:
        pass


SHORTCUT_NAME = "EyeSync.lnk"
ICON_PATH = HERE / "models" / "eyesync.ico"


def _shortcut_dirs() -> list[Path]:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "[Environment]::GetFolderPath('Desktop'); "
                          "[Environment]::GetFolderPath('Programs')"],
                         capture_output=True, text=True,
                         creationflags=subprocess.CREATE_NO_WINDOW).stdout.splitlines()
    return [Path(line.strip()) for line in out if line.strip()]


def make_icon() -> Path | None:
    """The window's eye icon as a Windows .ico (needs Pillow); EyeSync.exe has it built in."""
    if app_paths.FROZEN:
        return exe()
    try:
        import main_window
        img = main_window.icon_image()
        ICON_PATH.parent.mkdir(exist_ok=True)
        img.save(ICON_PATH, sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])
        return ICON_PATH
    except Exception:                       # noqa: BLE001 - shortcut works without an icon
        return None


def add_shortcuts() -> None:
    """Desktop + Start menu shortcuts that start EyeSync in the background."""
    icon = make_icon()
    for folder in _shortcut_dirs():
        lnk = folder / SHORTCUT_NAME
        ps = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:ES_LNK); "
              "$s.TargetPath = $env:ES_TARGET; $s.Arguments = $env:ES_ARGS; "
              "$s.WorkingDirectory = $env:ES_DIR; "
              "$s.Description = 'Control the mouse with your head'; "
              "if ($env:ES_ICON) { $s.IconLocation = $env:ES_ICON }; $s.Save()")
        program, arguments = launch_parts(standby=False)
        env = {**os.environ, "ES_LNK": str(lnk), "ES_TARGET": program, "ES_ARGS": arguments,
               "ES_DIR": str(home()), "ES_ICON": str(icon) if icon else ""}
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], env=env,
                           capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        print(f"Shortcut: {lnk}" if r.returncode == 0 else f"Could not create {lnk}: {r.stderr}")


def remove_shortcuts() -> None:
    for folder in _shortcut_dirs():
        lnk = folder / SHORTCUT_NAME
        if lnk.exists():
            lnk.unlink()
            print(f"Removed {lnk}")


def main() -> int:
    if sys.platform != "win32":
        print("autostart.py supports Windows only.")
        return 1
    p = argparse.ArgumentParser(description="EyeSync autostart at login")
    p.add_argument("action", choices=("install", "uninstall", "status", "start", "stop",
                                      "shortcuts"))
    p.add_argument("--now", action="store_true", help="with install: also start it right away")
    p.add_argument("--standby", action="store_true",
                   help="with start: don't open yet, wait for the website dashboard")
    a = p.parse_args()

    if a.action == "install":
        enable()
        register_url()
        print("Installed: at every login EyeSync waits for the website dashboard and opens "
              f"with it.\n  {command(True)}")
        if a.now:
            start(standby=True)
    elif a.action == "uninstall":
        disable()
        unregister_url()
        print("Uninstalled: EyeSync no longer starts at login.")
        remove_shortcuts()
    elif a.action == "status":
        cmd = installed_command()
        off = " - but turned OFF in Windows' Startup apps" if cmd and turned_off_in_windows() else ""
        print(f"Autostart: {'INSTALLED' if cmd else 'not installed'}{off}"
              + (f"\n  {cmd}" if cmd else ""))
        print(f"Running  : {'yes' if running() else 'no'}")
    elif a.action == "start":
        start(a.standby)
    elif a.action == "stop":
        stop()
    elif a.action == "shortcuts":
        add_shortcuts()
    return 0


if __name__ == "__main__":
    sys.exit(main())
