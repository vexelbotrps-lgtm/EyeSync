"""
desktop.py - OS integration: global hotkeys and background-running tweaks.
=========================================================================

Global hotkeys
--------------
OpenCV's cv2.waitKey() only sees keys while its window has focus, so the old
single-letter keys stopped working as soon as the preview was minimised or
another app was in front. GlobalHotkeys registers system-wide hotkeys with
the Win32 RegisterHotKey API instead, which fire no matter which window is
focused - including when EyeSync's own window is minimised or hidden.

All hotkeys use Ctrl+Alt+Shift + a function key. Practically no application
binds that combination, and function keys avoid the AltGr (= Ctrl+Alt)
character-typing shortcuts used by many non-US keyboard layouts. If another
program has already registered one of them, Windows refuses the
registration; we log it and leave the other program's shortcut untouched
instead of stealing it.

On macOS/Linux the same hotkeys are provided through ``pynput`` if installed.

Background running (Windows)
----------------------------
Windows 11 throttles processes whose windows are minimised or in the
background ("EcoQoS" / efficiency mode) and coarsens their timers, which
would drop the tracking frame rate and the 30 Hz output rate exactly when
you minimise the preview. keep_running_in_background() opts this process out
of that throttling and requests 1 ms timer resolution.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
import time
from typing import Callable

log = logging.getLogger("eyesync")

# Virtual-key codes for F5..F12
VK = {"F5": 0x74, "F6": 0x75, "F7": 0x76, "F8": 0x77, "F9": 0x78, "F10": 0x79, "F11": 0x7A,
      "F12": 0x7B}
HOTKEY_PREFIX = "Ctrl+Alt+Shift+"


class GlobalHotkeys(threading.Thread):
    """
    Registers {"F9": callback, ...} as Ctrl+Alt+Shift+<key> system hotkeys.
    Callbacks run on this thread, so they must be quick and thread-safe
    (setting flags/events is ideal).
    """

    def __init__(self, bindings: dict[str, Callable[[], None]]):
        super().__init__(name="hotkeys", daemon=True)
        self.bindings = bindings
        self.registered: list[str] = []
        self._ready = threading.Event()
        self._pynput_listener = None

    def start(self) -> None:
        super().start()
        self._ready.wait(2.0)   # so the caller can log what got registered

    def run(self) -> None:
        if sys.platform == "win32":
            self._run_win32()
        else:
            self._run_pynput()

    # -- Windows ------------------------------------------------------------
    def _run_win32(self) -> None:
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_NOREPEAT = 0x1, 0x2, 0x4, 0x4000
        WM_HOTKEY = 0x0312
        mods = MOD_CONTROL | MOD_ALT | MOD_SHIFT | MOD_NOREPEAT

        ids: dict[int, Callable[[], None]] = {}
        for i, (key, callback) in enumerate(self.bindings.items(), start=1):
            # RegisterHotKey with hWnd=NULL posts WM_HOTKEY to this thread's
            # message queue, which the GetMessage loop below receives.
            if user32.RegisterHotKey(None, i, mods, VK[key]):
                ids[i] = callback
                self.registered.append(key)
            else:
                log.warning(f"[hotkeys] {HOTKEY_PREFIX}{key} is already used by another "
                            "program; leaving it alone (that EyeSync command is unavailable)")
        self._ready.set()

        msg = wintypes.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == WM_HOTKEY and msg.wParam in ids:
                    try:
                        ids[msg.wParam]()
                    except Exception:
                        log.exception("[hotkeys] Hotkey handler failed")
        finally:
            for i in ids:
                user32.UnregisterHotKey(None, i)

    # -- macOS / Linux --------------------------------------------------------
    def _run_pynput(self) -> None:
        try:
            from pynput import keyboard
        except Exception as exc:
            log.warning(f"[hotkeys] Global hotkeys unavailable ({exc}); pip install pynput")
            self._ready.set()
            return
        mapping = {f"<ctrl>+<alt>+<shift>+<{k.lower()}>": cb for k, cb in self.bindings.items()}
        self._pynput_listener = keyboard.GlobalHotKeys(mapping)
        self.registered = list(self.bindings)
        self._ready.set()
        self._pynput_listener.run()

    def stop(self) -> None:
        if sys.platform == "win32" and self.native_id is not None:
            WM_QUIT = 0x0012
            ctypes.windll.user32.PostThreadMessageW(self.native_id, WM_QUIT, 0, 0)
        elif self._pynput_listener is not None:
            self._pynput_listener.stop()


def keep_running_in_background() -> None:
    """Opt out of Windows background throttling (no-op on other systems)."""
    if sys.platform != "win32":
        return
    kernel32 = ctypes.windll.kernel32

    class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
        _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong),
                    ("StateMask", ctypes.c_ulong)]

    # ControlMask = EXECUTION_SPEED | IGNORE_TIMER_RESOLUTION, StateMask = 0
    # means "never throttle my execution speed or my timers".
    ProcessPowerThrottling = 4
    state = PROCESS_POWER_THROTTLING_STATE(1, 0x1 | 0x4, 0)
    try:
        kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), ProcessPowerThrottling,
                                       ctypes.byref(state), ctypes.sizeof(state))
    except Exception as exc:  # older Windows versions lack the API
        log.debug(f"SetProcessInformation unavailable: {exc}")

    ABOVE_NORMAL_PRIORITY_CLASS = 0x8000
    kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS)

    # 1 ms timer resolution keeps the 30 Hz output loops on schedule.
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass
    log.info("[system] Background throttling disabled; priority above normal")


_instance_handles: list = []


def acquire_single_instance(name: str) -> bool:
    """
    Claim a system-wide named mutex. Returns False if another process already
    holds it (i.e. EyeSync is already running). The mutex is released
    automatically when this process exits, even if it crashes.
    """
    if sys.platform != "win32":
        import fcntl, tempfile, os
        f = open(os.path.join(tempfile.gettempdir(), f"{name}.lock"), "w")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        _instance_handles.append(f)
        return True
    ERROR_ALREADY_EXISTS = 183
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    handle = kernel32.CreateMutexW(None, False, "Local\\" + name)
    if not handle or kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
        return False
    _instance_handles.append(handle)   # keep it alive for the process lifetime
    return True


def cursor_position() -> tuple[int, int] | None:
    """Where the mouse cursor is now (screen pixels), or None if unknown."""
    if sys.platform != "win32":
        return None
    import ctypes.wintypes
    pt = ctypes.wintypes.POINT()
    if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
        return pt.x, pt.y
    return None


def is_running(name: str) -> bool:
    """Does another process hold the single-instance mutex ``name``?"""
    if sys.platform != "win32":
        return False
    SYNCHRONIZE = 0x00100000
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenMutexW.restype = ctypes.c_void_p
    handle = kernel32.OpenMutexW(SYNCHRONIZE, False, "Local\\" + name)
    if handle:
        kernel32.CloseHandle(ctypes.c_void_p(handle))
    return bool(handle)


class OpenSignal:
    """
    A named Windows event: opening the EyeSync shortcut while EyeSync is
    closed (its launcher waiting in standby) asks the launcher to start it.
    """

    NAME = "Local\\EyeSync.Open"

    def __init__(self):
        self.handle = None
        if sys.platform == "win32":
            kernel32 = ctypes.windll.kernel32
            kernel32.CreateEventW.restype = ctypes.c_void_p
            self.handle = kernel32.CreateEventW(None, False, False, self.NAME)   # auto-reset

    def clear(self) -> None:
        if self.handle:
            ctypes.windll.kernel32.ResetEvent(ctypes.c_void_p(self.handle))

    def is_set(self) -> bool:
        """True once after send() (it resets itself)."""
        WAIT_OBJECT_0 = 0
        return bool(self.handle) and ctypes.windll.kernel32.WaitForSingleObject(
            ctypes.c_void_p(self.handle), 0) == WAIT_OBJECT_0

    @classmethod
    def send(cls) -> bool:
        """Wake the waiting launcher. False if there is none."""
        if sys.platform != "win32":
            return False
        EVENT_MODIFY_STATE = 0x0002
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenEventW.restype = ctypes.c_void_p
        handle = kernel32.OpenEventW(EVENT_MODIFY_STATE, False, cls.NAME)
        if not handle:
            return False
        try:
            return bool(kernel32.SetEvent(ctypes.c_void_p(handle)))
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(handle))


def bring_window_to_front(title: str, window_class: str | None = None) -> None:
    """
    Raise a window above other apps (e.g. the EyeSync window when EyeSync
    is started a second time). ``window_class``
    narrows the match - "TkTopLevel" for the EyeSync window, so a browser
    tab that happens to be titled "EyeSync" is never picked.
    """
    if sys.platform != "win32":
        return
    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(window_class, title)
    if hwnd:
        SW_RESTORE = 9
        user32.ShowWindow(hwnd, SW_RESTORE)
        user32.SetForegroundWindow(hwnd)


def play_warning_sound() -> None:
    """Attention sound (e.g. the close gesture has started)."""
    if sys.platform == "win32":
        import winsound
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
    else:
        print("\a", end="", flush=True)


def play_voice_sound(on: bool) -> None:
    """Voice typing starts (rising tones) or stops (falling). Blocks ~0.2 s - call it
    before the microphone opens, so the sound isn't heard as speech."""
    if sys.platform == "win32":
        import winsound
        for freq in ((660, 990) if on else (990, 660)):
            try:
                winsound.Beep(freq, 90)
            except RuntimeError:                 # no sound device: the bar still shows it
                break
    else:
        print("\a", end="", flush=True)


def play_tick_sound(double: bool = False) -> None:
    """
    A tick while the eyes are still closed (non-blocking): opening them now
    clicks - one tick a left click, a double tick a right click. Higher than
    the click sounds that confirm a click afterwards. 60 ms: the 22 ms tick
    of 1.4.0-1.5.1 was too short for laptop speakers - the user's recording of
    2026-10-20 shows them waiting through it, opening only after the double.
    """
    if sys.platform == "win32":
        import winsound

        def ticks():
            for i in range(2 if double else 1):
                try:
                    winsound.Beep(1500, 60)
                except RuntimeError:             # no sound device
                    return
                if double and i == 0:
                    time.sleep(0.08)
        threading.Thread(target=ticks, daemon=True).start()
    else:
        print("\a", end="", flush=True)


def play_click_sound(right: bool = False) -> None:
    """Short sound confirming a click (non-blocking); a right click sounds different."""
    if sys.platform == "win32":
        import winsound
        if right:       # two quick high beeps, off this thread (Beep blocks)
            threading.Thread(target=lambda: [winsound.Beep(1400, 45) for _ in range(2)],
                             daemon=True).start()
        else:
            winsound.MessageBeep(winsound.MB_OK)
    else:
        print("\a", end="", flush=True)
