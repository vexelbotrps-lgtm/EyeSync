"""Offline Windows speech commands, OSK controls, and Unicode text entry."""
from __future__ import annotations

import ctypes
import logging
import os
import re
import sys
from ctypes import wintypes
from pathlib import Path
from typing import Optional

log = logging.getLogger("gaze.voice")

import app_paths

_SCRIPT_PATH = app_paths.APP_DIR / "control_backend" / "voice_commands.ps1"
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_WM_CLOSE = 0x0010
_SWP_NOZORDER = 0x0004
_SWP_NOACTIVATE = 0x0010
_SW_SHOWNORMAL = 1
_VK_TAB = 0x09
_KEYEVENTF_KEYUP = 0x0002
_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion", "trillion", "first",
    "second", "third",
}
_MATH_WORDS = {
    "plus", "minus", "times", "multiplied", "multiply", "divided",
    "divide", "over", "add", "subtract", "equals", "equal", "squared",
    "cubed", "decimal", "point", "negative", "positive",
}

_COMMANDS = {
    "open the on screen keyboard": "open_keyboard",
    "close the on screen keyboard": "close_keyboard",
    "increase the size of the on screen keyboard": "increase_keyboard",
    "decrease the size of the on screen keyboard": "reduce_keyboard",
    "let me adjust the position of the on screen keyboard": "start_adjusting",
    "done adjusting the position of the on screen keyboard": "stop_adjusting",
}
_WISPR_ACTIONS = {
    "open_keyboard", "close_keyboard", "start_adjusting", "stop_adjusting",
    "increase_keyboard", "reduce_keyboard",
}


def normalize_phrase(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def command_action(text: str) -> Optional[str]:
    return _COMMANDS.get(normalize_phrase(text))


def wispr_command_action(text: str) -> Optional[str]:
    action = command_action(text)
    return action if action in _WISPR_ACTIONS else None


def contains_numeric_expression(text: str) -> bool:
    tokens = set(re.findall(r"[a-z]+|\d+(?:[.,]\d+)*", text.casefold()))
    return any(
        re.fullmatch(r"\d+(?:[.,]\d+)*", token)
        or token in _NUMBER_WORDS
        or token in _MATH_WORDS
        for token in tokens
    )


def speech_script_path() -> Path:
    return _SCRIPT_PATH


def _require_windows() -> None:
    if sys.platform != "win32":
        raise OSError("Voice commands and the on-screen keyboard require Windows.")


def _get_osk_window() -> int:
    _require_windows()
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.EnumWindows.argtypes = (
        ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM),
        wintypes.LPARAM,
    )
    user32.EnumWindows.restype = wintypes.BOOL
    user32.GetWindowThreadProcessId.argtypes = (
        wintypes.HWND, ctypes.POINTER(wintypes.DWORD)
    )
    kernel32.OpenProcess.argtypes = (
        wintypes.DWORD, wintypes.BOOL, wintypes.DWORD
    )
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = (
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

    found: list[int] = []
    callback_type = ctypes.WINFUNCTYPE(
        wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
    )

    def inspect(hwnd, _parameter):
        process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        process = kernel32.OpenProcess(
            _PROCESS_QUERY_LIMITED_INFORMATION, False, process_id.value
        )
        if not process:
            return True
        try:
            image_path = ctypes.create_unicode_buffer(32768)
            length = wintypes.DWORD(len(image_path))
            if (kernel32.QueryFullProcessImageNameW(
                    process, 0, image_path, ctypes.byref(length))
                    and Path(image_path.value).name.casefold() == "osk.exe"):
                found.append(int(hwnd))
                return False
        finally:
            kernel32.CloseHandle(process)
        return True

    callback = callback_type(inspect)
    enumerated = user32.EnumWindows(callback, 0)
    if not enumerated and not found:
        error = ctypes.get_last_error()
        if error:
            raise ctypes.WinError(error)
    return found[0] if found else 0


def open_onscreen_keyboard() -> bool:
    _require_windows()
    if _get_osk_window():
        return True
    windows_dir = os.environ.get("WINDIR")
    if not windows_dir:
        raise OSError("Cannot open the on-screen keyboard: WINDIR is not set.")
    keyboard = Path(windows_dir) / "System32" / "osk.exe"
    if not keyboard.is_file():
        raise FileNotFoundError(
            f"Windows on-screen keyboard was not found at {keyboard}."
        )
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    shell32.ShellExecuteW.argtypes = (
        wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR,
        wintypes.LPCWSTR, ctypes.c_int,
    )
    shell32.ShellExecuteW.restype = ctypes.c_void_p
    result = shell32.ShellExecuteW(
        None, "runas", str(keyboard), None, None, _SW_SHOWNORMAL
    )
    if result is None or result <= 32:
        raise OSError(
            "Windows could not start the on-screen keyboard "
            f"(ShellExecute error {result})."
        )
    return True


def close_onscreen_keyboard() -> bool:
    hwnd = _get_osk_window()
    if not hwnd:
        return False
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.PostMessageW.argtypes = (
        wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
    )
    user32.PostMessageW.restype = wintypes.BOOL
    if not user32.PostMessageW(hwnd, _WM_CLOSE, 0, 0):
        error = ctypes.get_last_error()
        raise ctypes.WinError(error) if error else OSError(
            "Windows did not close the on-screen keyboard."
        )
    return True


def is_keyboard_titlebar_cursor_position() -> bool:
    hwnd = _get_osk_window()
    if not hwnd:
        return False
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetWindowRect.argtypes = (
        wintypes.HWND, ctypes.POINTER(wintypes.RECT)
    )
    user32.GetWindowRect.restype = wintypes.BOOL
    user32.GetCursorPos.argtypes = (ctypes.POINTER(wintypes.POINT),)
    user32.GetCursorPos.restype = wintypes.BOOL

    rect = wintypes.RECT()
    point = wintypes.POINT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        error = ctypes.get_last_error()
        raise ctypes.WinError(error) if error else OSError(
            "Cannot read the on-screen keyboard position."
        )
    if not user32.GetCursorPos(ctypes.byref(point)):
        error = ctypes.get_last_error()
        raise ctypes.WinError(error) if error else OSError(
            "Cannot read the pointer position."
        )
    titlebar_height = max(32, (rect.bottom - rect.top) // 12)
    return (
        rect.left <= point.x <= rect.right
        and rect.top <= point.y <= rect.top + titlebar_height
    )


def resize_onscreen_keyboard(scale: float) -> bool:
    if not 0.5 <= scale <= 2.0:
        raise ValueError("Keyboard resize factor must be between 0.5 and 2.0.")
    hwnd = _get_osk_window()
    if not hwnd:
        return False
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetWindowRect.argtypes = (
        wintypes.HWND, ctypes.POINTER(wintypes.RECT)
    )
    user32.GetWindowRect.restype = wintypes.BOOL
    user32.SetWindowPos.argtypes = (
        wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, wintypes.UINT,
    )
    user32.SetWindowPos.restype = wintypes.BOOL

    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        error = ctypes.get_last_error()
        raise ctypes.WinError(error) if error else OSError(
            "Cannot read the on-screen keyboard size."
        )
    width = int((rect.right - rect.left) * scale)
    height = int((rect.bottom - rect.top) * scale)
    width = max(400, min(width, 1800))
    height = max(200, min(height, 1100))
    if not user32.SetWindowPos(
            hwnd, None, rect.left, rect.top, width, height,
            _SWP_NOZORDER | _SWP_NOACTIVATE):
        error = ctypes.get_last_error()
        raise ctypes.WinError(error) if error else OSError(
            "Windows did not resize the on-screen keyboard."
        )
    return True


class _KeyboardInput(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _InputUnion(ctypes.Union):
    _fields_ = [
        ("ki", _KeyboardInput),
        ("mi", _MouseInput),
    ]


class _Input(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = [("type", wintypes.DWORD), ("value", _InputUnion)]


def type_text(text: str) -> None:
    _require_windows()
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SendInput.argtypes = (
        wintypes.UINT, ctypes.POINTER(_Input), ctypes.c_int
    )
    user32.SendInput.restype = wintypes.UINT
    units = text.encode("utf-16-le")
    events = (_Input * len(units))()
    for index in range(0, len(units), 2):
        code_unit = units[index] | (units[index + 1] << 8)
        down, up = events[index], events[index + 1]
        down.type = up.type = 1
        down.ki.wScan = up.ki.wScan = code_unit
        down.ki.dwFlags = 0x0004
        up.ki.dwFlags = 0x0004 | 0x0002
    if events:
        sent = user32.SendInput(
            len(events), events, ctypes.sizeof(_Input)
        )
        if sent != len(events):
            error = ctypes.get_last_error()
            raise ctypes.WinError(error) if error else OSError(
                f"Windows accepted {sent} of {len(events)} text input events."
            )


def set_tab_key_down(down: bool) -> None:
    """Send one Tab key-down or key-up event."""
    _require_windows()
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SendInput.argtypes = (
        wintypes.UINT, ctypes.POINTER(_Input), ctypes.c_int
    )
    user32.SendInput.restype = wintypes.UINT
    event = _Input()
    event.type = 1
    event.ki.wVk = _VK_TAB
    event.ki.dwFlags = 0 if down else _KEYEVENTF_KEYUP
    sent = user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(_Input))
    if sent != 1:
        error = ctypes.get_last_error()
        raise ctypes.WinError(error) if error else OSError(
            "Windows did not accept the Tab key event."
        )


def play_tab_confirmation() -> bool:
    """Play a short rising confirmation tone after the Tab key is pressed."""
    _require_windows()
    import winsound

    try:
        winsound.Beep(1480, 100)
    except (OSError, RuntimeError) as exc:
        log.warning("Tab was pressed, but its confirmation tone failed: %s", exc)
        return False
    return True
