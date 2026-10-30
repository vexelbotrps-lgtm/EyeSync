"""Offline speech commands and Unicode keyboard input for Windows."""
from __future__ import annotations

import ctypes
import logging
import re
import sys
from ctypes import wintypes
from pathlib import Path
from typing import Optional

log = logging.getLogger("gaze.voice")

import app_paths

_SCRIPT_PATH = app_paths.APP_DIR / "control_backend" / "voice_commands.ps1"
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


def press_virtual_key(code: int) -> None:
    """Press and release a non-text key in the currently focused window."""
    _require_windows()
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SendInput.argtypes = (
        wintypes.UINT, ctypes.POINTER(_Input), ctypes.c_int
    )
    user32.SendInput.restype = wintypes.UINT
    events = (_Input * 2)()
    events[0].type = events[1].type = 1
    events[0].ki.wVk = events[1].ki.wVk = code
    events[1].ki.dwFlags = _KEYEVENTF_KEYUP
    sent = user32.SendInput(2, events, ctypes.sizeof(_Input))
    if sent != 2:
        error = ctypes.get_last_error()
        raise ctypes.WinError(error) if error else OSError(
            f"Windows accepted {sent} of 2 key input events."
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
