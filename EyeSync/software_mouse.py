"""
software_mouse.py - Standalone cursor control (no ESP32 required).
==================================================================

When no XIAO ESP32-S3 is plugged in, live_inference.py can drive the
computer's own cursor directly through the operating system instead.

* Windows: user32.SetCursorPos + SendInput via ctypes (no extra packages).
* macOS / Linux: uses ``pynput`` if it is installed (pip install pynput);
  otherwise software control is reported as unavailable.

Limitation on Windows: a normal (non-admin) process cannot click inside
windows that run as administrator (UIPI). The ESP32 is a real USB mouse
and does not have this limitation.
"""

from __future__ import annotations

import ctypes
import logging
import sys

log = logging.getLogger("eyesync")


class SoftwareMouse:
    def __init__(self):
        self.available = False
        self._impl = None

        if sys.platform == "win32":
            self._init_windows()
        else:
            try:
                from pynput.mouse import Button, Controller  # optional dependency
                self._pynput = Controller()
                self._button = Button.left
                self._right_button = Button.right
                self._impl = "pynput"
                self.available = True
            except Exception as exc:
                log.warning(f"[mouse] Software mouse unavailable ({exc}); "
                            "install it with: pip install pynput")

    # -- Windows ------------------------------------------------------------
    def _init_windows(self):
        from ctypes import wintypes

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                        ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                        ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]

        class INPUT(ctypes.Structure):
            class _U(ctypes.Union):
                # Padding member makes the union as large as the biggest
                # INPUT variant so sizeof(INPUT) matches what Windows expects.
                _fields_ = [("mi", MOUSEINPUT), ("_pad", ctypes.c_byte * 32)]
            _anonymous_ = ("u",)
            _fields_ = [("type", wintypes.DWORD), ("u", _U)]

        self._INPUT, self._MOUSEINPUT = INPUT, MOUSEINPUT
        self._user32 = ctypes.windll.user32
        self._user32.SendInput.argtypes = (wintypes.UINT, ctypes.c_void_p, ctypes.c_int)
        self._impl = "win32"
        self.available = True

    def _send_buttons(self, *flags: int) -> None:
        INPUT_MOUSE = 0
        inputs = (self._INPUT * len(flags))()
        for i, flag in enumerate(flags):
            inputs[i].type = INPUT_MOUSE
            inputs[i].mi = self._MOUSEINPUT(0, 0, 0, flag, 0, 0)
        self._user32.SendInput(len(flags), ctypes.byref(inputs), ctypes.sizeof(self._INPUT))

    # -- public API ---------------------------------------------------------
    def move(self, x: int, y: int) -> None:
        """Move the cursor to absolute screen pixel (x, y)."""
        if self._impl == "win32":
            self._user32.SetCursorPos(int(x), int(y))
        elif self._impl == "pynput":
            self._pynput.position = (int(x), int(y))

    def click(self, right: bool = False) -> None:
        """Left (or right) click at the current cursor position."""
        if self._impl == "win32":
            if right:
                MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP = 0x0008, 0x0010
                self._send_buttons(MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP)
            else:
                MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP = 0x0002, 0x0004
                self._send_buttons(MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP)
        elif self._impl == "pynput":
            self._pynput.click(self._right_button if right else self._button)

    def scroll(self, units: int) -> None:
        """Turn the mouse wheel: + = up (away from you), in Windows' units (120 = one notch)."""
        if self._impl == "win32":
            MOUSEEVENTF_WHEEL = 0x0800
            inputs = (self._INPUT * 1)()
            inputs[0].type = 0                                   # INPUT_MOUSE
            # mouseData is a DWORD: a downward turn is a negative number as 32 bits.
            inputs[0].mi = self._MOUSEINPUT(0, 0, int(units) & 0xFFFFFFFF, MOUSEEVENTF_WHEEL, 0, 0)
            self._user32.SendInput(1, ctypes.byref(inputs), ctypes.sizeof(self._INPUT))
        elif self._impl == "pynput":
            self._pynput.scroll(0, units / 120)
