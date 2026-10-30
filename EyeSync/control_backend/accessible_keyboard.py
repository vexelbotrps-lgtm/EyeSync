"""EyeSync's movable on-screen keyboard, kept at the app's integrity level.

Windows runs osk.exe with UIAccess, so a portable app cannot resize or close its
window. This keyboard belongs to EyeSync and uses ordinary desktop input.
"""
from __future__ import annotations

import ctypes
import tkinter as tk
from ctypes import wintypes
from tkinter import ttk
from typing import Callable

from control_backend import voice_commands as vc


class AccessibleKeyboard:
    MIN_WIDTH = 500
    MIN_HEIGHT = 190
    MAX_WIDTH = 1800
    MAX_HEIGHT = 760

    def __init__(self, parent: tk.Tk,
                 status: Callable[[str], None] | None = None) -> None:
        self.parent = parent
        self.status = status or (lambda _message: None)
        self.window: tk.Toplevel | None = None
        self.target = 0
        self.moving = False
        self._move_offset = (0, 0)
        self.shift = False
        self.caps = False

    def _user32(self):
        return ctypes.windll.user32

    def _own_window(self, hwnd: int) -> bool:
        if not hwnd:
            return True
        user32 = self._user32()
        if hwnd == user32.GetAncestor(self.parent.winfo_id(), 2):
            return True
        return (self.window is not None
                and hwnd == user32.GetAncestor(self.window.winfo_id(), 2))

    def _remember_target(self) -> None:
        if self.window is None:
            return
        hwnd = int(self._user32().GetForegroundWindow())
        if not self._own_window(hwnd):
            self.target = hwnd
        self.parent.after(100, self._remember_target)

    def open(self) -> bool:
        if self.window is not None:
            self.window.deiconify()
            self.window.lift()
            return True
        current = int(self._user32().GetForegroundWindow())
        if not self._own_window(current):
            self.target = current
        window = tk.Toplevel(self.parent)
        self.window = window
        window.title("EyeSync On-Screen Keyboard")
        window.attributes("-topmost", True)
        x = max(0, (window.winfo_screenwidth() - 940) // 2)
        y = max(0, window.winfo_screenheight() - 360)
        window.geometry(f"940x320+{x}+{y}")
        window.minsize(self.MIN_WIDTH, self.MIN_HEIGHT)
        window.maxsize(self.MAX_WIDTH, self.MAX_HEIGHT)
        window.protocol("WM_DELETE_WINDOW", self.close)

        body = ttk.Frame(window, padding=6)
        body.pack(fill="both", expand=True)
        rows = (
            "1234567890-=",
            "qwertyuiop[]\\",
            "asdfghjkl;'",
            "zxcvbnm,./",
        )
        for chars in rows:
            row = ttk.Frame(body)
            row.pack(fill="both", expand=True, pady=1)
            for char in chars:
                ttk.Button(row, text=char.upper(),
                           command=lambda value=char: self._type(value)).pack(
                    side="left", fill="both", expand=True, padx=1)

        row = ttk.Frame(body)
        row.pack(fill="both", expand=True, pady=1)
        for label, action, width in (
            ("Shift", self._toggle_shift, 1),
            ("Caps", self._toggle_caps, 1),
            ("Tab", lambda: self._key(0x09), 1),
            ("Space", lambda: self._type(" "), 4),
            ("Backspace", lambda: self._key(0x08), 2),
            ("Enter", lambda: self._key(0x0D), 2),
        ):
            ttk.Button(row, text=label, command=action).pack(
                side="left", fill="both", expand=True, padx=1, ipadx=8 * width)
        window.update_idletasks()
        # Keep the destination application focused when a key is clicked.
        user32 = self._user32()
        hwnd = user32.GetAncestor(window.winfo_id(), 2)
        ex_style = user32.GetWindowLongPtrW(hwnd, -20)
        user32.SetWindowLongPtrW(hwnd, -20, ex_style | 0x08000000)  # WS_EX_NOACTIVATE
        if not user32.GetWindowLongPtrW(hwnd, -20) & 0x08000000:
            window.destroy()
            self.window = None
            raise OSError("Windows did not allow the keyboard to keep the text field focused.")
        user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0,
                            0x0001 | 0x0002 | 0x0010 | 0x0020)  # frame changed
        self.parent.after(100, self._remember_target)
        return True

    def close(self) -> bool:
        if self.window is None:
            return False
        self.moving = False
        self.window.destroy()
        self.window = None
        return True

    def resize(self, scale: float) -> bool:
        if self.window is None:
            return False
        window = self.window
        window.update_idletasks()
        width = max(self.MIN_WIDTH, min(self.MAX_WIDTH, round(window.winfo_width() * scale)))
        height = max(self.MIN_HEIGHT, min(self.MAX_HEIGHT, round(window.winfo_height() * scale)))
        window.geometry(f"{width}x{height}+{window.winfo_x()}+{window.winfo_y()}")
        return True

    def start_moving(self) -> bool:
        if self.window is None:
            return False
        point = self._cursor()
        self._move_offset = (self.window.winfo_x() - point[0],
                             self.window.winfo_y() - point[1])
        self.moving = True
        self._follow_cursor()
        return True

    def stop_moving(self) -> None:
        self.moving = False

    def _cursor(self) -> tuple[int, int]:
        point = wintypes.POINT()
        if not self._user32().GetCursorPos(ctypes.byref(point)):
            raise ctypes.WinError(ctypes.get_last_error())
        return point.x, point.y

    def _follow_cursor(self) -> None:
        if not self.moving or self.window is None:
            return
        x, y = self._cursor()
        self.window.geometry(f"+{x + self._move_offset[0]}+{y + self._move_offset[1]}")
        self.parent.after(40, self._follow_cursor)

    def _focus_target(self) -> bool:
        current = int(self._user32().GetForegroundWindow())
        if not self._own_window(current):
            self.target = current
        if not self.target or not self._user32().IsWindow(self.target):
            self.status("Click the text field you want to type in, then press a key.")
            return False
        focused = (current == self.target
                   or bool(self._user32().SetForegroundWindow(self.target)))
        if not focused:
            self.status("Windows could not focus that text field. Click it again.")
        return focused

    def _type(self, char: str) -> None:
        if not self._focus_target():
            return
        if char.isalpha() and (self.shift != self.caps):
            char = char.upper()
        elif self.shift:
            shifted = dict(zip("1234567890-=[]\\;',./", "!@#$%^&*()_+{}|:\"<>?"))
            char = shifted.get(char, char)
        try:
            vc.type_text(char)
        except OSError as exc:
            self.status(f"Could not type into that window: {exc}")
        self.shift = False

    def _key(self, code: int) -> None:
        if self._focus_target():
            try:
                vc.press_virtual_key(code)
            except OSError as exc:
                self.status(f"Could not type into that window: {exc}")
        self.shift = False

    def _toggle_shift(self) -> None:
        self.shift = not self.shift

    def _toggle_caps(self) -> None:
        self.caps = not self.caps
