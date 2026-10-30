"""Local command field for executing allow-listed phrases dictated with Wispr Flow."""
from __future__ import annotations

import argparse
import ctypes
import queue
import threading
import tkinter as tk
from tkinter import ttk

from control_backend.accessible_keyboard import AccessibleKeyboard
from control_backend import voice_commands as vc


class WisprCommandPanel:
    DEBOUNCE_MS = 1200

    def __init__(self, root: tk.Tk, parent_pid: int = 0) -> None:
        self.root = root
        self.parent_pid = parent_pid
        self.commands: queue.Queue[str] = queue.Queue()
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()
        self.stopping = threading.Event()
        self.pending_after: str | None = None
        self.connected = True
        self.keyboard = AccessibleKeyboard(root, lambda message: self.status_text.set(message))
        self._app_icon = self.create_app_icon()

        root.title("EyeSync Standalone Command Panel")
        root.iconphoto(True, self._app_icon)
        root.protocol("WM_DELETE_WINDOW", self.close)

        frame = ttk.Frame(root, padding=16)
        frame.pack(fill="both", expand=True)
        gesture_frame = ttk.LabelFrame(frame, text="EyeSync voice", padding=10)
        gesture_frame.pack(fill="x", pady=(0, 12))
        ttk.Label(
            gesture_frame,
            text=(
                "EyeSync's offline voice typing stays in the main app. "
                "Use your speech tool to dictate one of the exact keyboard "
                "commands into the field below. The EyeSync keyboard can be "
                "moved and resized without Windows administrator permission."
            ),
            wraplength=540,
        ).pack(anchor="w")
        self.gesture_status = tk.StringVar(
            value="Voice typing remains available in EyeSync. Dictate a listed command below."
        )
        ttk.Label(
            gesture_frame, textvariable=self.gesture_status, wraplength=540
        ).pack(anchor="w", pady=(8, 0))

        ttk.Label(
            frame,
            text="Dictate an on-screen keyboard command",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            frame,
            text=(
                "Click the command field, dictate with your speech tool, then pause. Only an exact "
                "supported command runs; other dictated text is not acted on."
            ),
            wraplength=520,
        ).pack(anchor="w", pady=(8, 12))

        self.command_text = tk.StringVar()
        self.entry = ttk.Entry(frame, textvariable=self.command_text, font=("Segoe UI", 12))
        self.entry.pack(fill="x")
        self.entry.bind("<Return>", self.run_now)
        self.command_text.trace_add("write", self.schedule_command)

        ttk.Button(frame, text="Run command now", command=self.run_now).pack(
            anchor="e", pady=(8, 12)
        )
        ttk.Label(
            frame,
            text=(
                "Open the on-screen keyboard · Close the on-screen keyboard\n"
                "Increase the size of the on-screen keyboard\n"
                "Decrease the size of the on-screen keyboard\n"
                "Let me adjust the position of the on-screen keyboard\n"
                "Done adjusting the position of the on-screen keyboard"
            ),
            wraplength=520,
        ).pack(anchor="w")

        self.status_text = tk.StringVar(value="Ready for a command.")
        ttk.Label(frame, textvariable=self.status_text, wraplength=520).pack(
            anchor="w", pady=(14, 0)
        )

        root.update_idletasks()
        self.minimum_width = max(420, frame.winfo_reqwidth())
        self.minimum_height = max(320, frame.winfo_reqheight())
        root.minsize(self.minimum_width, self.minimum_height)
        root.geometry(
            f"{max(600, self.minimum_width)}x{max(440, self.minimum_height)}"
        )
        threading.Thread(target=self.command_worker, daemon=True).start()
        self.root.after(150, self.poll_events)
        if self.parent_pid:
            self.root.after(500, self.check_parent)
        self.entry.focus_set()

    def check_parent(self) -> None:
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
        kernel.GetExitCodeProcess.restype = ctypes.c_int
        kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel.OpenProcess(0x1000, False, self.parent_pid)
        if not handle:
            self.close()
            return
        code = ctypes.c_ulong()
        ok = kernel.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel.CloseHandle(handle)
        if not ok or code.value != 259:
            self.close()
        elif not self.stopping.is_set():
            self.root.after(500, self.check_parent)

    def schedule_command(self, *_args) -> None:
        if self.pending_after is not None:
            self.root.after_cancel(self.pending_after)
            self.pending_after = None
        phrase = self.command_text.get()
        if vc.wispr_command_action(phrase) is not None:
            self.pending_after = self.root.after(
                self.DEBOUNCE_MS, self.submit_command
            )

    def run_now(self, _event=None) -> str:
        if self.pending_after is not None:
            self.root.after_cancel(self.pending_after)
            self.pending_after = None
        self.submit_command()
        return "break"

    def submit_command(self) -> None:
        self.pending_after = None
        phrase = self.command_text.get().strip()
        if vc.wispr_command_action(phrase) is None:
            self.status_text.set(
                "No supported command matched. Nothing was done; try one of the listed phrases."
            )
            return
        self.commands.put(phrase)
        self.command_text.set("")
        self.status_text.set("Sending command…")

    @staticmethod
    def create_app_icon() -> tk.PhotoImage:
        image = tk.PhotoImage(width=48, height=48)
        background = "#101827"
        eye = "#e8f4fa"
        iris = "#36d6c2"
        pupil = "#101827"
        highlight = "#ffffff"
        for y in range(48):
            row = []
            for x in range(48):
                nx = (x - 23.5) / 19.0
                ny = (y - 23.5) / 12.5
                edge = 0.72 * max(0.0, 1.0 - nx * nx) ** 0.5
                color = eye if abs(nx) <= 1 and abs(ny) <= edge else background
                iris_distance = ((x - 24) ** 2 + (y - 24) ** 2) ** 0.5
                if iris_distance <= 7:
                    color = iris
                if iris_distance <= 3.5:
                    color = pupil
                if (x - 21) ** 2 + (y - 21) ** 2 <= 2:
                    color = highlight
                row.append(color)
            image.put("{" + " ".join(row) + "}", to=(0, y))
        return image

    def command_worker(self) -> None:
        while not self.stopping.is_set():
            try:
                phrase = self.commands.get(timeout=0.1)
            except queue.Empty:
                continue
            self.events.put(("command", vc.wispr_command_action(phrase) or ""))

    def _run_command(self, action: str) -> str:
        try:
            if action == "open_keyboard":
                self.keyboard.open()
                return "EyeSync keyboard opened. Click the text field you want to type in."
            if action == "close_keyboard":
                return ("EyeSync keyboard closed." if self.keyboard.close()
                        else "Keyboard was not open.")
            if action in ("increase_keyboard", "reduce_keyboard"):
                factor = 1.15 if action == "increase_keyboard" else 1 / 1.15
                return ("Keyboard size changed." if self.keyboard.resize(factor)
                        else "Open the keyboard first.")
            if action == "start_adjusting":
                return ("Moving keyboard. Say done adjusting to stop."
                        if self.keyboard.start_moving() else "Open the keyboard first.")
            if action == "stop_adjusting":
                self.keyboard.stop_moving()
                return "Keyboard released."
            return "No supported command matched."
        except OSError as exc:
            self.keyboard.stop_moving()
            return f"Command failed: {exc}"

    def poll_events(self) -> None:
        while True:
            try:
                kind, message = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "connected":
                self.connected = True
                self.gesture_status.set(message)
                self.status_text.set("Connected to EyeSync.")
            elif kind == "disconnected":
                self.connected = False
                self.gesture_status.set(message)
                self.status_text.set(message)
            elif kind == "gesture":
                self.gesture_status.set(message)
            elif kind == "result":
                self.status_text.set(message)
            elif kind == "command":
                self.status_text.set(self._run_command(message))
        if not self.stopping.is_set():
            self.root.after(150, self.poll_events)

    def close(self) -> None:
        self.stopping.set()
        self.keyboard.close()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", type=int, default=0)
    args = parser.parse_args()
    root = tk.Tk()
    WisprCommandPanel(root, args.parent_pid)
    root.mainloop()


if __name__ == "__main__":
    main()
