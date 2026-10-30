"""
main_window.py - EyeSync's one and only window.
===============================================

Everything a user needs, in one place:

    ● status            off / on / can't see your face / paused
    camera view         what EyeSync sees, with the face mesh drawn on it
    Pause  Recenter  Calibrate   (nine-point pointer calibration)
    Voice typing        (while on: a bar at the bottom of the screen, VoiceOverlay)
    Sensitivity         calmer ... faster
    Click by            blinking, or holding the cursor still
    Show camera, Face gestures, Start with Windows
    API key             for other websites and apps (Copy / New key)
    Help, Open log, Blink test (for testing on a new face), Quit EyeSync

The tracking engine (live_inference.run) runs in a background thread; this
window only reads its shared state and calls its Actions, so a slow or
minimised window never slows the tracking. Closing the window (X) only
minimises it - EyeSync stays ready for the website. "Quit EyeSync" ends it.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import gaze_engine
import app_paths

log = logging.getLogger("eyesync")

TITLE = "EyeSync"
PREVIEW_W, PREVIEW_H = 320, 240
COLORS = {                       # status dot colours (also the window icon)
    "off": "#969696",
    "active": "#28b95a",
    "noface": "#e13c3c",
    "paused": "#f0aa1e",
}


def icon_image(color: str = COLORS["active"]):
    """Load the EyeSync eye-and-pointer app symbol, with a safe fallback."""
    from PIL import Image, ImageDraw

    icon = Path(__file__).resolve().parent / "models" / "eyesync-icon.png"
    if icon.is_file():
        return Image.open(icon).convert("RGBA").resize((64, 64), Image.Resampling.LANCZOS)

    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((2, 14, 62, 50), fill="white", outline="#282828", width=3)
    d.ellipse((18, 18, 46, 46), fill=color, outline="#282828", width=2)
    d.ellipse((27, 27, 37, 37), fill="#141414")
    return img


class MainWindow:
    def __init__(self, root: tk.Tk, start_minimized: bool, help_path: Path, log_path: Path):
        self.root, self.help_path, self.log_path = root, help_path, log_path
        self.state = self.actions = self.status = None      # set by attach()
        self.engine: threading.Thread | None = None
        self._photo = None
        self._last_frame_id = None

        root.title(TITLE)
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self.root.iconify)    # X = minimise
        try:
            from PIL import ImageTk
            self._icon = ImageTk.PhotoImage(icon_image())
            root.iconphoto(True, self._icon)
        except Exception:                                      # noqa: BLE001 - cosmetic
            pass
        self._build()
        self.overlay = CalibrationOverlay(root, lambda: self._do("cancel_calibration"),
                                          self._calibration_shown)
        self.voice_bar = VoiceOverlay(root, self._stop_voice)
        # Centre in the visible desktop area (clear of the taskbar).
        root.update_idletasks()
        w, h = root.winfo_reqwidth(), root.winfo_reqheight()
        left, top, right, bottom = _work_area(root)
        x = max(left, left + (right - left - w) // 2)
        y = max(top, top + (bottom - top - h) // 2 - 20)
        root.geometry(f"+{x}+{y}")
        if start_minimized:
            root.iconify()

    # -- layout -------------------------------------------------------------
    def _build(self) -> None:
        r = self.root
        style = ttk.Style(r)
        style.configure("Status.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("Hint.TLabel", font=("Segoe UI", 9), foreground="#c0392b")
        style.configure("Big.TButton", font=("Segoe UI", 10), padding=(10, 6))
        pad = {"padx": 12}

        top = ttk.Frame(r)
        top.pack(fill="x", pady=(12, 0), **pad)
        self.dot = tk.Canvas(top, width=22, height=22, highlightthickness=0)
        self.dot_id = self.dot.create_oval(3, 3, 19, 19, fill=COLORS["off"], outline="")
        self.dot.pack(side="left")
        self.status_var = tk.StringVar(value="Starting EyeSync...")
        ttk.Label(top, textvariable=self.status_var, style="Status.TLabel").pack(
            side="left", padx=8)
        # Camera view size in screen pixels: the 360x270 image, scaled with
        # the Windows display scaling (125%, 150%...) so it matches the text.
        zoom = max(1.0, float(r.tk.call("tk", "scaling")) / (96 / 72))
        self.view_size = (round(PREVIEW_W * zoom), round(PREVIEW_H * zoom))
        self.hint_var = tk.StringVar(value="")
        ttk.Label(r, textvariable=self.hint_var, style="Hint.TLabel",
                  wraplength=self.view_size[0]).pack(fill="x", pady=(2, 6), **pad)

        # A fixed-size frame: a Label alone would size itself in characters
        # (not pixels) whenever it shows text instead of the camera image.
        box = tk.Frame(r, width=self.view_size[0], height=self.view_size[1], bg="#1e1e1e")
        box.pack_propagate(False)
        box.pack(**pad)
        self.view = tk.Label(box, bg="#1e1e1e", fg="#bbbbbb",
                             text="Starting the EyeSync camera…",
                             wraplength=self.view_size[0] - 40, font=("Segoe UI", 10))
        self.view.pack(fill="both", expand=True)

        buttons = ttk.Frame(r)
        buttons.pack(fill="x", pady=(8, 2), **pad)
        self.pause_btn = ttk.Button(buttons, text="Pause", style="Big.TButton",
                                    command=lambda: self._do("pause"))
        self.recenter_btn = ttk.Button(buttons, text="Recenter", style="Big.TButton",
                                       command=lambda: self._do("recenter"))
        self.calibrate_btn = ttk.Button(buttons, text="Calibrate", style="Big.TButton",
                                        command=lambda: self._do("calibrate"))
        for i, b in enumerate((self.pause_btn, self.recenter_btn, self.calibrate_btn)):
            b.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 6, 0))
            buttons.columnconfigure(i, weight=1)
        self.recenter_tip = ttk.Label(r, text="Recenter: puts the cursor in the middle of the "
                                              "screen.", foreground="#777777")
        self.recenter_tip.pack(anchor="w", **pad)
        self.calib_var = tk.StringVar(value="")
        ttk.Label(r, textvariable=self.calib_var, foreground="#777777",
                  wraplength=self.view_size[0]).pack(anchor="w", **pad)

        settings = ttk.LabelFrame(r, text="Settings")
        settings.pack(fill="x", pady=8, **pad)
        ttk.Label(settings, text="Click by").grid(row=2, column=0, sticky="nw", padx=8, pady=6)
        self.click_var = tk.StringVar(value="blink")
        clk = ttk.Frame(settings)
        clk.grid(row=2, column=1, sticky="w", pady=(4, 0))
        ttk.Radiobutton(clk, text="Smile: left click; open mouth: right click",
                        value="blink",
                        variable=self.click_var, command=self._set_click).pack(anchor="w")

        self.camera_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings, text="Show camera", variable=self.camera_var).grid(
            row=3, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 2))
        self.gestures_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings, text="Face gestures (see below)", variable=self.gestures_var,
                        command=self._set_gestures).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 2))
        self.autostart_var = tk.BooleanVar(value=False if app_paths.STANDALONE
                                           else _autostart_enabled())
        if not app_paths.STANDALONE:
            ttk.Checkbutton(settings, text="Start with Windows (opens with the website dashboard)",
                            variable=self.autostart_var, command=self._set_autostart).grid(
                row=6, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 4))

        ttk.Label(settings, text="API key").grid(row=7, column=0, sticky="w", padx=8,
                                                 pady=(2, 8))
        key_row = ttk.Frame(settings)
        key_row.grid(row=7, column=1, sticky="w", pady=(2, 8))
        self.key_var = tk.StringVar(value="-")
        ttk.Label(key_row, textvariable=self.key_var, font=("Consolas", 9)).pack(side="left")
        self.copy_key_btn = ttk.Button(key_row, text="Copy", width=7, command=self._copy_key)
        self.copy_key_btn.pack(side="left", padx=(8, 0))
        self.new_key_btn = ttk.Button(key_row, text="New key", width=8, command=self._new_key)
        self.new_key_btn.pack(side="left", padx=(4, 0))

        ttk.Label(r, foreground="#555555", wraplength=self.view_size[0], justify="left",
                  text="Calibrated gaze moves the pointer. Hold a smile for about 1.2 s "
                       "to left click, or open your mouth for about 1 s to right click. "
                       "Both eyes closed for 1 s toggles offline voice typing "
                       "with a beep. The voice bar's Stop button also works. "
                       "Gestures work only while this button says Pause. "
                       "The EyeSync Command Panel accepts on-screen keyboard commands.").pack(
                           anchor="w", pady=(0, 8), **pad)

        bottom = ttk.Frame(r)
        bottom.pack(fill="x", pady=(0, 6), **pad)
        ttk.Button(bottom, text="Help", command=lambda: _open(self.help_path)).pack(side="left")
        ttk.Button(bottom, text="Open log", command=lambda: _open(self.log_path)).pack(
            side="left", padx=6)
        # For testing EyeSync on a new face: a guided test, saved as numbers only.
        self.blink_test_btn = ttk.Button(bottom, text="Blink test",
                                         command=lambda: self._do("blink_test"))
        self.blink_test_btn.pack(side="left")
        ttk.Button(bottom, text="Quit EyeSync", command=lambda: self._do("quit")).pack(
            side="right")
        ttk.Label(r, text="Closing this window keeps EyeSync ready (in the taskbar).",
                  foreground="#777777").pack(anchor="w", pady=(0, 8), **pad)
        self._set_controls_enabled(False)

    # -- engine binding -----------------------------------------------------
    def attach(self, state, actions, status) -> None:
        """Called (on the Tk thread) once the engine is up."""
        self.state, self.actions, self.status = state, actions, status
        snap = state.read()
        self.click_var.set(snap.click_mode)
        self.gestures_var.set(snap.gestures)
        self._set_controls_enabled(True)
        self._show_key()

    def _set_controls_enabled(self, on: bool) -> None:
        for b in (self.pause_btn, self.recenter_btn, self.calibrate_btn,
                  self.blink_test_btn):
            b.state(["!disabled"] if on else ["disabled"])
        has_key = on and getattr(self.actions, "api_key", None) is not None
        for b in (self.copy_key_btn, self.new_key_btn):
            b.state(["!disabled"] if has_key else ["disabled"])

    def _show_key(self) -> None:
        """Only the ends of the key, so a screen share or recording doesn't leak it."""
        api_key = getattr(self.actions, "api_key", None)
        key = api_key.get() if api_key is not None else ""
        self.key_var.set(f"{key[:4]}…{key[-4:]}" if key else "-")

    def _copy_key(self) -> None:
        if getattr(self.actions, "api_key", None) is None:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self.actions.api_key.get())
        self.copy_key_btn.configure(text="Copied")
        self.root.after(1500, lambda: self.copy_key_btn.configure(text="Copy"))

    def _new_key(self) -> None:
        if getattr(self.actions, "api_key", None) is None:
            return
        from tkinter import messagebox
        if messagebox.askyesno(
                "Make a new API key?",
                "Websites and apps using the current key are disconnected and will need "
                "the new one. Your EyeSync website is not affected.\n\nMake a new key?",
                default="no", parent=self.root):
            self.actions.new_api_key()
            self._show_key()

    def _do(self, name: str) -> None:
        if self.actions is not None:
            getattr(self.actions, name)()

    def _stop_voice(self) -> None:
        if self.actions is not None:
            self.actions.voice(False, "the bar's Stop button")

    def _calibration_shown(self) -> None:
        """The result has been on the screen long enough: close the guide."""
        if self.state is not None and self.state.read().calib_step == "done":
            self.state.update(calib_step="", calib_text="", calib_progress=0.0)

    def _set_gestures(self) -> None:
        if self.actions is not None:
            self.actions.set_gestures(self.gestures_var.get())

    def _set_click(self) -> None:
        if self.actions is not None:
            self.actions.set_click_mode(self.click_var.get())

    def _set_autostart(self) -> None:
        import autostart
        try:
            if self.autostart_var.get():
                autostart.enable()
            else:
                autostart.disable()
        except OSError as exc:
            log.warning(f"[window] Could not change autostart: {exc}")
            self.autostart_var.set(_autostart_enabled())

    # -- periodic refresh ---------------------------------------------------
    def refresh(self) -> None:
        """Mirror the engine state into the window (~15 times a second)."""
        if self.state is not None:
            snap = self.state.read()
            key, text = self.status()
            self.status_var.set(text)
            self.dot.itemconfigure(self.dot_id, fill=COLORS[key])
            self.hint_var.set(snap.hint)
            self.pause_btn.configure(text="Resume" if snap.paused else "Pause")
            import app_paths
            calibrated = (app_paths.DATA_DIR / "gaze_model.joblib").is_file()
            self.calib_var.set("Pointer calibrated for this computer. Press Calibrate to "
                               "retrain it." if calibrated else
                               "Press Calibrate to train the pointer for your face and screen.")
            self.overlay.update(snap)
            self.voice_bar.update(snap)
            self.blink_test_btn.state(["disabled"] if snap.blink_test else ["!disabled"])
            if self.click_var.get() != snap.click_mode:
                self.click_var.set(snap.click_mode)
            # Only render the camera while it can be seen (saves CPU).
            visible = self.root.state() != "iconic" and self.camera_var.get()
            if snap.preview_on != visible:
                self.state.update(preview_on=visible)
            self._show_frame(visible)

    def _show_frame(self, visible: bool) -> None:
        frame_id, frame = self.state.get_preview()
        if not visible or frame is None:
            if self._photo is not None or not visible:
                self._photo = None
                msg = ("Camera view hidden" if not self.camera_var.get() else
                       "Starting the EyeSync camera…")
                self.view.configure(image="", text=msg)
            return
        if frame_id == self._last_frame_id:
            return
        self._last_frame_id = frame_id
        from PIL import Image, ImageTk
        img = Image.fromarray(frame[:, :, ::-1])                # BGR -> RGB
        if img.size != self.view_size:
            img = img.resize(self.view_size, Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(img)
        self.view.configure(image=self._photo, text="")


class CalibrationOverlay:
    """
    The full-screen guide while calibrating (live_inference.Calibration): a
    dot to look at, the instruction, the step's progress and Cancel (or
    Esc). It covers the whole screen, so the head turns towards the real
    screen edges. The engine moves through the steps; this only shows them.
    """

    STEPS = ("centre", "left", "right", "up", "down", "blink", "smile")   # = Calibration.STEPS
    DOT = {"left": (0.03, 0.5), "right": (0.97, 0.5), "up": (0.5, 0.05), "down": (0.5, 0.95)}
    GAZE = dict(zip(gaze_engine.GAZE_STEPS, gaze_engine.GAZE_POINTS))     # the eye calibration's dots
    RESULT_S = 6.0                 # the result stays this long
    BG, FG, ACCENT, DIM = "#11161c", "#f2f2f2", "#28b95a", "#7a8590"

    def __init__(self, root: tk.Tk, on_cancel, on_result_shown):
        self.root, self.on_cancel, self.on_result_shown = root, on_cancel, on_result_shown
        self.top: tk.Toplevel | None = None
        self.canvas: tk.Canvas | None = None
        self.drawn = None
        self.done_since: float | None = None

    def update(self, snap) -> None:
        import time
        step = snap.calib_step
        if not step:
            self._hide()
            return
        if self.top is None:
            self._show()
        if step == "done":
            self.done_since = self.done_since or time.monotonic()
            if time.monotonic() - self.done_since > self.RESULT_S:
                self.on_result_shown()
        else:
            self.done_since = None
        key = (step, snap.calib_text, snap.calib_progress)
        if key != self.drawn:
            self.drawn = key
            self._draw(step, snap.calib_text, snap.calib_progress)

    def _show(self) -> None:
        top = tk.Toplevel(self.root, bg=self.BG)
        top.overrideredirect(True)
        top.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0")
        top.attributes("-topmost", True)
        try:
            top.attributes("-alpha", 0.95)
        except tk.TclError:
            pass
        self.canvas = tk.Canvas(top, bg=self.BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        tk.Button(top, text="Cancel (Esc)", command=self.on_cancel, font=("Segoe UI", 12),
                  bg="#2a323b", fg=self.FG, activebackground="#3a444f", activeforeground=self.FG,
                  relief="flat", padx=16, pady=6).place(relx=0.98, rely=0.97, anchor="se")
        top.bind("<Escape>", lambda _e: self.on_cancel())
        top.focus_force()
        self.top, self.drawn = top, None

    def _hide(self) -> None:
        if self.top is not None:
            self.top.destroy()
        self.top = self.canvas = self.drawn = self.done_since = None

    def _draw(self, step: str, text: str, progress: float) -> None:
        c = self.canvas
        c.delete("all")
        self.top.update_idletasks()
        w, h = self.top.winfo_width(), self.top.winfo_height()
        mid_y = h * 0.6
        if step == "done":
            c.create_text(w / 2, h * 0.4, text="✓", fill=self.ACCENT, font=("Segoe UI", 64, "bold"))
            c.create_text(w / 2, mid_y, text=text, fill=self.FG, width=w * 0.7,
                          font=("Segoe UI", 22), justify="center")
            return
        # Where to look: a ring with a dot, at the centre or at an edge - or,
        # calibrating the eyes, at one of 9 places.
        fx, fy = self.GAZE.get(step) or self.DOT.get(step, (0.5, 0.5))
        x, y = fx * w, fy * h
        if step in self.DOT:
            c.create_line(w / 2, h / 2, x, y, fill="#2f3943", width=6, arrow="last",
                          arrowshape=(26, 30, 12))
        # The eyes' dot shrinks while it is measured: something to keep looking at.
        r = 9 + 12 * (1 - min(progress, 1.0)) if step in self.GAZE else 9
        c.create_oval(x - 30, y - 30, x + 30, y + 30, outline=self.ACCENT, width=5)
        c.create_oval(x - r, y - r, x + r, y + r, fill=self.FG, outline="")
        if step in self.STEPS:
            c.create_text(w / 2, mid_y - 70, text=f"Calibration - step {self.STEPS.index(step) + 1} "
                          f"of {len(self.STEPS)}", fill=self.DIM, font=("Segoe UI", 14))
        elif step in self.GAZE:
            c.create_text(w / 2, mid_y - 70, text=f"Eye calibration - dot "
                          f"{gaze_engine.GAZE_STEPS.index(step) + 1} of {len(self.GAZE)}",
                          fill=self.DIM, font=("Segoe UI", 14))
        c.create_text(w / 2, mid_y, text=text, fill=self.FG, width=w * 0.7,
                      font=("Segoe UI", 24, "bold"), justify="center")
        bx, by = w / 2 - 220, mid_y + 70
        c.create_rectangle(bx, by, bx + 440, by + 12, outline=self.DIM)
        if progress > 0:
            c.create_rectangle(bx, by, bx + 440 * min(progress, 1.0), by + 12,
                               fill=self.ACCENT, outline="")


class VoiceOverlay:
    """
    The voice typing bar at the bottom of the screen while voice typing is
    on (live_inference.Snapshot.voice): what it's doing - listening, hearing
    you, typing - with the microphone level, the last phrase typed (or a
    problem) and a Stop button. It never takes the keyboard focus from the
    window being typed into (WS_EX_NOACTIVATE) and stays above other windows.
    """

    BG, FG, DIM, WARN = "#11161c", "#f2f2f2", "#9aa4ae", "#f0aa1e"
    DOT = {"starting": "#7a8590", "listening": "#28b95a", "hearing": "#e13c3c",
           "working": "#f0aa1e", "error": "#7a8590"}
    TEXT = {"starting": "Starting voice typing...", "listening": "Listening - say what to type",
            "hearing": "Hearing you...", "working": "Typing...", "error": "Voice typing"}
    TIP = "Close both eyes for 1 s again, or click Stop"

    def __init__(self, root: tk.Tk, on_stop):
        self.root, self.on_stop = root, on_stop
        self.top: tk.Toplevel | None = None
        self.canvas: tk.Canvas | None = None
        self.stop_btn: tk.Button | None = None
        self.drawn = None

    def update(self, snap) -> None:
        if not snap.voice:
            self._hide()
            return
        if self.top is None:
            self._show()
        countdown = math.ceil(snap.voice_left) if snap.voice_left > 0 else 0
        key = (snap.voice, snap.voice_text, snap.voice_hint, round(snap.voice_level, 1), countdown)
        if key != self.drawn:
            self.drawn = key
            self._draw(snap, countdown)

    def _show(self) -> None:
        zoom = max(1.0, float(self.root.tk.call("tk", "scaling")) / (96 / 72))
        self.zoom = zoom
        w, h = round(600 * zoom), round(64 * zoom)
        left, _, right, bottom = _work_area(self.root)
        top = tk.Toplevel(self.root, bg=self.BG)
        top.withdraw()
        top.overrideredirect(True)
        top.attributes("-topmost", True)
        top.geometry(f"{w}x{h}+{left + (right - left - w) // 2}+{bottom - h - round(14 * zoom)}")
        self.canvas = tk.Canvas(top, width=w, height=h, bg=self.BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.stop_btn = tk.Button(top, text="Stop", command=self.on_stop, font=("Segoe UI", 11),
                                  bg="#2a323b", fg=self.FG, activebackground="#3a444f",
                                  activeforeground=self.FG, relief="flat", padx=14, pady=4,
                                  takefocus=0)
        self.stop_btn.place(relx=1.0, rely=0.5, x=-round(12 * zoom), anchor="e")
        self.top, self.drawn = top, None
        show_without_focus(top)

    def _hide(self) -> None:
        if self.top is not None:
            self.top.destroy()
        self.top = self.canvas = self.stop_btn = self.drawn = None

    def _draw(self, snap, countdown: int) -> None:
        c, z = self.canvas, self.zoom
        c.delete("all")
        h = int(c.cget("height"))
        # A microphone in a dot (the dot's colour says what it's doing) and the level.
        cx, cy, r = round(32 * z), h // 2, round(18 * z)
        c.create_oval(cx - r, cy - r, cx + r, cy + r, fill=self.DOT.get(snap.voice, "#7a8590"),
                      outline="")
        mw, mh = round(5 * z), round(8 * z)
        c.create_rectangle(cx - mw, cy - mh - round(2 * z), cx + mw, cy + round(3 * z),
                           fill="white", outline="")
        c.create_arc(cx - round(9 * z), cy - round(8 * z), cx + round(9 * z), cy + round(8 * z),
                     start=200, extent=140, style="arc", outline="white", width=max(1, round(2 * z)))
        c.create_line(cx, cy + round(8 * z), cx, cy + round(12 * z), fill="white",
                      width=max(1, round(2 * z)))
        x = round(64 * z)
        c.create_text(x, round(h * 0.34), text=self.TEXT.get(snap.voice, ""), anchor="w",
                      fill=self.FG, font=("Segoe UI", 12, "bold"))
        if snap.voice_hint:
            line, color = snap.voice_hint, self.WARN
        elif snap.voice_text:
            text = snap.voice_text if len(snap.voice_text) <= 60 else "..." + snap.voice_text[-57:]
            line, color = f"“{text}”", self.DIM
        else:
            line, color = self.TIP, self.DIM
        room = int(c.cget("width")) - x - round((16 if snap.voice == "error" else 110) * z)
        c.create_text(x, round(h * 0.72), text=line, anchor="w", fill=color,
                      font=("Segoe UI", 10), width=room)
        if snap.voice in ("listening", "hearing"):
            bar = round(120 * z * min(1.0, snap.voice_level))
            c.create_rectangle(x + round(330 * z), round(h * 0.30), x + round(450 * z),
                               round(h * 0.38), outline="#2f3943", fill="#1b2128")
            if bar:
                c.create_rectangle(x + round(330 * z), round(h * 0.30), x + round(330 * z) + bar,
                                   round(h * 0.38), outline="", fill=self.DOT["listening"])
        if snap.voice == "error":
            self.stop_btn.place_forget()


def show_without_focus(top: tk.Toplevel) -> None:
    """
    Show a borderless Tk window WITHOUT taking the keyboard focus (and never
    take it when clicked): voice typing types into the window that has it.
    """
    top.update_idletasks()
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        hwnd = wintypes.HWND(int(top.wm_frame(), 16))
        GWL_EXSTYLE, WS_EX_TOOLWINDOW, WS_EX_TOPMOST, WS_EX_NOACTIVATE = -20, 0x80, 0x8, 0x08000000
        user32.GetWindowLongW.argtypes = (wintypes.HWND, ctypes.c_int)
        user32.SetWindowLongW.argtypes = (wintypes.HWND, ctypes.c_int, wintypes.LONG)
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
                              | WS_EX_TOPMOST)
    except (AttributeError, OSError, ValueError, tk.TclError):
        pass                                        # not Windows: an ordinary topmost window
    top.deiconify()


def _work_area(root: tk.Tk) -> tuple[int, int, int, int]:
    """The desktop minus the taskbar, in pixels (whole screen if unknown)."""
    try:
        import ctypes
        import ctypes.wintypes
        r = ctypes.wintypes.RECT()
        if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(r), 0):  # SPI_GETWORKAREA
            return r.left, r.top, r.right, r.bottom
    except (AttributeError, OSError):
        pass
    return 0, 0, root.winfo_screenwidth(), root.winfo_screenheight()


def _open(path: Path) -> None:
    try:
        os.startfile(str(path))                                 # noqa: S606 - user's file
    except OSError as exc:
        log.warning(f"[window] Could not open {path}: {exc}")


def _autostart_enabled() -> bool:
    try:
        import autostart
        return autostart.enabled()          # also off when switched off in Windows' Startup apps
    except Exception:                                           # noqa: BLE001 - not Windows
        return False


def run_with_window(args, engine_main) -> int:
    """
    Start the engine (``engine_main(args, on_ready)``, a coroutine function)
    in a background thread and the window on this thread. Returns the
    engine's exit code once it stops (Quit, F12 or the close gesture).
    """
    import asyncio

    import app_paths
    import face_engine as fe

    fe.get_screen_size()         # opt into DPI awareness BEFORE Tk measures anything,
    root = tk.Tk()                # or text is clipped on scaled (125%+) displays
    win = MainWindow(root, getattr(args, "minimized", False),
                     app_paths.APP_DIR / "README.md", Path(args.log_path))
    ready: dict = {}
    result = {"code": 1}

    def on_ready(state, actions, status):                      # engine thread
        ready.update(state=state, actions=actions, status=status)

    def engine():
        try:
            result["code"] = asyncio.run(engine_main(args, on_ready))
        except Exception:
            log.exception("[main] Engine crashed")

    win.engine = threading.Thread(target=engine, name="engine", daemon=True)
    win.engine.start()

    def tick():
        if ready and win.state is None:
            win.attach(ready["state"], ready["actions"], ready["status"])
        if not win.engine.is_alive():
            root.destroy()
            return
        if ready and ready["state"].stop.is_set() and root.state() != "withdrawn":
            # Quitting: the camera can take a few seconds to shut down - hide
            # the window now, or it looks as if Quit didn't work.
            root.withdraw()
        try:
            win.refresh()
        except Exception:                                       # noqa: BLE001 - keep ticking
            log.exception("[window] Refresh failed")
        root.after(66, tick)

    root.after(66, tick)
    try:
        root.mainloop()
    finally:
        if ready:
            ready["state"].stop.set()
        win.engine.join(10)
    return result["code"]
