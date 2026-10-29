"""Nine-point gaze calibration for the combined EyeSync mouse controller."""
from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import app_paths  # noqa: E402
import face_engine as fe  # noqa: E402
from control_backend.adapter import gc  # noqa: E402

POINTS = 9
SAMPLES_PER_POINT = 50
SETTLE_S = 0.9


def wait_for_process(pid: int) -> None:
    """Wait for EyeSync to release the camera before calibrating."""
    if not pid:
        return
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel.OpenProcess(0x100000, False, pid)
    if handle:
        kernel.WaitForSingleObject(handle, 30000)
        kernel.CloseHandle(handle)
    time.sleep(1.0)


def train_model(samples: np.ndarray, targets: np.ndarray,
                screen: tuple[int, int], camera: tuple[int, int], out: Path) -> None:
    """Fit the source project's HybridGazeModel to this user's face."""
    scaler = StandardScaler().fit(samples)
    scaled = scaler.transform(samples)
    normalized = targets / np.asarray(screen, dtype=np.float64)
    rng = np.random.default_rng(0)
    augmented = np.vstack([scaled] + [
        scaled + rng.normal(0, 0.10, scaled.shape) for _ in range(5)])
    answers = np.vstack([normalized] * 6)
    model = gc.HybridGazeModel().fit(augmented, answers)
    bundle = {
        "model": model, "x_scaler": scaler,
        "screen_w": screen[0], "screen_h": screen[1],
        "feature_names": gc.FEATURE_NAMES, "camera_size": camera,
        "z_lo": scaled.min(axis=0) - 0.5,
        "z_hi": scaled.max(axis=0) + 0.5,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(".tmp")
    joblib.dump(bundle, temp)
    os.replace(temp, out)


def run_calibration() -> bool:
    if sys.platform != "win32":
        raise RuntimeError("This EyeSync package requires Windows.")
    ctypes.windll.user32.SetProcessDPIAware()
    root = tk.Tk()
    root.title("EyeSync gaze calibration")
    root.attributes("-fullscreen", True)
    root.configure(bg="black")
    screen = (root.winfo_screenwidth(), root.winfo_screenheight())
    canvas = tk.Canvas(root, width=screen[0], height=screen[1],
                       bg="black", highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    abort = [False]
    root.bind("<Escape>", lambda _event: abort.__setitem__(0, True))
    root.protocol("WM_DELETE_WINDOW", lambda: abort.__setitem__(0, True))

    def draw(message: str, point=None, progress: float = 0.0) -> None:
        canvas.delete("all")
        if point is not None:
            x, y = point
            canvas.create_oval(x-20, y-20, x+20, y+20, fill="white", outline="")
            canvas.create_oval(x-6, y-6, x+6, y+6, fill="#d63d3d", outline="")
        canvas.create_text(screen[0]//2, 48, text=message, fill="white",
                           font=("Segoe UI", 20, "bold"))
        canvas.create_text(screen[0]//2, screen[1]-55,
                           text="Keep your eyes open. Press Esc to cancel.",
                           fill="#bbbbbb", font=("Segoe UI", 14))
        if point is not None:
            x0, x1 = screen[0] * .3, screen[0] * .7
            canvas.create_rectangle(x0, screen[1]-28, x1, screen[1]-18,
                                    fill="#333333", outline="")
            canvas.create_rectangle(x0, screen[1]-28, x0+(x1-x0)*progress,
                                    screen[1]-18, fill="#32c878", outline="")
        root.update()

    cap = mesh = None
    try:
        draw("EyeSync: follow the nine dots with your head")
        time.sleep(1.5)
        cap = fe.open_any_camera(width=640, height=480)
        mesh = fe.FaceMeshBackend()
        fractions = (.08, .50, .92)
        points = [(int(screen[0]*x), int(screen[1]*y))
                  for y in fractions for x in fractions]
        features, targets = [], []
        camera_size = (640, 480)
        for index, point in enumerate(points):
            start = time.monotonic()
            count = 0
            while count < SAMPLES_PER_POINT:
                if abort[0]:
                    return False
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("The camera stopped delivering frames.")
                height, width = frame.shape[:2]
                camera_size = (width, height)
                landmarks = mesh.process(frame)
                message = f"Point {index+1}/{POINTS}: point your nose at the dot"
                if landmarks is None:
                    message = "Face not detected: face the camera"
                elif time.monotonic() - start >= SETTLE_S:
                    face = np.asarray(landmarks[:468], dtype=np.float64)[:, :2]
                    low, high = face.min(axis=0), face.max(axis=0)
                    features.append(np.clip([*low, *high], 0, 1))
                    targets.append(point)
                    count += 1
                draw(message, point, count / SAMPLES_PER_POINT)
        draw("Training your gaze model...")
        out = app_paths.DATA_DIR / "gaze_model.joblib"
        train_model(np.asarray(features), np.asarray(targets),
                    screen, camera_size, out)
        return True
    finally:
        if mesh is not None:
            mesh.close()
        if cap is not None:
            cap.release()
        root.destroy()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, default=0)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args(argv)
    wait_for_process(args.wait_pid)
    try:
        success = run_calibration()
    except Exception as exc:
        messagebox.showerror("EyeSync calibration", str(exc))
        return 1
    if not success:
        return 1
    messagebox.showinfo("EyeSync calibration", "Calibration saved. EyeSync is ready.")
    if args.restart:
        if app_paths.FROZEN:
            command = [sys.executable, "--portable"]
        else:
            command = [sys.executable, str(ROOT / "live_inference.py"),
                       "--always-active", "--mouse", "software", "--stall-timeout", "45",
                       "--site-origin", "http://localhost:8000",
                       "--site-origin", "https://aniruddh9140.github.io"]
        subprocess.Popen(command, cwd=str(app_paths.DATA_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
