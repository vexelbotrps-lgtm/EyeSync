# EyeSync in brahmastra

This is the separate working copy. The website source and its Windows app came
from `face-control-website-main`. The calibrated gaze model, native Windows
mouse output, smile left click, mouth-open right click, and on-screen keyboard
command panel came from
`Face Controlled Access to Computer`.

Double-click `Run-EyeSync.bat`. It uses this folder's Python 3.15.2 virtual
environment, starts the Windows EyeSync app in software-mouse mode, and opens
the local command panel. The only face gestures are a held smile for left
click, an open mouth for right click, and both eyes closed for 1 second to
toggle offline voice typing. A rising beep confirms start; a falling beep
confirms stop. Voice typing stays on through silence and has no session time
limit. The on-screen bar's Stop button also works. The command
panel accepts the six exact on-screen keyboard phrases shown in its window.
All three face gestures work only while the EyeSync button says **Pause**.
When it says **Resume**, gesture holds are cleared and no gesture fires.

This copy includes the other project's saved gaze model, which was calibrated
for a 1366 x 768 display. The model output is scaled to the current display.
Press **Calibrate** in the EyeSync app to train the active pointer model for
your own face, camera and screen. EyeSync briefly closes to release the camera,
shows nine dots, saves your new model, then restarts. The older **Calibrate eyes**
button was for the original gaze-jump feature and has been removed.

For a downloadable Windows package, use
`EyeSync-Standalone-Windows-v1.0.1.zip` in this folder. Upload that ZIP as a GitHub
Release download. Users extract the whole ZIP and double-click
`EyeSync-Standalone\Start EyeSync.bat`; no Python installation is needed. The first run
guides them through the nine-dot pointer calibration. Rebuild the package with
`.venv\Scripts\python.exe build_portable.py` after moving the previous ZIP.

The standalone build uses a separate executable and window title, process lock,
local WebSocket ports 8773/8774 and personal files beside its executable. It
does not invoke the original EyeSync installer, updater, autostart registration
or port-closing helper.

The page can be previewed locally with:

```
.venv\Scripts\python.exe -m http.server 8000
```

Open `http://localhost:8000` in a browser. The static site's release download
still points to the original published EyeSync release; run this local copy
with `Run-EyeSync.bat` to use the combined controls.

To rebuild the Python environment if necessary:

```
py -3.15 -m venv .venv
.venv\Scripts\python.exe -m pip install -r EyeSync\requirements.txt
```

The voice and face model files are in `EyeSync/models`. The website project's
browser-only head demo still demonstrates its original blink interaction.
