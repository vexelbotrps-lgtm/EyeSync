# EyeSync and the EyeSync website

The website: **https://aniruddh9140.github.io/face-control-website/**
(GitHub Pages). EyeSync accepts connections from
`https://aniruddh9140.github.io` without an API key. This file is the
protocol reference; `WEBSITE_CHANGES.md` is the to-do list for the website.

EyeSync listens on `ws://localhost:8765` and `ws://localhost:8080` (any
path). There are two programs behind those ports:

- the **standby**: small, always running after login, no camera. It
  listens on the ports while EyeSync is closed and starts EyeSync when
  asked;
- **EyeSync** itself: camera, cursor control, the data and the commands.

## Connection requests

Add one of these to the address (the website's own Origin is enough, no key):

| Address | When | Standby (EyeSync closed) | EyeSync running |
|---|---|---|---|
| `ws://localhost:8765/?probe=1` | "Is EyeSync installed? running?" - automatic checks | sends `{"type":"eyesync","state":"standby",...}`, closes with **4003**. Never starts EyeSync. | sends `{"type":"eyesync","state":"running",...status}`, closes with **4003**. Doesn't switch cursor control on. |
| `ws://localhost:8765/?start=1` | the user pressed **Connect** | sends `{"type":"eyesync","state":"starting",...}`, closes with **4002**, and **starts EyeSync** (ready in ~5-20 s) | a normal connection (as below) |
| `ws://localhost:8765` | normal connection | refused (HTTP 503 -> the browser reports code 1006), as before | data + commands; cursor control is on while connected |

Nothing answering at all (code 1006 and the socket never opened) means
EyeSync is not installed, or nothing is waiting in the background (its
login start was switched off in Windows, or the standby was stopped).

**Opening EyeSync from the website:** EyeSync registers the link
`eyesync://start` with Windows (installing EyeSync.exe, or install.bat, does). Opening it
from a click on the website starts EyeSync for the website, minimised,
even when nothing is waiting. The browser asks the user first. Then probe
until it answers, and connect.

The `eyesync` message from a probe or start:

```json
{"type": "eyesync", "state": "standby", "version": "1.1.1", "closed_by_user": true}
```

`closed_by_user` is true when the user closed EyeSync themselves (mouth
wide open 3 s, Quit, F12, or the website's `quit` command): don't start it
again automatically; wait for the user to press Connect. False = it is
waiting since login, or closed because the dashboard was closed: a
dashboard that auto-connects may start it.

Older pages without `?probe` / `?start` keep working: the standby still
starts EyeSync on two plain tries less than 0.75 s apart (a Connect click
in the old reconnect logic), and at login on a dashboard that keeps trying.

### Close codes

| Code | Meaning | What the website should do |
|---|---|---|
| 4000 | EyeSync was closed by the user ("press Connect to start it again") | stop reconnecting automatically; show Connect |
| 4001 | refused: other websites need the API key | show the reason; don't retry |
| 4002 | standby: EyeSync is starting | reconnect (plain address) every ~1 s for up to 30 s |
| 4003 | answer to `?probe=1` | read the `eyesync` message before it |
| 1006 without opening | nothing listening | "EyeSync not installed or not running" |

## What EyeSync sends

About 30 messages a second:

```json
{"x": 412.6, "y": 301.2, "blink": false, "confidence": 1.0, "face": true}
```

x/y are in an **800 x 600** frame, which is the website's default
"Tracker frame width / height" under **Cursor and mapping**, so the
website's gaze cursor follows EyeSync's cursor with no setup. If you change
that setting on the website, start EyeSync with the same size:
`--dashboard-size 1920x1080` (or `native` for screen pixels with "Use
coordinates as window pixels"). `blink` is true while the eyes are closed
and on every click.

Messages that have a `"type"` are not tracking data: `status`, `preview`,
`eyesync`, `error` (below). Don't count them as rejected frames.

## Commands (JSON text the page sends)

| Purpose | Message |
|---|---|
| Ask for the status (and get every later change) | `{"action": "status"}` |
| Enable / disable mouse control | `{"action": "mouse", "value": true}` / `false` |
| Pause / resume | `{"action": "pause", "value": true}` / `false` |
| Recenter the cursor | `{"action": "recenter"}` |
| Start EyeSync's calibration (a full-screen guide on the PC, ~30 s) | `{"action": "calibrate"}` |
| Cancel it | `{"action": "calibrate", "value": "cancel"}` |
| Start the eye calibration for gaze jump (9 dots on the PC's screen, ~20 s; since 1.3.0) | `{"action": "calibrate", "value": "gaze"}` |
| Gaze jump on / off (since 1.3.0) | `{"action": "gaze", "value": true}` / `false` |
| Speed 1-5 | `{"action": "speed", "value": 3}` |
| Click by blinking / by holding still | `{"action": "click_mode", "value": "blink"}` / `"dwell"` |
| Face gestures on/off | `{"action": "gestures", "value": true}` |
| Voice typing on / off / switch (since 1.2.0) | `{"action": "voice", "value": true}` / `false` / `{"action": "voice"}` |
| Camera view on the page on/off | `{"action": "preview", "value": true}` / `false` |
| Bring the EyeSync window to the front | `{"action": "show_window"}` |
| Quit EyeSync (it then needs Connect to start again) | `{"action": "quit"}` |

Each command is answered with the status; a bad one with
`{"type": "error", "error": "..."}`. After its first command, a page also
receives the status **whenever it changes** (checked 4 times a second): a
smile paused EyeSync, a hotkey changed the speed, calibration progress, the
face was lost...

```json
{"type": "status", "version": "1.2.0", "mouse_enabled": true, "paused": false,
 "active": true, "face": true, "camera": true, "speed": 1, "click_mode": "blink",
 "gestures": true, "calibrating": false, "calibrated": "2026-10-16",
 "calibration": {"step": "", "text": "", "progress": 0.0},
 "hint": "", "hardware_mouse": false,
 "gaze": {"calibrated": "2026-10-18", "on": true, "error_px": 120},
 "voice": {"on": false, "state": "off", "hint": ""}}
```

- `active`: EyeSync moves the cursor now (connected, not paused, mouse enabled)
- `face` / `camera`: a face is seen / the camera delivers pictures
- `calibrating`, `calibration`: while EyeSync's calibration runs: `step` is
  `starting`, `centre`, `left`, `right`, `up`, `down`, `blink`, `smile`,
  then `done` (with the result in `text`); `progress` 0-1 of the step. The
  eye calibration's steps are `gaze1` ... `gaze9` (one per dot) instead.
- `calibrated`: date of the user's saved calibration, "" if never
- `gaze` (1.3.0+; missing in older versions): gaze jump. `calibrated`: date
  of the eye calibration ("" if never), `on`: gaze jump is switched on (it
  works once the eyes are calibrated), `error_px`: how accurately EyeSync
  knows where the user looks. Where the user looks is never sent.
- `hint`: advice EyeSync shows in its window (e.g. "too dark"), "" if none
- `hardware_mouse`: the XIAO USB mouse is connected
- `voice` (1.2.0+; missing in older versions): voice typing. `on`: the
  microphone is listening. `state`: `off`, `starting`, `listening`
  (waiting for speech), `hearing` (the user is talking), `working`
  (recognising and typing), or `error` for ~5 s after a problem, which
  `hint` describes ("Can't use the microphone ..."). What the user says is
  **never** sent to the page: it is typed into whatever has the keyboard
  focus on the PC.

Voice typing types into the window that has the keyboard focus, so after
the page's voice typing button is pressed, the browser and this page are in
front. The page should tell the user to **click (blink) where the text
should go, then speak**. Text typed while the dashboard itself has the
focus goes into the page, so don't give the dashboard single-key keyboard
shortcuts.

The camera view, after `{"action": "preview", "value": true}`, ~8 a second:

```json
{"type": "preview", "image": "data:image/jpeg;base64,/9j/4AAQ..."}
```

320 x 240, mirrored, with the face mesh drawn. Show it in an `<img>`; it
never needs to leave the browser.

If a page disables mouse control and then closes, EyeSync turns it back on
for the next visit.

## Calibration: EyeSync's and the website's

EyeSync's calibration (`calibrate`) fits the head pointer, blink clicks and
smile gesture to the user, and is saved on the PC. The website's 9-point
calibration measures how accurately the cursor reaches its targets; EyeSync
only logs its `{"type": "calibration", "event": ...}` messages and doesn't
reply.

## Downloads and updates

EyeSync is one file, `EyeSync.exe`: the whole application (no Python
needed), which installs itself the first time it's opened.
`py -3.15 build_exe.py` builds it and `latest.json` into
`E:\EyeSync-Release`; `python make_release.py` adds the zip (Python
version).

```json
{"version": "1.2.0", "file": "EyeSync.exe", "size": 220000000,
 "sha256": "...", "date": "2026-10-18", "zip": "EyeSync-1.2.0.zip"}
```

`EyeSync.exe` is over GitHub's 100 MB file limit for repositories, so
attach it to a GitHub Release:
`…/releases/latest/download/EyeSync.exe` always gives the newest one. Keep
`latest.json` on the site itself (browsers can't `fetch` release files
from another site). The website compares `version` with the version
EyeSync reports and offers the download when it's newer. Opening a newer
EyeSync.exe updates EyeSync and keeps the user's settings, calibration and
API key.

## Notes for the website

- The website's "Dwell to select" (off by default) clicks buttons on the
  page by itself. EyeSync already clicks with blinks, winks or its own dwell
  setting, so leave the website's dwell off, or a button could be pressed
  twice.
- Other websites and apps need EyeSync's API key (`?key=...` in the
  address); see the README. They can use `probe`/`start` with the key too.
- Testing a copy of the website from another address (e.g.
  `http://localhost:8000`): EyeSync only trusts the real site's origin, so
  add `?key=<API key>` to the address while testing.
