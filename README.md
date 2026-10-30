# EyeSync

EyeSync is a Windows face-controlled pointer with offline voice typing. Download
**EyeSync-Setup-v1.0.8.exe** from this repository's
[Releases](https://github.com/vexelbotrps-lgtm/EyeSync/releases) and open it.
Setup installs the application in your user folder and creates an EyeSync
shortcut on the Desktop, or in a folder you choose. Click the shortcut to run the
nine-dot calibration on first launch and open the app. Python is bundled.

For a portable copy, download **EyeSync-Standalone-Windows-v1.0.8.zip**, extract the
whole ZIP, and run **Start EyeSync.bat**. The source is open under the [MIT License](LICENSE).
The app's third-party components have their own notices in
[THIRD_PARTY_NOTICES.md](EyeSync/THIRD_PARTY_NOTICES.md).
Developers can download **EyeSync-Developer-Source-v1.0.8.zip**, which includes
the application source, runtime models, and setup instructions.

Smile to left click, open your mouth to right click, and close both eyes for
one second to toggle voice typing. Gestures work while the application button
says **Pause**; they stop while it says **Resume**.
Voice typing runs offline with a higher-accuracy English model tuned and tested
for accented speech, including Indian English.

The Command Panel opens EyeSync's own on-screen keyboard. Its voice commands
can open, close, resize and move it without Windows administrator permission.
Click the text field you want to use before pressing its keys.

This edition has its own executable, local WebSocket ports, window and process
lock. It does not install, update, or stop the original EyeSync application.

The combined Windows app source is in `EyeSync/`. The original website source
and its older design notes follow below. The saved personal gaze model is not
published; each user calibrates their own.

## Website source and background

**EyeSync**: the website for EyeSync, a Windows program that turns a webcam into a mouse you steer with your head and click with a blink. The site is where people download EyeSync and the dashboard they control it from. It is a static page (`index.html`) with inline CSS and JavaScript, plus a small lazy-loaded browser-tracking engine (`ai/head-demo-engine.mjs`), built to run on GitHub Pages with no build step and no server.

The EyeSync program lives in its own folder (`E:\EyeSync`); its `WEBSITE.md` is the protocol reference this site follows.

## What it does

- **Home page.** Visitors land on an introduction (`#home`) dressed as a little retro desktop: a menu bar with a clock, and sections as windows you can nudge by their title bars. It covers what EyeSync does, the face gestures, how to get started, and the privacy and accessibility approach.
- **Live head demo.** "Try it with your head" under the home page's CRT lets a visitor steer that screen with their own head and deliberately blink to click, with no install. Google's MediaPipe face landmarker and the site's signal engine are served locally and loaded only when the button is pressed. The engine uses median-based centering, motion-adaptive One Euro smoothing and face-loss recovery. Ordinary blinks do not click: close both eyes until the tick (about 0.32 s, or 0.4 s while moving), then open them. Slow reopenings, long closures and smile squints are rejected; asymmetric two-eyed blinks still work. Camera pictures stay in the tab: nothing is stored or sent, and the camera turns off on Stop, when the tab is hidden, or when the visitor leaves the page. The library's own usage reporting is switched off.
- **Sync, the mascot.** A small pixel computer in the corner of the home page watches the pointer, blinks when you click, and has tips when pressed.
- **Why we made it.** `#why` explains who EyeSync is for: people living with paralysis, or with a disability or injury that makes a mouse and keyboard hard or impossible to use. It covers the problem, who it's for, what EyeSync promises, who it may not suit, and a button to copy a link to share. Every page links to it from the top menu and the footer, and the home page has a window about it right after the introduction.
- **Download page.** `#download` reads `downloads/latest.json` and offers `EyeSync.exe` from the newest GitHub Release, for example "Download EyeSync 1.1.1 for Windows (124 MB)", with the release date, size and SHA-256, and the source zip for developers. EyeSync.exe is the whole program and installs itself. The page explains how to install, update and remove it. Guests can open it without an account.
- **Accounts with Supabase.** People sign up and sign in with an email address and password, handled by Supabase Auth (see [Accounts](#accounts)), so an account works on any computer. Supabase keeps only a bcrypt hash of the password and rate-limits sign-in attempts. The dashboard settings and the last accuracy test are saved to the account a moment after each change and loaded after signing in.
- **Sessions.** With **Keep me signed in** (ticked by default), the Supabase session goes in `localStorage`, so you stay signed in after closing the site, in every tab, for 30 days since your last visit. Unticked, it stays in `sessionStorage` for this tab only and ends after 30 minutes without activity. Protected views (`#dashboard`, `#calibration`) check it on every navigation; Supabase confirms it on each visit and every 10 minutes, so a deleted account is signed out everywhere. Signing out in one tab signs out the others.
- **Cookie consent.** A banner appears on the first visit. Optional preference storage (`fc.prefs`) stays in memory until the visitor chooses **Accept**. Choosing **Essential only** deletes it. The Cookie Policy and Terms and Conditions open as pages (`#cookies`, `#terms`) or as dialogs from the registration form.
- **Mandatory agreement.** The registration form has an unchecked "I agree to the Terms and Conditions and Cookie Policy" checkbox. No account is created unless it is checked, and the versions agreed to are recorded with the account (in its Supabase user data and its `profiles` row).
- **Connecting to EyeSync.** On the Dashboard and Calibration pages, the site asks EyeSync whether it is installed, running or closed (see [Connecting to EyeSync](#connecting-to-eyesync)). It connects to a running EyeSync, starts a waiting one, and never restarts an EyeSync the user closed: only a click on **Connect** does. When nothing answers, it offers **Open EyeSync** and the Download page.
- **EyeSync controls.** Once EyeSync answers, a panel shows its status (face found, camera, cursor control, advice such as "too dark", calibration date, hardware mouse, version) and its controls: mouse control on/off, pause, recenter, speed 1 to 5, click by blinking or by holding still, face gestures, gaze jump (EyeSync 1.3.0 and newer), voice typing (EyeSync 1.2.0 and newer), show the EyeSync window, and quit (with an "Are you sure?" step). EyeSync pushes every change, so a smile that pauses it on the PC shows up on the page by itself.
- **Calibrate EyeSync.** Starts EyeSync's own 30-second calibration, which shows a full-screen guide on the PC, with live progress and a Cancel button on the page. It is on the Dashboard and first on the Calibration page.
- **Accuracy test.** The site's own 9-point routine. Targets light up one at a time while the page measures how far the cursor lands from each one. It sends its events to the tracker, which EyeSync only logs.
- **Gaze jump.** EyeSync 1.3.0 also knows roughly where the user looks: look at a far spot, nudge the head towards it, and the cursor jumps there; the head still does the aiming. The dashboard shows a **Gaze jump** switch, "Eyes calibrated (about N px)" or "Eyes not calibrated", and a **Calibrate eyes** button next to **Calibrate EyeSync** that starts the 9-dot eye calibration on the PC's screen, with its progress (`gaze1` to `gaze9`) on the page. All of it is hidden for older versions, which send no `gaze` status. Both calibrations start at `starting` and end at `done`, so the page remembers which one it asked for (or saw from its steps). Where the user looks is never sent to the page, only whether gaze jump is on and how accurate it is.
- **Voice typing.** EyeSync 1.2.0 types what the user says, recognised offline on the PC. The dashboard shows a switch (hidden for older versions, which send no `voice` status), the state (listening, hearing you, typing) and any microphone problem. What is said is never sent to the page; after the switch is turned on it lets go of the keyboard focus, and the page tells the user to click where the text should go.
- **Camera view.** An optional "Show camera" switch shows EyeSync's camera picture (320×240, mirrored, with the face mesh) about 8 times a second. The pictures are only put on an `<img>`: never stored, uploaded or logged. The view turns off when you leave the Dashboard.
- **Update notice.** When the connected EyeSync is older than `latest.json`, the Dashboard says so and links to the Download page. Versions are compared as numbers per part, so 1.10.0 is newer than 1.9.0.
- **Live cursor.** The red crosshair is positioned with CSS `top`/`left` from each `{"x", "y", "blink"}` update. Every update is validated, and a stale stream is flagged.
- **Other trackers.** Any program that sends `{"x", "y", "blink"}` on ports 8766 to 8770 or 8000, or on an address entered under **Enter an address yourself**, still works, with automatic reconnects.
- **Typography.** Fraunces (headings), Atkinson Hyperlegible (body text, designed for low vision) and IBM Plex Mono (labels) are bundled in `fonts/` as Latin-subset WOFF files, about 113 KB in total, under the SIL Open Font License (licence texts included). No font host is ever contacted.
- **Assistive extras.** Every control is at least 56 px with solid colour-inversion hover states. Light and dark themes are both included, and reduced-motion preferences are respected. Optional dwell-to-select presses a button when the crosshair rests on it; it is switched off while EyeSync is connected, because EyeSync clicks by itself and a button would otherwise be pressed twice.

## Run it locally

```bash
python -m http.server 8000
```

Then open http://localhost:8000. You can also open `index.html` directly from disk, but then the Download page can't read `latest.json`.

EyeSync only trusts the GitHub Pages origin, so a local copy can't talk to the real EyeSync. Use the mock below, or EyeSync's API key as described in its `WEBSITE.md`.

## Test the browser tracking engine

The signal engine has no browser or MediaPipe dependency, so its timing and tracking tests run with Node.js and no install step:

```bash
node --test tests/head-demo-engine.test.mjs
```

The suite covers calibration outliers and timing, scale-invariant pose extraction, jitter and fast movement, natural and deliberate blinks at 30/60 fps, asymmetric blinks, slow reopenings, smile squints, cooldowns, face loss and recentering. `HEAD_DEMO_AI_VERSION` in the engine and `HEAD_AI_VERSION` in `index.html` must be bumped together when the engine changes, so returning visitors receive the new file rather than a cached copy.

## Test without hardware

The dashboard has a built-in simulator: click **Run simulator** on the Dashboard.

`tools/mock_eyesync.py` plays EyeSync 1.3 and its standby: probes, starting, status pushes, every command, calibration progress, the eye calibration and gaze jump, voice typing, the camera view (with Pillow) and quitting with code 4000. Type `smile`, `speed 3`, `face off`, `hint …`, `eyes`, `say …`, `mic off`, `calibrate`, `calibrate eyes`, `quit` or `idle` while it runs to act "on the PC". `--version 1.2.0` plays an older EyeSync without gaze jump, `--version 1.1.4` one without voice typing too.

```bash
pip install websockets pillow
python tools/mock_eyesync.py                     # standby, as after login
python tools/mock_eyesync.py --state running --version 1.0.0   # shows the update notice
```

If the real EyeSync is already using ports 8765 and 8080, run the mock with `--ports 8771 8772` and test a copy of `index.html` whose `EYESYNC_PORTS` is `[8771, 8772]`.

`tools/mock_tracker.py` is a plain tracker that only streams `{"x", "y", "blink"}` at 60 updates a second and prints the accuracy-test events it receives:

```bash
python tools/mock_tracker.py --port 8766
```

Both mocks only accept pages from the GitHub Pages site and `http://localhost:8000`. Use `--origin` to change that.

## Connecting to EyeSync

EyeSync and its standby listen on `ws://localhost:8765` and `ws://localhost:8080` (`EYESYNC_PORTS`). The standby runs from login, has no camera, and starts EyeSync when asked. The site makes three kinds of request:

| Request | Sent when | Standby (EyeSync closed) | EyeSync running |
|---|---|---|---|
| `?probe=1` | every automatic check, every 5 s while the Dashboard or Calibration page is open and nothing is connected | `{"type":"eyesync","state":"standby","closed_by_user":…}`, close 4003; never starts EyeSync | `{"type":"eyesync","state":"running",…}`, close 4003 |
| `?start=1` | a click on **Connect**, or auto-connect when `closed_by_user` is false | `"starting"`, close 4002, and EyeSync starts; the site then tries the plain address every second for up to 30 s | a normal connection; the site closes it and connects to the plain address |
| plain address | only after EyeSync said it is running, or while it starts | refused (HTTP 503, seen as 1006) | the live connection: updates and commands |

What the Dashboard does with the answers:

- **Nothing answers:** "EyeSync isn't running on this computer", with **Open EyeSync** and **Download EyeSync**. Open EyeSync is the link `eyesync://start`, which the installed EyeSync registers with Windows. After the browser asks the user, EyeSync starts minimised; the site asks its ports every second for up to 30 s and then carries on as for Connect. If nothing answers by then: "EyeSync doesn't seem to be installed on this computer."
- **Running:** connect.
- **Waiting, not closed by the user:** start it if **Connect to EyeSync automatically** is ticked, otherwise say "Press Connect to start it".
- **Closed by the user** (mouth gesture, Quit, F12 or the website's Quit): never start it by itself. "EyeSync was closed. Press Connect to start it."
- **Close 4000 while connected:** the same; the site doesn't reconnect by itself.
- **Close 4001** (the page's origin isn't trusted): show the reason and stop trying until Connect is pressed.

Right after connecting, the site sends `{"action":"status"}`. From then on EyeSync pushes every status change. Messages with a `"type"` (`status`, `preview`, `eyesync`, `error`) are routed to the EyeSync panel and never counted as rejected updates; errors appear as a toast.

EyeSync 1.0's standby ignored `probe` and `start` and treated two tries within 0.75 s as a Connect click, so automatic tries to EyeSync's ports stay at least 1.5 s apart (`EYESYNC_GAP_MS`).

### Tracking data and other trackers

Each update is one JSON text frame, about 30 a second from EyeSync:

```json
{"x": 412.6, "y": 301.2, "blink": false, "confidence": 1.0, "face": true}
```

`x` and `y` are numbers. By default they are pixels in an 800×600 tracker frame, which is EyeSync's frame and is scaled to the browser window; leave **Tracker frame** at 800×600 for EyeSync. `blink` is optional and defaults to `false`. Updates over 4 KB, updates that aren't JSON, and updates without numeric `x`/`y` are rejected.

During the accuracy test the page sends:

```json
{"type": "calibration", "event": "start", "points": 9, "viewport": {"w": 1280, "h": 800}, "frame": {"w": 800, "h": 600}}
{"type": "calibration", "event": "point", "index": 0, "label": "Top left", "viewport": {"x": 60, "y": 60}, "source": {"x": 38, "y": 45}, "normalized": {"x": 0.0469, "y": 0.075}}
{"type": "calibration", "event": "point_done", "index": 0, "samples": 50, "mean_error_px": 23}
{"type": "calibration", "event": "complete", "mean_error_px": 18, "points": [{"i": 0, "e": 23, "n": 50}]}
{"type": "calibration", "event": "cancel"}
```

`source` is the target position in tracker-frame coordinates. A tracker can ignore these messages.

## Publishing a new EyeSync version

In the EyeSync folder, run `py -3.15 build_exe.py` (and `python make_release.py` for the source zip). Both write to `E:\EyeSync-Release`.

1. `EyeSync.exe` is over GitHub's 100 MB limit for files in a repository, so it doesn't go in here. Create a GitHub Release in this repository (tag `v<version>`, for example `v1.1.1`) and attach `EyeSync.exe` and `EyeSync-<version>.zip`. Keep those file names: the page links to `https://github.com/Aniruddh9140/face-control-website/releases/latest/download/EyeSync.exe`, which always gives the newest release.
2. Copy `latest.json` into `downloads/` here and push.

The page itself doesn't change: the Download button and the update notice both come from `latest.json`. Browsers can't `fetch` files on the Releases page from this site (no CORS), which is why `latest.json` lives here.

## Browser notes

- GitHub Pages is served over HTTPS. Browsers allow an HTTPS page to open `ws://localhost`, which is exempt from mixed-content blocking. Chrome, Edge and Firefox work the same way.
- Newer Chrome and Edge versions may ask the user to allow the site to reach "apps and services on this device" (local network access). If that is denied, the Dashboard says: "Allow this site to connect to apps on this device (click the icon in the address bar), then reload."
- To accept a tracker on another machine, add its host to both `LOOPBACK_HOSTS` in the script and `connect-src` in the Content-Security-Policy meta tag.

## Accounts

Accounts live in the Supabase project `pbztuguamyjznxldpbcc` (https://pbztuguamyjznxldpbcc.supabase.co), on the free plan. The page talks to it with plain `fetch` calls (no Supabase library), using the project's **publishable** key, which is meant to be public. What protects the data is `supabase/schema.sql`: run once in the project's SQL Editor, it creates the `profiles` table (one row per account: `prefs`, `agreed`), Row Level Security policies that let each signed-in user read and change only their own row, and `delete_my_account()`, which deletes only the caller. Never put the project's secret (service_role) key or database password in the site.

Project settings this relies on: **Authentication → Sign In / Providers**: Email on, **Confirm email** off (Supabase's built-in email only reaches the project's own team; connect an email service such as Brevo or Resend under **Authentication → Emails → SMTP** before turning confirmation or password reset on). **URL Configuration**: Site URL `https://aniruddh9140.github.io/face-control-website/`.

Free projects are paused after about a week without any requests. While paused, sign-in shows "The account service isn't answering"; restore it from the Supabase dashboard.

To test accounts without touching the real project, run `tools/mock_supabase.py` and point a copy of the page at it:

```bash
python tools/mock_supabase.py --token-ttl 70     # short tokens, to see them refreshed
```

In the copy, set `SUPABASE_URL` to `http://127.0.0.1:54321` and put the same address in the CSP's `connect-src`.

## Security model

Sign-in and passwords are Supabase's job. The page:

- sends passwords only to Supabase over HTTPS, never stores them, and keeps the session tokens in `localStorage` or `sessionStorage` as described above; a deleted or signed-out-everywhere account is noticed on the next visit;
- relies on Row Level Security, not on the page, to keep each account's data private: the publishable key alone can read nothing;
- builds all DOM with `textContent` and element APIs (no `innerHTML`), and enforces Trusted Types in Chromium browsers. Its one exception is a default policy that lets the face tracker load its helper script, and only same-origin scripts under `/vendor/mediapipe/`;
- sets a strict Content-Security-Policy: scripts only inline or from this site (the bundled face tracker, with `'wasm-unsafe-eval'` for its WebAssembly), `fetch` only to its own files (`latest.json`, the face model) and the Supabase project, WebSockets to loopback only, images only from `data:` URLs (the camera view);
- checks every message from EyeSync: status fields are typed and length-limited, camera pictures must be base64 JPEG data URLs, and `latest.json` file names must look like `EyeSync.exe` or `EyeSync-1.2.0.zip`;
- never asks for or stores EyeSync's API key, and never stores camera pictures;
- refuses to run sign-in when framed by another site (clickjacking guard), since GitHub Pages can't send `frame-ancestors` headers.

## Legal text

The Cookie Policy and Terms and Conditions in `index.html` (`#tpl-cookies`, `#tpl-terms`) are starting templates. Have them reviewed before relying on them, and bump the version constants (`TERMS_VERSION`, `COOKIE_POLICY_VERSION`) when they change.
# Brahmastra EyeSync build

This copy uses the calibrated gaze, native Windows mouse, smile left click,
mouth-open right click and command panel from the desktop project while
retaining this project's offline voice typing. See [BRAHMASTRA_README.md](BRAHMASTRA_README.md)
for how to run it with Python 3.15.2. The older documentation below describes
the original upstream release.

