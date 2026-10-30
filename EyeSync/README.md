# EyeSync

Control the mouse cursor with your head using an ordinary webcam, and click
by blinking or winking, or by holding the cursor still. No special hardware
and no hands needed. EyeSync waits quietly in the background and switches
on only while the EyeSync website dashboard is open.

- **Camera:** any webcam. Tracking runs at about 30 frames per second.
- **Mouse:** moves the Windows cursor directly, or uses a Seeed XIAO ESP32-S3
  plugged in as a USB mouse when one is connected.
- **Voice typing:** close your eyes for 2.5 seconds and say what you want to
  type. It is typed wherever the text cursor is, in any program. Speech is
  recognised on this PC; nothing is sent anywhere.
- **Website:** [the EyeSync website](https://aniruddh9140.github.io/face-control-website/)
  receives live position and click data over WebSocket on
  `ws://localhost:8765` (also `ws://localhost:8080/eye-tracker`). On its
  Dashboard, press **Connect**.

## Install

EyeSync is **one file, `EyeSync.exe`**: the whole application, nothing
else needed (no Python, no administrator rights).

1. Download **EyeSync.exe** from the EyeSync website.
2. Double-click it. If Windows says "Windows protected your PC", choose
   **More info**, then **Run anyway** (EyeSync isn't signed by a paid
   certificate).
3. Press **Install**, then **Finish**: the EyeSync website opens. Press
   **Connect** on its Dashboard.

It copies itself to `%LocalAppData%\Programs\EyeSync`, adds an **EyeSync**
shortcut to the Desktop and Start menu, and sets EyeSync to wait for the
website at every Windows login. You can delete the downloaded file
afterwards. Its first start takes about half a minute; after that, the
website's Connect starts it in a few seconds.

**Opening EyeSync from the website:** pressing **Connect** on the
Dashboard starts EyeSync by itself. If nothing is waiting in the background
(for example, after the login start was switched off in Windows), the
website's **Open EyeSync** button opens it. The browser first asks whether
to open EyeSync.

**Update:** download the new EyeSync.exe and open it (**Update**). Your
settings, calibration and API key stay. **Remove:** Windows Settings > Apps > Installed apps >
EyeSync > **Uninstall**.

### Install from the zip (Python version)

For developers, or if EyeSync.exe can't be used:

1. Download the EyeSync zip from the EyeSync website.
2. Right-click it, choose **Extract All...**, then **Extract**.
3. In the extracted folder, double-click **`install.bat`**. If Windows
   says "Windows protected your PC", choose **More info**, then **Run
   anyway**. Wait a few minutes.

It:

- installs EyeSync in `%LocalAppData%\Programs\EyeSync`, so afterwards you
  can delete the zip and the extracted folder
- installs Python if the PC doesn't have it yet
- installs everything else EyeSync needs, and checks that it works
- adds an **EyeSync** shortcut to the Desktop and Start menu
- sets EyeSync to wait for the website, now and at every Windows login
- opens the EyeSync website: press **Connect** on its Dashboard, and EyeSync
  starts

The **EyeSync** shortcut opens its window any time. Needs Windows 10 or 11,
any webcam, and an internet connection during setup.

**Update:** download the new zip and run its `install.bat` the same way.
Your settings, calibration and API key stay.

**Remove:** double-click **`uninstall.bat`** in
`%LocalAppData%\Programs\EyeSync`. It stops EyeSync, removes the shortcuts
and the start at login, and asks before deleting the folder.

### Sharing EyeSync with other people

Run **`py -3.15 build_exe.py`** (about 5 minutes). Everything to give out
goes to a separate folder next to this one, **`..\EyeSync-Release`**:

- `EyeSync.exe`, the application and the download
- `latest.json`
- the website docs
- `READ ME FIRST.txt`, which says what each file is for

The build runs the self-tests inside the built program, and again in a
test install. The program contains none of your personal data.

For the zip (Python) version, run **`python make_release.py`** first; the
zip goes to the same release folder. It builds `dist/EyeSync-<version>.zip`
with only the program - none of your personal data (settings, calibration,
log or API key) - and runs the self-tests on the unpacked copy. It also
writes `dist/latest.json`, which the website uses to offer updates. Put both
on the website (see `WEBSITE.md`). Each person gets their own settings and
API key.

### Code signing

Windows shows "Windows protected your PC" for programs that aren't
code-signed, and some antivirus programs distrust them. Signing needs a
code-signing certificate: from a certificate authority (it comes on a USB
token or in the authority's cloud), or Microsoft's own cloud signing
(Azure Trusted Signing) where it's offered to you. Even signed, the warning
fades only as more people download the program.

With a certificate, set **`EYESYNC_SIGN`** to signtool's arguments before
building, and the build signs `EyeSync.exe` before its self-tests (so the
file tested is the file given out), then checks the signature. For a
certificate in the Windows certificate store, for example:

    set EYESYNC_SIGN=/fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /sha1 <thumbprint>

`signtool.exe` comes with the Windows SDK; the build finds it there (or set
`EYESYNC_SIGNTOOL`). If signing is asked for and fails, the build stops.
Since 1.5.6 the exe also carries its name and version (Properties >
Details).

### Testing EyeSync on someone new

EyeSync's blink, gesture and tilt settings were measured on one person's
face. To check them on someone else, ask them to press **Blink test** at
the bottom of the EyeSync window. EyeSync hands over the camera for about
4.5 minutes of guided parts: reading, clicking, right-clicking, moving the
head, clicking in the corners, tilting, the mouth gestures and reading
aloud. It then opens Explorer at the saved file (in the `recordings` folder
next to EyeSync) for them to send to you. The file holds only numbers, no
pictures, plus their calibration. Then run
`python tools/replay_blinks.py <their file>`: it says how many clicks,
scrolls and gestures worked for them, and what went wrong.

### Tested on public video of other people

`tools/video_features.py` runs video files through EyeSync's own tracking
and saves the same numbers as the Blink test; `tools/dataset_report.py`
then says what EyeSync would have done. Nobody in a dataset is using
EyeSync, so every click, gesture or scroll is a false one. Since 1.5.7
EyeSync is checked this way on the **GRID audiovisual corpus** (Cooke,
Barker, Cunningham & Shao, 2006; CC BY 4.0;
https://zenodo.org/records/3625687): ten people reading sentences aloud,
250 minutes. Talking set off no scroll, quit, recenter or calibrate; one
reader's real smiles paused EyeSync 3 times in 25 minutes. Blinks that
reopen slowly clicked about 4 times a minute on the first four readers, so
the tick now comes only while the eyes are shut: 3 a minute on them, 1.2 on
six readers EyeSync had never seen, with no false right clicks. Most of
the rest come from one reader's long closures. A tick at 0.45 s would bring
all ten down to 0.7 a minute, but it lost quick meant clicks, so the tick
stays at 0.3 s (1.5.13). Put the numbers
in `..\EyeSync-datasets\grid\features2` and the tests check this too.

## Daily use

Everything is in the one EyeSync window:

- **Status** at the top, with a coloured dot:
  - grey: off, waiting for the EyeSync website
  - green: on, you control the cursor
  - red: on, but it can't see your face (check the lighting and that you're in view)
  - amber: paused
- **Camera view:** what EyeSync sees. Advice appears just above it, for
  example when the room is too dark.
- **Pause**, **Recenter**, **Calibrate**, **Voice typing** and
  **Calibrate eyes** buttons
- **Settings:** speed, how to click, show camera, face gestures, gaze jump,
  start with Windows, and the **API key** for other websites and apps (see
  below)
- **Help**, **Open log** and **Quit EyeSync**

The window works with the EyeSync cursor too, so you never need the
keyboard. Closing the window (X) only minimises it. Use **Quit EyeSync** to
end it.

**EyeSync opens only with the website dashboard.** After Windows starts,
nothing opens: EyeSync waits invisibly (no window, no camera). When you
open the EyeSync website dashboard, EyeSync starts by itself (minimised in
the taskbar, ready in a few seconds). When you close the dashboard, EyeSync
closes again 30 seconds later and waits for the next time. The **EyeSync**
shortcut opens it at any time.

**After you quit EyeSync** (mouth wide open for 3 s, **Quit**, or F12), it
stays closed even though the dashboard keeps trying to reconnect: it starts
again only when you press **Connect** on the dashboard. The dashboard shows
why the connection ended ("EyeSync was closed - press Connect to start it
again").

1. **Open the EyeSync website.** EyeSync activates within a few seconds:
   the webcam light comes on and you control the cursor. When you close the
   website, cursor control stops after 4 seconds (so reloading the page or
   following a link inside the site never interrupts you) and the camera
   turns off a second later.
2. **Move the cursor with your head.** The cursor follows your nose the way
   it follows a mouse: turn your face a little to the right and the cursor
   goes right. Like a mouse, **slow movements are precise** (for small
   buttons) and **a normal turn goes far**: about 23 degrees crosses the
   whole screen at speed 1. The cursor glides smoothly, updating 120 times
   a second, even though the webcam sees you only 30 times a second.
   Holding your head still holds the cursor still,
   and nodding while you blink never moves it. If your head ends up turned
   uncomfortably, push the cursor against the screen edge and turn back,
   press **Recenter** (or pull your mouth to your left for 1.5 s), or use
   gaze jump (look at the spot, nudge your head).
   It works without calibrating, but **Calibrate** (see below) fits it to
   you.
3. **Click**, in one of two ways (Settings → **Click by**):
   - **Blink** (the default): close both eyes and listen. A short high **tick**
     (0.3 seconds into the blink, also after calibrating - until 1.5.12
     calibrating could move it to 0.35-0.45 s, slower than many quick
     clicks) means "open now and it clicks";
     open your eyes and it left-clicks. No tick, no click, so
     ordinary quick blinks are ignored - and while your head is moving
     (reading, looking around) the tick comes a little later (at least about
     0.4 s), so blinks while you read don't click.
     You hear a click sound every time a click registers. Works with
     glasses.
     **Right click:** keep your eyes closed past the tick until the
     **double tick** (a second after the tick: about 1.3 seconds), then open
     them (two short beeps confirm it). Most people can't wink one eye cleanly, but everyone can
     blink a little longer.
     **No need to hurry:** after each tick you have time to open your eyes -
     after the tick it's still a click until the double tick,
     and after the double tick still a right click until about 2.4 seconds.
     Longer is resting: no click. 2.5 seconds is voice typing (see below).
     (Until 1.5.1 the double tick came at 0.8 s and resting at 1.5 s; until
     1.5.3 the tick came at 0.2 s and the double tick at 1.2 s.)
     **Winks** work too if you can wink cleanly: the right eye alone is a
     right click, the left eye alone a left click. Winks are ignored while
     your head is turned far to one side. It's only a wink while the other
     eye stays open: if one of your eyes closes later or less than the
     other, your blinks are still blinks, with both ticks and voice typing.
     After a smile gesture, blinks count again once your face has relaxed.
     **You can blink while moving or tilting your head.** A deliberate
     blink (eyes fully shut until the tick) clicks where the cursor was
     when you closed your eyes. Once you hear the tick, turning your head
     no longer cancels the click (until 1.5.10 a head drifting during a
     long closure could). Quick natural blinks while you turn don't click
     and don't stop the cursor.
     **Moving your head alone never clicks.** Looking at the bottom of the
     screen makes your eyes look half closed to the camera. EyeSync learns
     how your open eyes look in every head position, and with the head
     moving it only counts eyes that close fully.
   - **Hold still:** move the cursor onto something and keep it still for
     1 second. You hear a click sound. To click again, move away first, so
     resting on a page never clicks repeatedly. Choose this if deliberate
     blinking is hard for you.
4. **Scroll:** tilt your head, an ear towards a shoulder, and hold it.
   **Left ear down scrolls up, right ear down scrolls down.** Tilt further
   to scroll faster; straighten your head to stop. The cursor stays where
   it was while you scroll, and the camera view says "Scrolling up" or
   "Scrolling down". A quick tilt or glancing at a screen corner doesn't
   scroll.
5. **Type by talking:** see **Voice typing** below.

### Voice typing: type by talking

1. **Click (blink) where the text should go**, such as a message box,
   a document or a search field.
2. **Close both eyes for 2.5 seconds.** Keep them closed past the tick and
   the double tick (the clicks) until you hear two rising tones, then open
   them. Or press
   **Voice typing** in the EyeSync window, Ctrl+Alt+Shift+F5, or the
   website's voice typing button.
3. **Speak normally, in sentences.** A bar at the bottom of the screen shows
   that EyeSync is listening (green), hearing you (red) or typing (amber),
   and shows the last thing it typed. Each time you pause (about ¾ of a
   second), what you said is typed 1–2 seconds later, with capitals and
   punctuation. Pausing in the middle of a sentence is fine; it carries on
   the same sentence.
4. **To stop**, do any of these:
   - close your eyes for 2.5 seconds again (two falling tones);
   - say **"stop listening"**;
   - press **Stop** on the bar;
   - just stop talking: it stops by itself after 8 seconds of silence, and
     the bar counts down the last 3.

Say these on their own, with a short pause before and after:

| Say | Does |
|---|---|
| **"new line"** | Starts a new line (Shift+Enter, so a chat box doesn't send the message) |
| **"new paragraph"** | Two new lines |
| **"press enter"** | Presses Enter, for example to send a message or search |
| **"delete that"** (or "scratch that") | Deletes the last thing it typed; say it again for the one before |
| **"backspace"**, **"tab"**, **"space"** | Presses that key |
| **"comma"**, **"period"**, **"question mark"**, **"exclamation mark"**, **"colon"** | Types that mark, replacing the one Whisper put there |
| **"stop listening"** | Ends voice typing |

"New line" and "stop listening" also work at the end of a sentence ("…see
you soon, new line").

- **Private:** the microphone is on only while voice typing is, and you
  always hear and see when it is. What you say is turned into text on
  this PC, by Whisper (OpenAI's speech model, run offline); nothing is sent
  anywhere. Your words aren't written to the log and never go to the website.
- **Noise:** fans, keyboards and background sounds are recognised as
  not being speech and never turn into text. Another person talking near
  the microphone will be typed, though.
- **English only**, with any accent: its settings were tested on Indian,
  Pakistani, Sri Lankan and American speakers. Speak at a normal pace and
  volume, a whole sentence at a time; a headset microphone is the most
  accurate in a noisy room.
- If the EyeSync window is in front (you pressed its button), the text goes
  to the window you used before it. Windows doesn't let any program type
  into programs running as administrator; the bar says so.
- While voice typing listens, smiling and moving your mouth as you talk
  don't trigger the smile or mouth-to-the-side gestures. Head-tilt
  scrolling still works.

### Calibrate: fit EyeSync to you (30 seconds)

EyeSync works without calibrating, but calibrating makes it fit *you*.
Press **Calibrate** in the window (or Ctrl+Alt+Shift+F10, or pull your
mouth to your right for 3 seconds). The camera
turns on for it even if the website isn't open. A full-screen guide then
shows you what to do, and moves on by itself; you never need to click:

1. Look at the dot in the middle and keep still.
2. Turn your head (not just your eyes) towards the dots on the left, right,
   top and bottom edges, holding still at each.
3. Blink slowly 3 times, keeping your eyes closed for about half a second
   each time - the way you would close them to click.
4. Give a big smile.

What it changes:

- **Speed:** the head movement you naturally make to look across the
  screen now moves the cursor across it at speed 1, separately for
  left-right and up-down. You no longer have to turn too far, and the cursor
  doesn't fly off. It fine-tunes the speed by up to 30% either way, so one
  rushed step can't make the cursor much too slow or too fast. EyeSync
  switches to speed 1 afterwards; the higher speeds are still faster.
- **Blink clicks:** fitted to how far your eyes close (with or without
  glasses). The tick stays at 0.3 s: since 1.5.13 calibrating no longer
  makes it later. (1.5.9-1.5.12 set it from how long your slow calibration
  blinks stayed shut - up to 0.65, 0.45 and then 0.35 s - but quick meant
  clicks then opened before the tick and were lost.) The double tick comes a
  second after the tick. EyeSync also
  learns how your open eyes look at each screen edge, so looking there is
  never taken as a blink.
- **Smile gesture:** set just below your big smile, so a light smile stays
  well under it.

It's saved and used every time from then on. Calibrate again whenever
something changes, for example new glasses, a new seat or a new camera.
Press **Cancel** or Esc to stop; your previous settings stay. A step you
don't finish within 20 seconds is skipped and keeps its default.

### Gaze jump: look, then nudge (20 seconds to set up)

Moving the cursor across a big screen takes a big head turn. With **gaze
jump**, your eyes do the long trip: **look at the spot you want, then start
moving your head towards it**. The cursor jumps there, and your head does
the fine aiming as always. The camera view says "gaze jump" when it happens.

The cursor only jumps when you ask with your head. Looking around, reading
or keeping your head still never moves it. It also doesn't jump for short
distances, where your head is quicker and more precise than your eyes.

To set it up, press **Calibrate eyes** in the window. A dot visits 9
places on the screen; just look at each one. You can move your head
naturally, because EyeSync follows both your head and your eyes. When it's
done, the screen says how accurately it knows where you look, for example
"about 120 px". Gaze jump switches on if that's accurate enough. If not,
add light on your face or sit a little closer, and calibrate again. Switch
it on or off with **Gaze jump** under Settings.

Recalibrate your eyes after moving the camera, the screen or your seat.
The eye tracking comes from JEOresearch's webcam 3D eye tracker
([EyeTracker](https://github.com/JEOresearch/EyeTracker), MIT licence),
reworked and given the 9-point calibration it didn't have yet (see
`gaze_engine.py`).

### Face gestures

Hold an expression for 1.5 seconds to change a setting. A sound confirms
it, and the camera view shows a progress bar while you hold.

| Gesture | Does |
|---|---|
| Close both eyes, open at the **tick** (~⅓ s) | Left click |
| Close both eyes, open at the **double tick** (~1.3 s) | Right click (two short beeps) |
| Wink the left / right eye, if you can wink | Left / right click |
| **Big smile** (a wide, toothy one) | Pause / resume |
| **Mouth pulled clearly to your left** | Recenter the cursor |
| **Mouth pulled clearly to your right, 3 s** (a beep, then a countdown) | Calibrate |
| **Head tilted to the left** (left ear towards the left shoulder), held | Scroll up, while you hold it |
| **Head tilted to the right** (right ear towards the right shoulder), held | Scroll down, while you hold it |
| **Both eyes closed, 2.5 s** (rising tones) | Voice typing on; again (falling tones) = off |
| **Mouth wide open, 3 s** | Quit EyeSync |

For scrolling, tilt clearly (about 22 degrees more than you usually hold
your head) for a third of a second; it then scrolls until you straighten
up, faster the further you tilt. Turning to a screen corner, which leans
many people's heads a little, or a head you usually hold a little tilted,
doesn't count: EyeSync learns your normal tilt. (Until 1.5.4 a tilt counted
from 15 degrees, and at the far corners the cursor could stop, jump back
and ignore blinks.) While EyeSync is paused, or the website isn't connected, a tilt does
nothing. (Before 1.4.0, head tilts recentered and calibrated.)

For the mouth gestures, pull your lips clearly towards one cheek: your
left (the side of your left ear) recenters, your right calibrates. EyeSync
goes by which way your mouth actually moved, measured from your face (a head
turn is taken out), and learns where your mouth normally rests. A pull that
lifts that corner of your mouth counts too; it isn't taken for a smile. Keep
your lips closed: with your mouth open (talking) nothing fires. Calibrating takes about 30 seconds
and can't be stopped hands-free once it starts, so that gesture needs
3 seconds: half a second in, a beep sounds and the camera view counts down
("Calibrating in 2.1s"); relax your mouth to cancel. Since 1.5.14 it starts
only when the face tracker sees the mouth itself move to the side: a smile
that rises more on one side no longer beeps (1.5.11 did, now and then, just
before a smile pause). Changing the speed
is no longer a gesture (until 1.5.0 the mouth to one side did that): use
the window, F7 / F8 or the website.

Talking, a light smile and opening your mouth for a moment don't trigger
anything. The quit countdown beeps only after the mouth has stayed wide
open for half a second. Gestures work with your head turned towards a
screen edge too. Turn gestures off in Settings if a long natural smile
still pauses EyeSync by mistake.

### Keys

Everything is also in the window. These optional shortcuts work
everywhere, even while EyeSync is minimised. Hold **Ctrl + Alt + Shift** and press:

| Key | Action |
|---|---|
| **F5** | Voice typing on / off |
| **F6** | Recenter: puts the cursor in the middle of the screen |
| **F7** / **F8** | Speed down (calmer) / up (faster). Five levels; your choice is saved. |
| **F9** | Pause or resume cursor control |
| **F10** | Calibrate (press again to cancel) |
| **F11** | Show the EyeSync window |
| **F12** | Quit EyeSync |

**Quit without the keyboard:** open your mouth wide and hold it for
3 seconds. A beep sounds when the countdown starts; close your mouth to
cancel.

### In a dark room

EyeSync keeps working in dim and dark rooms: when the camera picture gets
too dark and grainy to find your face, it cleans up and brightens the
picture by itself (the log says "Low light"). Your screen also lights your
face, so turning up the screen brightness helps a lot. In complete
darkness no ordinary webcam can see you; the window then says it's too
dark.

## API key: use EyeSync from other websites or apps

Out of the box, only the EyeSync website can connect to EyeSync. Any other
website or program has to show EyeSync's **API key**, so a random web page
you visit can't switch your cursor on or read where it is.

The key is made automatically the first time EyeSync starts, is different on
every PC, and stays the same until you replace it. In the window, under
Settings → **API key**:

- **Copy** puts the key on the clipboard. The window only shows its first
  and last few characters, so screen sharing doesn't reveal it.
- **New key** replaces it. Anything using the old key is disconnected at
  once and needs the new key. The EyeSync website is not affected.

To connect, add the key to the address. Anything connected with the key
switches EyeSync on, just like the website does:

```js
const key = "PASTE-YOUR-KEY-HERE";      // better: let the user paste it in, and keep
                                        // it in localStorage - never publish it
const ws = new WebSocket("ws://localhost:8080/eye-tracker?key=" + encodeURIComponent(key));
ws.onmessage = (e) => {
  const { x, y, blink, confidence, face } = JSON.parse(e.data);
  // x, y: cursor position in a 1920x1080 space; blink: true on a click
};
ws.onclose = (e) => {
  if (e.code === 4001) console.log("EyeSync refused:", e.reason);
  // "API key required", "invalid API key" or "API key changed"
};
```

Messages arrive about 30 times a second. Port 8765 works the same way.
Websites and apps can also send commands (enable/disable mouse control,
recenter, speed, click style): see **`WEBSITE.md`**.
Don't put the key in a public website's code: anyone could read it there.
Instead, have each user paste their own key into the page.

## Troubleshooting

| Problem | Fix |
|---|---|
| Your head is turned uncomfortably to reach things | Push the cursor against the screen edge, then turn back, or press **Recenter**. |
| Too much head movement needed, or too jumpy | Press **Calibrate** first. Then fine-tune with **Speed** in Settings: higher needs less head movement, lower is steadier. At speed 1 a head turn of about 23 degrees crosses the whole screen; speed 5 needs about 12. |
| The website can't find EyeSync after you restart the PC | EyeSync may be switched off in Windows (Settings > Apps > Startup, or Task Manager > Startup apps). Tick **Start with Windows** in the EyeSync window to switch it back on, or open the EyeSync shortcut once. |
| Website shows "not connected" | Open the **EyeSync** shortcut, or close and reopen the dashboard. If it still doesn't connect, press **Open log** in the window and look for errors about ports 8080 or 8765. |
| Another site or app can't connect | It needs the API key (see above). The log shows `Refused a connection ... API key required` or `invalid API key`. |
| A click when you only moved your head, or a blink that didn't click | Close your eyes fully until the tick (0.3 s; 0.4 s when the blink starts while your head moves). The log says `Head moved while the eyes only looked half closed ... not a blink` when it ignored a closure, with how far the eyelids closed and how far the head moved (sideways or up-down); `Eyes closed while the head moved, but the eyelids never shut fully` is the same for a closure that began while you were turning. Since 1.5.13 every closure of 0.3 s or more that didn't click is logged too (`too short for a click` or `the eyes were opening before the tick`). If it still happens, send `eyesync.log`. |
| "It's too dark for the camera" | Turn on a light, or turn up your screen brightness (the screen lights your face). |
| You plugged in a monitor, pressed Win+P, or changed the resolution | Nothing to do: within 2 seconds EyeSync follows the new size of your main screen (the log says `Display changed`). The cursor moves on the main screen only; make the screen you want to use the main one in Windows Settings > System > Display. |
| "Can't open the camera" | Another app (Zoom, Teams, a browser tab) is using the webcam, or camera access is off in Windows Settings > Privacy > Camera. Fix that; EyeSync retries every 2 seconds and also tries a second or third camera. |
| Voice typing: closing your eyes doesn't start it | Keep both eyes closed until you hear the two rising tones (2.5 s). The camera view shows a "keep closed: voice typing" bar. Face gestures must be on in Settings. |
| Voice typing types nothing | Click into a text box first. If the bar says "No sound from the microphone", allow it in Windows: Settings > Privacy & security > Microphone > *Let desktop apps access your microphone*. EyeSync uses the Windows default microphone (Settings > System > Sound > Input). |
| Clicks don't work in some windows (Task Manager, installers, the "Do you want to allow this app to make changes" screen) | Windows doesn't let ordinary programs click in windows that run as administrator, or on that secure screen. The XIAO hardware mouse (below) works there, because Windows sees it as a real mouse. |
| Voice typing gets words wrong | Speak in whole sentences at a normal pace, close to the microphone, and pause between them. Say **"delete that"** to remove the last sentence. |

Everything EyeSync does is logged to **`eyesync.log`** in this folder. Every
5 minutes it also writes a `[health]` line with the frame rate, how often your
face was tracked, and any errors. If EyeSync ever closes by itself, the launcher
restarts it within seconds; `eyesync_crash.log` then says where it crashed.

## The XIAO ESP32-S3 as a hardware mouse (optional)

Flash `firmware/xiao_esp32s3_mouse/xiao_esp32s3_mouse.ino` with the Arduino
IDE. The board settings are at the top of that file; you need **USB Mode:
USB-OTG (TinyUSB)** and **USB CDC On Boot: Enabled**.

Plug the board in and EyeSync uses it automatically. Unplug it and EyeSync
goes back to moving the cursor itself. The board's BOOT button switches the
mouse output on and off.

## Your data

EyeSync stores no camera pictures and no sound. Nothing leaves this PC except the
cursor position and clicks it sends to the EyeSync website (and to apps
you give the API key). If you turn on the camera view on the website, the
small camera picture goes to that page in your browser too, only while it's
shown. Voice typing uses the microphone only while it's on, recognises
speech on this PC, and keeps nothing: your words go only into the text box
you're typing in. Where you look (gaze jump) is worked out on this PC and
never sent anywhere; only the eye calibration's result is saved, in
EyeSync's settings. The **Blink test** saves numbers only (how open the
eyes are, head angles, expression scores), in the `recordings` folder on
this PC; nothing is sent unless you send the file yourself.

## Files

| File | Purpose |
|---|---|
| `live_inference.py` | The app: camera, face tracking, head pointer, blink and wink clicks, website link, hotkeys |
| `face_engine.py` | Camera access, low-light handling, MediaPipe face mesh |
| `gaze_engine.py` | Eye gaze: the eye model (adapted from JEOresearch's EyeTracker), the 9-dot eye calibration, gaze jump |
| `THIRD_PARTY_NOTICES.md` | Licences of code EyeSync adapts (EyeTracker, MIT) |
| `voice_typing.py` | Voice typing: microphone, cutting speech into phrases, Whisper speech recognition, voice commands, typing |
| `models/` | The face model, and the speech models: Whisper small.en q8_0 (`ggml-small.en-q8_0.bin`, MIT licence) and Silero voice detection (`ggml-silero-v6.2.0.bin`, MIT) |
| `desktop.py`, `software_mouse.py` | Windows hotkeys, sounds, background priority, cursor control |
| `site_access.py` | Who may connect (the website, the API key) and the standby that starts EyeSync when the website opens |
| `main_window.py` | The EyeSync window |
| `make_release.py` | Builds the zip (Python version) into `..\EyeSync-Release` (no personal data) |
| `install.bat`, `uninstall.bat` | Install or update the zip (Python) version; remove it |
| `build_exe.py` | Builds `EyeSync.exe`, the whole application in one file (no Python needed) |
| `eyesync_app.py`, `installer.py` | The program inside EyeSync.exe; how it installs and updates itself |
| `launcher.py`, `app_paths.py` | Starts EyeSync and restarts it after a crash; where its files are |
| `autostart.py`, `EyeSync.pyw` | Start at login, shortcuts, restart after a crash. `python autostart.py uninstall` removes autostart and the shortcuts. |
| `WEBSITE.md` | How EyeSync and the website talk: connecting and starting EyeSync, data, commands, status, camera view, downloads |
| `WEBSITE_CHANGES.md` | The to-do list for the website (download page, controls) |
| `tests/` | Self-tests: `python -m unittest discover -s tests` |
| `tools/` | Checks for developers (not in the zip): `record_blinks.py` (also the window's **Blink test**) + `replay_blinks.py` measure blink clicks, gestures and tilt scrolling on a face; `video_features.py` + `dataset_report.py` do it for public videos of other people; `soak.py` runs an hour of simulated use through the real tracking code, to check that memory and speed stay flat (run it before a release) |
| `eyesync_api_key.txt` | Your private API key. Don't share it; **New key** in the window replaces it. |
# Brahmastra controls

This local copy uses the desktop project's calibrated gaze model, native mouse
output, smile left click and mouth-open right click. Offline voice typing from
this project is retained. See `../BRAHMASTRA_README.md` for the launcher and
current limitations. The older documentation below describes the original
upstream controls.

