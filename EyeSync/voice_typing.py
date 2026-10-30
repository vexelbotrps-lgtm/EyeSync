"""
voice_typing.py - type by talking (speech recognition on this PC).
==================================================================

Close both eyes for 1 second and say what you want to type: it is typed
where the text cursor is, in any program. Close both eyes for 1 second again
to end the session, or click Stop on the voice bar. Silence does not end it.

How it works
  * Microphone (sounddevice): on ONLY while voice typing is - a sound and the
    bar at the bottom of the screen (main_window.VoiceOverlay) show it.
  * SpeechSegmenter cuts what you say into phrases at your pauses, using
    Silero voice detection (SileroVad) every 32 ms - so a fan, typing or a
    microphone's noise bursts never count as talking.
  * Recognizer: each phrase is turned into text by Whisper (OpenAI's speech
    model, the high-accuracy English "small" size) running in whisper.cpp - offline,
    nothing leaves the PC. Silero voice detection inside it drops phrases
    that hold no speech: without it, Whisper "hears" words in noise
    ("Thank you.", "(buzzing)"). ~1-1.5 s per phrase on a 4-core laptop,
    on 3 threads (a core stays free for the face tracking) that don't hold
    up the rest of EyeSync (whisper.cpp releases Python's GIL). Its
    settings were chosen on real accented speech (see Recognizer).
  * Composer joins the phrases like normal writing (spaces, a sentence
    carried on over a pause isn't capitalised) and turns voice commands
    (COMMANDS: "new line", "delete that"...) into keys.
  * Typist types it as keyboard input (SendInput with Unicode characters, so
    any keyboard layout works) into the window in front.
  * Your words are never written to the log - only how many characters
    were typed - and are never sent to the website.

Like the software mouse, it can't type into programs running as
administrator (Windows doesn't allow it); it says so instead.
"""

from __future__ import annotations

import ctypes
import importlib.util
import logging
import math
import os
import queue
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

import app_paths
import desktop

log = logging.getLogger("eyesync")

RATE = 16000                         # Whisper's sample rate
MODELS_DIR = app_paths.APP_DIR / "models"
MODEL_FILE = "ggml-small.en-q8_0.bin"  # high-accuracy English Whisper; see Recognizer
VAD_FILE = "ggml-silero-v6.2.0.bin"      # Silero voice detection (noise -> no text)
END_PAUSE_S = 0.7        # a pause this long ends a phrase (then it is typed)
MAX_PHRASE_S = 20.0      # longer without a pause: cut at the quietest moment
CONTEXT_S = 300.0        # phrases this close together in one window join up as one text
MIC_SILENT_S = 1.5       # a microphone giving pure zeros this long is blocked by Windows
START_SOUND_S = 0.4      # the start sound (0.18 s) and the microphone's delay: not listened to
STATES = ("starting", "listening", "hearing", "working", "error")   # Snapshot.voice ("" = off)


def available() -> str:
    """"" if voice typing can work here, else why not (shown to the user)."""
    if sys.platform != "win32":
        return "Voice typing needs Windows"
    for name in (MODEL_FILE, VAD_FILE):
        if not (MODELS_DIR / name).is_file():
            return f"Voice typing is missing its speech model ({name})"
    for module in ("_pywhispercpp", "sounddevice"):
        if importlib.util.find_spec(module) is None:
            return f"Voice typing is missing a part ({module}) - reinstall EyeSync"
    return ""


# --------------------------------------------------------------------------- #
# Phrases: cut the microphone's sound at the pauses
# --------------------------------------------------------------------------- #

def _whisper_library() -> Path:
    """whisper.cpp's DLL, as loaded by _pywhispercpp (next to it, also inside EyeSync.exe)."""
    import _pywhispercpp
    here = Path(_pywhispercpp.__file__).parent
    found = sorted(here.glob("whisper*.dll")) or sorted(Path(getattr(sys, "_MEIPASS", here)).glob("whisper*.dll"))
    if not found:
        raise OSError("whisper.cpp's library wasn't found")
    return found[0]


class SileroVad:
    """
    Silero voice detection - a small neural network that tells speech from
    every other sound - giving the probability of speech in each 32 ms
    (CHUNK) of audio, and remembering what it heard between calls. The
    same model the Recognizer uses inside whisper.cpp, called straight
    from whisper.cpp's library (ctypes): pywhispercpp's wrapper for it is
    broken (its context type isn't registered). ~0.3 ms per chunk.
    """

    CHUNK = 512

    class _Params(ctypes.Structure):           # whisper_vad_context_params
        _fields_ = [("n_threads", ctypes.c_int), ("use_gpu", ctypes.c_bool),
                    ("gpu_device", ctypes.c_int)]

    def __init__(self, model: Path = MODELS_DIR / VAD_FILE):
        lib = ctypes.CDLL(str(_whisper_library()))
        floats = ctypes.POINTER(ctypes.c_float)
        lib.whisper_vad_init_from_file_with_params.argtypes = (ctypes.c_char_p, self._Params)
        lib.whisper_vad_init_from_file_with_params.restype = ctypes.c_void_p
        lib.whisper_vad_detect_speech_no_reset.argtypes = (ctypes.c_void_p, floats, ctypes.c_int)
        lib.whisper_vad_detect_speech_no_reset.restype = ctypes.c_bool
        lib.whisper_vad_n_probs.argtypes = lib.whisper_vad_probs.argtypes = (ctypes.c_void_p,)
        lib.whisper_vad_n_probs.restype = ctypes.c_int
        lib.whisper_vad_probs.restype = floats
        lib.whisper_vad_free.argtypes = lib.whisper_vad_reset_state.argtypes = (ctypes.c_void_p,)
        self.lib, self.floats = lib, floats
        self.ctx = lib.whisper_vad_init_from_file_with_params(str(model).encode("utf-8"),
                                                             self._Params(1, False, 0))
        if not self.ctx:
            raise RuntimeError(f"could not load {Path(model).name}")
        # A new context's memory isn't cleared: without this every chunk scored 0.35.
        lib.whisper_vad_reset_state(self.ctx)

    def __call__(self, chunk: np.ndarray) -> float:
        """Speech probability (0..1) of one CHUNK of 16 kHz audio."""
        c = np.ascontiguousarray(chunk, np.float32)
        if not self.lib.whisper_vad_detect_speech_no_reset(self.ctx, c.ctypes.data_as(self.floats),
                                                           c.size):
            return 0.0
        n = self.lib.whisper_vad_n_probs(self.ctx)
        return float(self.lib.whisper_vad_probs(self.ctx)[n - 1]) if n > 0 else 0.0

    def close(self) -> None:
        if self.ctx:
            self.lib.whisper_vad_free(self.ctx)
            self.ctx = None


class SpeechSegmenter:
    """
    Cuts microphone audio into phrases at pauses (feed() returns the finished
    ones), by a voice detector's speech probability for every 32 ms (``vad``,
    SileroVad): a phrase starts at speech (START_P), goes on while it is
    likely (KEEP_P) and ends after END_PAUSE_S without. A little before and
    after is kept, so no word loses its first or last sound.

    Loudness alone was not enough: a real user's microphone silenced the
    room between words and let noise bursts through, which a loudness
    splitter took for talking - phrases ran on for 15-20 s, mixing speech
    and noise, and came out wrong or late.
    """

    CHUNK = SileroVad.CHUNK
    CHUNK_S = CHUNK / RATE        # 32 ms
    START_P, KEEP_P = 0.5, 0.35

    def __init__(self, vad, end_pause_s: float = END_PAUSE_S, preroll_s: float = 0.3,
                 tail_s: float = 0.3, max_s: float = MAX_PHRASE_S, min_speech_s: float = 0.25):
        self.vad = vad
        chunks = lambda s: max(1, round(s / self.CHUNK_S))              # noqa: E731
        self.end_chunks, self.tail_chunks = chunks(end_pause_s), chunks(tail_s)
        self.max_chunks, self.min_speech = chunks(max_s), chunks(min_speech_s)
        self.pre: deque[np.ndarray] = deque(maxlen=chunks(preroll_s))
        self.rest = np.zeros(0, np.float32)
        self.chunks: list[np.ndarray] = []      # the phrase being heard...
        self.probs: list[float] = []            # ... and its speech probabilities
        self.speaking = False
        self.silent = 0                         # chunks since the phrase was last speech
        self.level = 0.0                        # 0..1 loudness, for the microphone meter
        self.t = 0.0                            # seconds of audio fed
        self.last_voice_t = 0.0                 # ... when speech was last heard

    @property
    def quiet_s(self) -> float:
        """Seconds of audio since speech was last heard (or since the start)."""
        return self.t - self.last_voice_t

    def feed(self, samples: np.ndarray) -> list[np.ndarray]:
        x = np.concatenate([self.rest, np.asarray(samples, np.float32).ravel()])
        k = len(x) // self.CHUNK
        self.rest = x[k * self.CHUNK:]
        out = []
        for i in range(k):
            phrase = self._chunk(x[i * self.CHUNK:(i + 1) * self.CHUNK])
            if phrase is not None:
                out.append(phrase)
        return out

    def flush(self) -> np.ndarray | None:
        """The phrase still being heard (when voice typing stops), if it holds speech."""
        if not self.speaking:
            return None
        return self._finish(len(self.chunks) - self.silent + min(self.silent, self.tail_chunks))

    def _chunk(self, chunk: np.ndarray) -> np.ndarray | None:
        self.t += self.CHUNK_S
        p = self.vad(chunk)
        db = 20.0 * math.log10(float(np.sqrt(np.mean(chunk * chunk))) + 1e-9)
        self.level = float(np.clip((db + 60.0) / 45.0, 0.0, 1.0))
        if p >= self.START_P:
            self.last_voice_t = self.t
        if not self.speaking:
            self.pre.append(chunk)
            if p >= self.START_P:
                self.chunks = list(self.pre)
                self.probs = [0.0] * (len(self.chunks) - 1) + [p]
                self.pre.clear()
                self.speaking, self.silent = True, 0
            return None
        self.chunks.append(chunk)
        self.probs.append(p)
        self.silent = 0 if p >= self.KEEP_P else self.silent + 1
        if self.silent >= self.end_chunks:
            return self._finish(len(self.chunks) - self.silent + self.tail_chunks)
        if len(self.chunks) >= self.max_chunks:
            # Too long without a pause: cut where speech was least likely in
            # the last 1.5 s, and carry on with the rest as the next phrase.
            look = min(len(self.chunks) - 1, round(1.5 / self.CHUNK_S))
            cut = len(self.chunks) - look + int(np.argmin(self.probs[-look:]))
            phrase = self._audio(self.chunks[:cut], self.probs[:cut])
            self.chunks, self.probs = self.chunks[cut:], self.probs[cut:]
            return phrase
        return None

    def _finish(self, end: int) -> np.ndarray | None:
        chunks, probs = self.chunks[:end], self.probs[:end]
        for c in self.chunks[end:]:
            self.pre.append(c)
        self.chunks, self.probs = [], []
        self.speaking, self.silent = False, 0
        return self._audio(chunks, probs)

    def _audio(self, chunks, probs) -> np.ndarray | None:
        if sum(p >= self.START_P for p in probs) < self.min_speech or not chunks:
            return None                       # a cough or a bump, not a phrase
        return np.concatenate(chunks)


# --------------------------------------------------------------------------- #
# Speech -> text (Whisper in whisper.cpp)
# --------------------------------------------------------------------------- #

def _whisper_log(level, text) -> None:
    """whisper.cpp's messages: only its errors go to the log."""
    if level == 4:                           # GGML_LOG_LEVEL_ERROR
        log.warning(f"[voice] whisper.cpp: {str(text).strip()}")


class Recognizer:
    """
    One phrase of audio -> text. Whisper small.en q8_0 in whisper.cpp,
    loaded only when voice typing starts and freed when it stops (close()).

    Settings measured on real accented speech (EdAcc Indian, Urdu, Sinhalese
    and US English speakers, LibriSpeech; 119 phrases):
      * Whisper's own way of writing - in timestamped pieces - not one piece
        without timestamps: that stopped after the first sentence of a
        phrase now and then (a user's "wrong sentence"). 7 -> 3 phrases lost.
      * the encoder sees at least AUDIO_CTX_MIN (~20 s) of its 30 s window:
        less (~10 s) cost a quarter of the accuracy (20.6% -> 15.4% words
        wrong, 1.1 -> 1.6 s a phrase). The full 30 s was no better, only slower.
      * small.en q8_0 replaces base.en for stronger Indian-English accent
        recognition while remaining responsive on a four-core PC. The much
        larger large-v3-turbo model was rejected after taking over 90 seconds
        for the four-phrase end-to-end test on the supported reference PC.
      * Silero voice detection inside (vad): without it, more words wrong.
      * beam search (5 candidate wordings at a time): 2 -> 1 phrase lost,
        15.4% -> 15.2% words wrong, ~0.3 s more per phrase.
    max_tokens caps a runaway answer (repeating itself).
    """

    AUDIO_CTX_MIN = 1000        # of 1500 = 30 s

    def __init__(self, model: Path = MODELS_DIR / MODEL_FILE, vad: Path = MODELS_DIR / VAD_FILE,
                 threads: int | None = None):
        self.model_path, self.vad_path = Path(model), Path(vad)
        # A core fewer than the PC has (cpu_count counts each core twice): with all
        # 4 of a laptop's, the face tracking ran late 3x as often while it worked.
        self.threads = threads or max(2, min(6, (os.cpu_count() or 4) - 1))
        self._pw = self._ctx = self._params = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._ctx is not None:
                return
            import _pywhispercpp as pw
            pw.whisper_log_set(_whisper_log)            # unset again by close()
            t0 = time.monotonic()
            ctx = None
            try:
                ctx = pw.whisper_init_from_file_with_params(str(self.model_path),
                                                            pw.whisper_context_default_params())
            finally:
                if ctx is None:
                    pw.whisper_log_set(None)
            if ctx is None:
                raise RuntimeError(f"could not load {self.model_path.name}")
            p = pw.whisper_full_default_params(pw.whisper_sampling_strategy.WHISPER_SAMPLING_BEAM_SEARCH)
            p.beam_search = {"beam_size": 5, "patience": -1.0}
            p.n_threads = self.threads
            p.language = "en"
            p.translate = False
            p.no_timestamps = False          # Whisper's normal, timestamped decoding: see above
            p.single_segment = False
            p.no_context = True              # each phrase on its own (a prompt made it worse)
            p.suppress_blank = True
            p.suppress_nst = True            # suppress non-speech tokens in noisy rooms
            p.print_progress = p.print_realtime = p.print_timestamps = p.print_special = False
            if self.vad_path.is_file():
                p.vad = True
                p.vad_model_path = str(self.vad_path)
            self._pw, self._ctx, self._params = pw, ctx, p
            log.info(f"[voice] Speech model loaded in {time.monotonic() - t0:.1f}s "
                     f"({self.threads} threads)")

    def close(self) -> None:
        with self._lock:
            if self._ctx is not None:
                self._pw.whisper_free(self._ctx)
                self._ctx = self._params = None
                # A Python log function left registered in whisper.cpp is
                # released by a C++ destructor after Python has shut down,
                # which crashed EyeSync on exit (exit code 0xC0000005).
                self._pw.whisper_log_set(None)

    def __call__(self, audio: np.ndarray) -> str:
        self.load()
        pw, ctx, p = self._pw, self._ctx, self._params
        a = np.ascontiguousarray(audio, np.float32)
        secs = a.size / RATE
        p.audio_ctx = min(1500, max(self.AUDIO_CTX_MIN, math.ceil(secs / 30 * 1500) + 64))
        p.max_tokens = int(12 + 7 * secs)
        with self._lock:
            pw.whisper_full(ctx, p, a, a.size)
            n = pw.whisper_full_n_segments(ctx)
            text = "".join(pw.whisper_full_get_segment_text(ctx, i).decode("utf-8", "replace")
                           for i in range(n))
        return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- #
# Text -> what to type
# --------------------------------------------------------------------------- #

# Voice commands, said as a phrase of their own (a pause before and after).
# Whisper writes them in many ways ("New line.", "newline") - compared after
# normalise(); one-word ones it sometimes mishears, hence the extra spellings.
STOP_PHRASES = {"stop listening", "stop dictation", "stop dictating", "stop voice typing",
                "stop typing", "stop voice"}
UNDO_PHRASES = {"delete that", "scratch that", "undo that", "erase that", "remove that",
                "delete last"}
KEY_PHRASES = {
    "new line": "newline", "newline": "newline", "next line": "newline", "new life": "newline",
    "new paragraph": "paragraph", "next paragraph": "paragraph",
    "press enter": "enter", "hit enter": "enter", "enter": "enter", "press return": "enter",
    "tab": "tab", "press tab": "tab", "tab key": "tab",
    "backspace": "backspace", "back space": "backspace", "press backspace": "backspace",
    "space": "space", "spacebar": "space", "space bar": "space", "press space": "space",
}
PUNCTUATION_PHRASES = {
    "comma": ",", "coma": ",", "kama": ",", "period": ".", "full stop": ".", "dot": ".",
    "question mark": "?", "exclamation mark": "!", "exclamation point": "!",
    "colon": ":", "semicolon": ";", "semi colon": ";",
}
COMMANDS = sorted(STOP_PHRASES | UNDO_PHRASES | set(KEY_PHRASES) | set(PUNCTUATION_PHRASES))
# Said at the start or end of a longer phrase, these still count:
_EDGE_KEYS = re.compile(r"^(?:new (line|paragraph))\b[\s,.:;!?]*|\s*\bnew (line|paragraph)[.!?]*$",
                        re.IGNORECASE)
_EDGE_STOP = re.compile(r"[\s,.:;!?]*\bstop (?:listening|dictation|dictating|voice typing)[.!?]*$",
                        re.IGNORECASE)
# Whisper's sound labels - "[BLANK_AUDIO]", "(buzzing)", "*laughs*", "♪" - aren't speech.
_ANNOTATION = re.compile(r"\[[^\]]*\]|\*[^*]{0,40}\*|\([^()]{0,40}\)|[♪♫]+")
# ... nor the subtitle credits it learned from videos.
_CREDITS = ("thanks for watching", "thank you for watching", "please subscribe",
            "like and subscribe", "subtitles by", "amara org", "transcribed by")
# A phrase that carries on a sentence after a pause (no full stop before it)
# starting with one of these: Whisper capitalises it anyway (each phrase is
# new to it), so it's lowered.
_CONTINUING = set("""a an and as at be because but by can could did do does for from had has
have he her his how if in into is it its just my nor not of on or our please she should so
than that the their them then there these they this those to too was we were what when where
which while who why will with would you your also about after before since until""".split())
# Words that join on to the sentence before: a full stop Whisper put before
# one of these (it heard a pause) is taken back. Not just any lower-case
# start - Whisper also writes one when the first word was swallowed.
_JOINING = set("""and but or nor so because with without to of for from in on at by than then
that which who whom whose where while as if into about after before since until also plus
including like""".split())
_SENTENCE_END = ".!?"


def normalise(text: str) -> str:
    """Lower case, no punctuation or quotes: for comparing with the commands."""
    text = re.sub(r"[^\w\s']", " ", text.lower().replace("-", " "))
    return re.sub(r"\s+", " ", text).strip(" '")


def clean(text: str) -> str:
    """Whisper's text without sound labels and video credits."""
    text = _ANNOTATION.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    if not re.search(r"\w", text):
        return ""
    norm = normalise(text)
    if any(c in norm for c in _CREDITS):
        return ""
    return text


class Composer:
    """
    Turns recognised phrases into what to type, as operations:
        ("text", "..."), ("key", "enter" | "newline" | "tab" | "backspace"),
        ("back", n) = n backspaces, ("stop", "") = end voice typing.
    It remembers what it typed (``typed``, newest last) so phrases join up
    and "delete that" removes exactly the last one. reset() when the text
    cursor may have moved (another window, a click): then nothing is known
    about the text around it.
    """

    def __init__(self):
        self.typed: list[str] = []

    def reset(self) -> None:
        self.typed.clear()

    @property
    def last(self) -> str | None:
        return self.typed[-1] if self.typed else None

    def _remember(self, s: str) -> None:
        self.typed.append(s)
        del self.typed[:-50]

    def plan(self, raw: str) -> list[tuple[str, object]]:
        text = clean(raw)
        if not text:
            return []
        norm = normalise(text)
        if norm in STOP_PHRASES:
            return [("stop", "")]
        if norm in UNDO_PHRASES:
            return [("back", len(self.typed.pop()))] if self.typed else []
        if norm in KEY_PHRASES:
            return self._key(KEY_PHRASES[norm])
        if norm in PUNCTUATION_PHRASES:
            return self._punctuation(PUNCTUATION_PHRASES[norm])
        ops: list[tuple[str, object]] = []
        stop = _EDGE_STOP.search(text)
        if stop and stop.start() > 0:
            text = text[:stop.start()]
        keys_after = []
        while (m := _EDGE_KEYS.search(text)) is not None and text:
            word = (m.group(1) or m.group(2)).lower()
            key = "newline" if word == "line" else "paragraph"
            if m.start() == 0:                       # "New line. Dear John"
                ops += self._key(key)
            else:                                    # "... best wishes, new line"
                keys_after.insert(0, key)
            text = (text[:m.start()] + text[m.end():]).strip()
        if text and re.search(r"\w", text):
            ops += self._text(text)
        for key in keys_after:
            ops += self._key(key)
        if stop:
            ops.append(("stop", ""))
        return ops

    def _key(self, key: str) -> list[tuple[str, object]]:
        if key == "space":
            self._remember(" ")
            return [("text", " ")]
        if key == "backspace":
            if self.typed:
                self.typed[-1] = self.typed[-1][:-1]
                if not self.typed[-1]:
                    self.typed.pop()
            return [("key", "backspace")]
        if key == "paragraph":
            self._remember("\n\n")
            return [("key", "newline"), ("key", "newline")]
        self._remember("\t" if key == "tab" else "\n")
        return [("key", key)]

    def _punctuation(self, mark: str) -> list[tuple[str, object]]:
        ops: list[tuple[str, object]] = []
        last = self.last
        if last and last[-1] in ".,!?:;":
            # Whisper already ended the phrase with its own mark: replace it.
            ops.append(("back", 1))
            self.typed[-1] = last[:-1]
            if not self.typed[-1]:
                self.typed.pop()
        self._remember(mark)
        ops.append(("text", mark))
        return ops

    def _text(self, text: str) -> list[tuple[str, object]]:
        ops: list[tuple[str, object]] = []
        last = self.last
        if last:
            first = re.match(r"[A-Za-z']+", text)
            word = first.group(0) if first else ""
            dots = len(last) - len(last.rstrip(".…"))
            trailing_off = dots > 1 or last.endswith("…")
            if dots and word[:1].islower() and (trailing_off or word in _JOINING):
                # Whisper ended the last phrase with a full stop (or "..." -
                # trailing off), but this one carries the sentence on: take it back.
                ops.append(("back", dots))
                last = last[:-dots]
                if last:
                    self.typed[-1] = last
                else:
                    self.typed.pop()
                    last = self.last
            if last and last[-1] in _SENTENCE_END and word[:1].islower():
                text = text[0].upper() + text[1:]           # a new sentence
            elif (last and last[-1] not in _SENTENCE_END + ":\n" and word[:1].isupper()
                    and word.lower() in _CONTINUING):
                text = text[0].lower() + text[1:]
            if last and not last[-1].isspace() and text[0] not in ",.!?;:)":
                text = " " + text
        self._remember(text)
        ops.append(("text", text))
        return ops


# --------------------------------------------------------------------------- #
# Typing: keyboard input into the window in front
# --------------------------------------------------------------------------- #

VK_BACK, VK_TAB, VK_RETURN, VK_SHIFT = 0x08, 0x09, 0x0D, 0x10
KEYEVENTF_KEYUP, KEYEVENTF_UNICODE = 0x0002, 0x0004
KEYS = {"enter": [VK_RETURN], "tab": [VK_TAB], "backspace": [VK_BACK],
        # Shift+Enter: a new line in a chat box too, where Enter would send the message
        "newline": [VK_SHIFT, VK_RETURN]}


def key_events(ops) -> list[tuple[int, int, int]]:
    """The (virtual key, character, flags) keyboard events for the operations."""
    ev: list[tuple[int, int, int]] = []

    def press(*vks):
        ev.extend((vk, 0, 0) for vk in vks)
        ev.extend((vk, 0, KEYEVENTF_KEYUP) for vk in reversed(vks))

    for op, arg in ops:
        if op == "text":
            for ch in str(arg):
                if ch == "\n":
                    press(*KEYS["newline"])
                elif ch == "\t":
                    press(VK_TAB)
                else:
                    data = ch.encode("utf-16-le")      # characters beyond U+FFFF: two units
                    for i in range(0, len(data), 2):
                        unit = int.from_bytes(data[i:i + 2], "little")
                        ev.append((0, unit, KEYEVENTF_UNICODE))
                        ev.append((0, unit, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
        elif op == "key":
            press(*KEYS[str(arg)])
        elif op == "back":
            for _ in range(int(arg)):
                press(VK_BACK)
    return ev


class Typist:
    """Sends keyboard input to the window that has the keyboard focus (SendInput)."""

    BATCH = 100              # events per SendInput call (a pause between keeps apps in step)

    def __init__(self):
        from ctypes import wintypes

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                        ("dwExtraInfo", ctypes.c_size_t)]

        class INPUT(ctypes.Structure):
            class _U(ctypes.Union):
                _fields_ = [("ki", KEYBDINPUT), ("_pad", ctypes.c_byte * 32)]
            _anonymous_ = ("u",)
            _fields_ = [("type", wintypes.DWORD), ("u", _U)]

        self._INPUT, self._KEYBDINPUT = INPUT, KEYBDINPUT
        self._user32 = ctypes.windll.user32
        self._user32.SendInput.argtypes = (wintypes.UINT, ctypes.c_void_p, ctypes.c_int)

    def run(self, ops) -> int:
        """Type the operations; returns how many keyboard events Windows accepted."""
        events = key_events(ops)
        sent = 0
        for start in range(0, len(events), self.BATCH):
            chunk = events[start:start + self.BATCH]
            inputs = (self._INPUT * len(chunk))()
            for i, (vk, scan, flags) in enumerate(chunk):
                inputs[i].type = 1                            # INPUT_KEYBOARD
                inputs[i].ki = self._KEYBDINPUT(vk, scan, flags, 0, 0)
            sent += self._user32.SendInput(len(chunk), ctypes.byref(inputs),
                                           ctypes.sizeof(self._INPUT))
            if start + self.BATCH < len(events):
                time.sleep(0.01)
        return sent


class ForegroundTracker:
    """
    Which window typing goes to: the one in front, unless that is EyeSync's
    own window. poll() a few times a second remembers the previous window.
    """

    SKIP_CLASSES = {"Shell_TrayWnd", "Shell_SecondaryTrayWnd", "Progman", "WorkerW",
                    "Windows.UI.Core.CoreWindow"}     # taskbar, desktop, Start menu

    def __init__(self):
        from ctypes import wintypes
        self.user32 = ctypes.windll.user32
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
        self.user32.IsWindow.argtypes = self.user32.IsWindowVisible.argtypes = (wintypes.HWND,)
        self.user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
        self.user32.GetClassNameW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
        self.pid = os.getpid()
        self.last_other: int | None = None

    def foreground(self) -> int | None:
        return self.user32.GetForegroundWindow() or None

    def window_pid(self, hwnd: int) -> int:
        from ctypes import wintypes
        pid = wintypes.DWORD()
        self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return pid.value

    def ours(self, hwnd: int | None) -> bool:
        return bool(hwnd) and self.window_pid(hwnd) == self.pid

    def _class(self, hwnd: int) -> str:
        buf = ctypes.create_unicode_buffer(128)
        self.user32.GetClassNameW(hwnd, buf, 128)
        return buf.value

    def poll(self) -> None:
        fg = self.foreground()
        if fg and not self.ours(fg) and self._class(fg) not in self.SKIP_CLASSES:
            self.last_other = fg

    def peek(self) -> int | None:
        """The window typing would go to now (without bringing it to the front)."""
        fg = self.foreground()
        if fg and not self.ours(fg):
            return fg
        prev = self.last_other
        return prev if prev and self.user32.IsWindow(prev) and self.user32.IsWindowVisible(prev) else None

    def target(self) -> int | None:
        """The window to type into (brought to the front if needed), or None."""
        fg = self.foreground()
        if fg and not self.ours(fg):
            return fg
        prev = self.peek()
        if prev:
            self.user32.SetForegroundWindow(prev)       # allowed: EyeSync is in front
            for _ in range(20):
                time.sleep(0.02)
                if self.foreground() == prev:
                    time.sleep(0.05)                    # let it restore its text cursor
                    return prev
        return None

    def elevated(self, hwnd: int) -> bool:
        """Does the window belong to a program running as administrator (and EyeSync doesn't)?"""
        try:
            return _is_elevated(self.window_pid(hwnd)) and not _is_elevated(self.pid)
        except OSError:
            return False


def _is_elevated(pid: int) -> bool:
    from ctypes import wintypes
    kernel32, advapi32 = ctypes.windll.kernel32, ctypes.windll.advapi32
    kernel32.OpenProcess.restype = wintypes.HANDLE
    process = kernel32.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
    if not process:
        return False
    token = wintypes.HANDLE()
    try:
        if not advapi32.OpenProcessToken(process, 0x0008, ctypes.byref(token)):   # TOKEN_QUERY
            return False
        elevation, size = wintypes.DWORD(), wintypes.DWORD()
        ok = advapi32.GetTokenInformation(token, 20, ctypes.byref(elevation), 4,   # TokenElevation
                                          ctypes.byref(size))
        return bool(ok and elevation.value)
    finally:
        if token:
            kernel32.CloseHandle(token)
        kernel32.CloseHandle(process)


# --------------------------------------------------------------------------- #
# The microphone
# --------------------------------------------------------------------------- #

class Microphone:
    """
    The default microphone, mono at 16 kHz (Windows converts), opened only
    while voice typing is on. If 16 kHz is refused it records at the
    microphone's own rate and converts (resample).
    """

    def __init__(self, rate: int = RATE):
        self.rate = rate
        self.stream = None
        self.native = rate
        self.blocks: queue.SimpleQueue = queue.SimpleQueue()

    def open(self) -> None:
        import sounddevice as sd
        try:
            info = sd.query_devices(kind="input")
        except Exception as exc:                              # noqa: BLE001 - PortAudio's errors
            raise OSError("no microphone found") from exc
        self.name = str(info.get("name", "microphone"))

        def callback(data, frames, when, status):              # PortAudio's thread
            self.blocks.put(data[:, 0].copy())

        for rate in (self.rate, int(info.get("default_samplerate") or 0)):
            if rate <= 0:
                continue
            try:
                self.stream = sd.InputStream(samplerate=rate, channels=1, dtype="float32",
                                             blocksize=int(rate * 0.05), callback=callback)
                self.stream.start()
                self.native = rate
                return
            except Exception as exc:                          # noqa: BLE001
                self.stream = None
                error = exc
        raise OSError(str(error))

    def read(self, timeout: float) -> np.ndarray | None:
        try:
            block = self.blocks.get(timeout=timeout)
        except queue.Empty:
            return None
        return block if self.native == self.rate else resample(block, self.native, self.rate)

    def close(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:                                 # noqa: BLE001 - closing anyway
                pass
            self.stream = None


def resample(x: np.ndarray, rate_in: int, rate_out: int) -> np.ndarray:
    """Simple sample-rate conversion (averaging, then linear interpolation) - fine for speech."""
    if rate_in == rate_out or x.size == 0:
        return x
    k = max(1, int(round(rate_in / rate_out)))
    if k > 1:                                                 # low-pass before thinning out
        x = np.convolve(x, np.ones(k, np.float32) / k, mode="same")
    n = int(round(x.size * rate_out / rate_in))
    return np.interp(np.arange(n) * (rate_in / rate_out), np.arange(x.size), x).astype(np.float32)


# --------------------------------------------------------------------------- #
# Voice typing: one listening session at a time
# --------------------------------------------------------------------------- #

class VoiceTyping:
    """
    Listens for requests (state.voice_requests: (on, source), on=None means
    toggle - see request()) and runs one voice typing session at a time in
    its own thread. Publishes what it's doing in the shared state for the
    window, the bar on the screen and the website: voice ("" = off, see
    STATES), voice_text (the last phrase typed - shown on this PC only),
    voice_hint, voice_level (microphone meter), voice_left (countdown).
    """

    HINT_S = 5.0          # a problem stays on the screen this long after voice typing stopped

    def __init__(self, state, recognizer: Recognizer | None = None, mic_factory=None,
                 typist=None, tracker=None, sounds: bool = True, check=available,
                 vad_factory=None):
        self.state = state
        self.recognizer = recognizer or Recognizer()
        self.mic_factory = mic_factory or Microphone
        self.vad_factory = vad_factory or SileroVad
        self.typist, self.tracker = typist, tracker
        self.sounds, self.check = sounds, check
        self.composer = Composer()
        self.session: threading.Thread | None = None
        self.stop_session = threading.Event()
        self.thread = threading.Thread(target=self._control, name="voice-control", daemon=True)
        self.clear_at = 0.0
        self.context = (None, -1, -math.inf)     # (window, clicks, time) of the last typing

    # -- control ------------------------------------------------------------
    def start(self) -> "VoiceTyping":
        if sys.platform == "win32":
            self.typist = self.typist or Typist()
            self.tracker = self.tracker or ForegroundTracker()
        self.thread.start()
        return self

    def active(self) -> bool:
        return self.session is not None and self.session.is_alive()

    def close(self, timeout: float = 5.0) -> None:
        self.stop_session.set()
        if self.session is not None:
            self.session.join(timeout)
        self.thread.join(timeout)

    def _control(self) -> None:
        while not self.state.stop.is_set():
            try:
                on, source = self.state.voice_requests.get(timeout=0.25)
            except queue.Empty:
                on = source = None
            if self.tracker is not None:
                try:
                    self.tracker.poll()
                except OSError:
                    pass
            if source is not None:
                want = (not self.active()) if on is None else bool(on)
                if want and not self.active():
                    self._start(source)
                elif not want and self.active():
                    log.info(f"[voice] Stop requested ({source})")
                    self.stop_session.set()
            if self.clear_at and time.monotonic() > self.clear_at and not self.active():
                self.clear_at = 0.0
                self.state.update(voice="", voice_hint="")
        self.stop_session.set()

    def _start(self, source: str) -> None:
        problem = self.check() or ("" if self.typist is not None else "Voice typing needs Windows")
        if problem:
            log.warning(f"[voice] {problem}")
            self._show_problem(problem)
            return
        self.stop_session.clear()
        self.clear_at = 0.0
        self.state.update(voice="starting", voice_hint="", voice_text="", voice_level=0.0,
                          voice_left=0.0)
        self.session = threading.Thread(target=self._run_session, args=(source,),
                                        name="voice", daemon=True)
        self.session.start()

    def _show_problem(self, text: str) -> None:
        self.state.update(voice="error", voice_hint=text, voice_level=0.0, voice_left=0.0)
        self.clear_at = time.monotonic() + self.HINT_S

    # -- one session --------------------------------------------------------
    def _run_session(self, source: str) -> None:
        reason, problem = "stopped", ""
        mic = vad = None
        try:
            mic = self.mic_factory()
            try:
                mic.open()
            except Exception as exc:                     # noqa: BLE001 - shown to the user
                log.warning(f"[voice] Microphone: {exc}")
                problem = f"Can't use the microphone ({str(exc)[:40]})"
                return
            log.info(f"[voice] Voice typing ON (microphone on: {getattr(mic, 'name', '?')}) - {source}")
            # The sound plays while the microphone wakes up (a laptop's gave
            # silence for its first ~0.5 s); _listen skips the sound itself.
            if self.sounds:
                desktop.play_voice_sound(True)
            self.recognizer.load()
            vad = self.vad_factory()
            reason = self._listen(mic, vad)
        except Exception as exc:                         # noqa: BLE001 - never take EyeSync down
            log.exception("[voice] Voice typing failed")
            problem = f"Voice typing stopped: {str(exc)[:44]}"
        finally:
            if mic is not None:
                mic.close()
            if vad is not None and hasattr(vad, "close"):
                vad.close()
            self.recognizer.close()
            if self.sounds:
                desktop.play_voice_sound(False)
            if problem:
                log.warning(f"[voice] {problem}")
                self._show_problem(problem)
            else:
                self.state.update(voice="", voice_level=0.0, voice_left=0.0)
            log.info(f"[voice] Voice typing OFF (microphone off) - {problem or reason}")

    def _listen(self, mic, vad) -> str:
        seg = SpeechSegmenter(vad)
        last_audio = last_publish = time.monotonic()
        zeros, skip = 0, int(RATE * START_SOUND_S)  # the start sound, as recorded
        while True:
            if self.stop_session.is_set():
                rest = seg.flush()                  # what was said just before the stop
                if rest is not None:
                    self._phrase(rest)
                return "stopped"
            if self.state.stop.is_set():
                return "EyeSync is closing"
            block = mic.read(0.1)
            now = time.monotonic()
            if block is None:
                if now - last_audio > 3.0:
                    raise OSError("the microphone stopped sending sound")
                continue
            last_audio = now
            if skip > 0:
                block, skip = block[skip:], max(0, skip - block.size)
            zeros = zeros + block.size if not np.any(block) else 0
            if zeros > MIC_SILENT_S * RATE and seg.t < MIC_SILENT_S + 1:
                self.state.update(voice_hint="No sound - allow the microphone: Settings > Privacy "
                                  "> Microphone")
            for phrase in seg.feed(block):
                self._phrase(phrase)
            if now - last_publish > 0.08:
                last_publish = now
                self.state.update(voice="hearing" if seg.speaking else "listening",
                                  voice_level=round(seg.level, 2),
                                  voice_left=0.0)

    def _phrase(self, audio: np.ndarray) -> bool:
        """Recognise and type one phrase. True if it said to stop."""
        self.state.update(voice="working", voice_level=0.0)
        t0 = time.monotonic()
        text = self.recognizer(audio)
        took = time.monotonic() - t0
        if not clean(text):
            log.info(f"[voice] {audio.size / RATE:.1f}s of sound - no speech in it")
            return False
        window = self.tracker.peek() if self.tracker is not None else None
        snap = self.state.read()
        clicks = snap.click_count + snap.right_click_count
        last_window, last_clicks, last_time = self.context
        if window != last_window or clicks != last_clicks or t0 - last_time > CONTEXT_S:
            self.composer.reset()                # the text cursor may be somewhere else now
        ops = self.composer.plan(text)
        # A spoken stop phrase does not stop voice typing; use the eye gesture
        # or the visible bar's Stop button.
        stop = False
        ops = [op for op in ops if op[0] != "stop"]
        typed = "".join(str(a) for o, a in ops if o == "text")
        if ops and self.tracker is not None:
            window = self.tracker.target()       # EyeSync's window in front: the one before it
            if window is None:
                self.composer.reset()
                self.state.update(voice_hint="Click where the text should go, then say it again",
                                  voice_text=typed.strip())
                log.info("[voice] Nowhere to type: no other window was in front")
                return stop
            if self.tracker.elevated(window):
                self.composer.reset()
                self.state.update(voice_hint="Windows doesn't let EyeSync type into "
                                  "administrator apps", voice_text=typed.strip())
                log.info("[voice] Not typed: the window in front runs as administrator")
                return stop
        if ops:
            sent = self.typist.run(ops)
            if sent < len(key_events(ops)):
                log.warning(f"[voice] Windows accepted only {sent} of {len(key_events(ops))} "
                            "key presses")
            self.context = (window, clicks, time.monotonic())
            what = (f"{len(typed)} characters" if typed else
                    ", ".join(str(a) if o == "key" else f"{a} backspaces" for o, a in ops))
            log.info(f"[voice] Typed {what} ({audio.size / RATE:.1f}s of speech, "
                     f"recognised in {took:.2f}s)")
            self.state.update(voice_text=typed.strip() or describe(ops), voice_hint="")
        if stop:
            log.info('[voice] "Stop listening"')
        return stop


def describe(ops) -> str:
    """What keys a command pressed, for the bar on the screen."""
    names = {"newline": "new line", "enter": "Enter", "tab": "Tab", "backspace": "Backspace"}
    parts = [names.get(str(a), str(a)) if o == "key" else f"deleted {a} characters"
             for o, a in ops if o in ("key", "back")]
    return "(" + ", ".join(parts) + ")" if parts else ""


def request(state, on: bool | None = None, source: str = "button") -> None:
    """Ask for voice typing on (True), off (False) or the other way round (None)."""
    state.voice_requests.put((on, source))
