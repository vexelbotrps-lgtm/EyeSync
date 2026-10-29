"""
site_access.py - who may use EyeSync, and waiting for them while it is closed.
=============================================================================

* The EyeSync website is recognised by the browser's Origin header.
* Other websites and apps need EyeSync's private API key (ApiKey), passed as
  ws://localhost:8080/eye-tracker?key=<key>.
* wait_for_visit() is the standby used by EyeSync.pyw after you quit
  EyeSync: it listens on the website's ports and returns as soon as the
  website is opened again, so EyeSync starts by itself.

Deliberately light (standard library only): the standby runs all the time.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import selectors
import socket
import struct
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import app_paths

log = logging.getLogger("eyesync")

VERSION = "1.5.14"        # shown in the log and to the website; names the release files
PROJECT_DIR = app_paths.DATA_DIR    # the user's files (see app_paths)
# The EyeSync website (GitHub Pages since 2026-10-14; it replaced the Wix site).
DEFAULT_SITE_ORIGINS = ["https://aniruddh9140.github.io"]
DEFAULT_PORTS = (8080, 8765)
API_KEY_PATH = PROJECT_DIR / "eyesync_api_key.txt"


def is_website_origin(origin: str, allowed: list[str]) -> bool:
    """
    Does a browser Origin header belong to the EyeSync website? Compared by
    host name only, so http/https and ports don't matter.
    """
    host = (urlsplit(origin).hostname or "").lower()
    if not host:
        return False
    for entry in allowed:
        allowed_host = (urlsplit(entry if "//" in entry else "//" + entry).hostname or "").lower()
        if host == allowed_host:
            return True
    return False


class ApiKey:
    """
    EyeSync's private API key. Only the EyeSync website (recognised by its
    Origin) and websites or apps that connect with
    ws://localhost:8080/eye-tracker?key=<key> may use EyeSync; anything else
    is refused, so other web pages can't read where you look. Made on first
    run and kept in its own file (new default settings never change it).
    "New key" in the window replaces it, which disconnects everything that
    used the old one. The key is never written to the log.
    """

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.version = 0                   # bumped by regenerate()
        self._key = self._load() or self._create()

    def _load(self) -> str | None:
        try:
            key = self.path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return key if len(key) >= 20 and key.isascii() and key.isprintable() else None

    def _create(self) -> str:
        key = "esk_" + secrets.token_urlsafe(24)
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(key + "\n", encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            log.warning(f"[api] Could not save the API key to {self.path} ({exc}); "
                        "it works until EyeSync closes")
        return key

    def get(self) -> str:
        with self._lock:
            return self._key

    def regenerate(self) -> str:
        with self._lock:
            self._key = self._create()
            self.version += 1
        log.info("[api] New API key made - apps using the old key are disconnected")
        return self.get()

    def matches(self, candidate: str) -> bool:
        return hmac.compare_digest(candidate.encode("utf-8"), self.get().encode("utf-8"))


# --------------------------------------------------------------------------- #
# Standby: start EyeSync when the website is opened
# --------------------------------------------------------------------------- #

def _parse_request(conn: socket.socket) -> tuple[dict[str, str], dict[str, list[str]]]:
    """(headers with lower-case names, URL query) of a WebSocket handshake; empty if unreadable."""
    conn.settimeout(1.0)
    data = b""
    try:
        while b"\r\n\r\n" not in data and len(data) < 8192:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
    except OSError:
        pass
    lines = data.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    target = parts[1] if len(parts) >= 3 else "/"
    headers = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        if name.strip():
            headers[name.strip().lower()] = value.strip()
    return headers, parse_qs(urlsplit(target).query)


def _read_request(conn: socket.socket) -> tuple[str, str]:
    """(origin, key) from a WebSocket handshake; empty strings if unreadable."""
    headers, query = _parse_request(conn)
    return headers.get("origin", ""), query.get("key", [""])[0]


WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"     # RFC 6455


def _ws_frame(opcode: int, payload: bytes) -> bytes:
    """One unmasked server frame (FIN set)."""
    n = len(payload)
    head = bytes([0x80 | opcode]) + (bytes([n]) if n < 126 else bytes([126]) + struct.pack("!H", n))
    return head + payload


def _ws_answer(conn: socket.socket, headers: dict[str, str], message: dict, code: int,
               reason: str) -> bool:
    """
    Accept the WebSocket, send one JSON message and close with ``code`` -
    so the website learns that EyeSync is installed (a refused connection
    can't tell it that). False if this wasn't a WebSocket handshake.
    """
    key = headers.get("sec-websocket-key", "")
    if not key:
        return False
    accept = base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest()).decode()
    conn.sendall(b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                 b"Connection: Upgrade\r\nSec-WebSocket-Accept: " + accept.encode() + b"\r\n\r\n"
                 + _ws_frame(0x1, json.dumps(message).encode())
                 + _ws_frame(0x8, struct.pack("!H", code) + reason.encode()[:120]))
    try:                                     # let the browser answer the close, so its
        conn.settimeout(0.5)                 # onclose event reports our code cleanly
        conn.recv(256)
    except OSError:
        pass
    return True


def _listen(port: int) -> list[socket.socket]:
    """Listening sockets on both loopbacks (like the real server); [] if the port is busy."""
    socks = []
    for family, host in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        try:
            s = socket.socket(family, socket.SOCK_STREAM)
            s.bind((host, port))
            s.listen(16)
            s.setblocking(False)
            socks.append(s)
        except OSError:
            s.close()
    return socks


def wait_for_visit(ports=DEFAULT_PORTS, origins: list[str] | None = None,
                   key_path: Path = API_KEY_PATH, quiet_s: float = 12.0,
                   confirm_s: float = 9.0, wake=None, after_quit: bool = True,
                   click_gap_s: float = 0.75) -> str:
    """
    Block until the EyeSync website (or an app with the API key) is opened,
    then release the ports and return "website". ``wake()`` is polled twice
    a second; when it returns True (the EyeSync shortcut was opened) this
    returns "shortcut" at once.

    After you quit EyeSync (mouth wide open, Quit, F12 - ``after_quit``),
    only a click on the website's **Connect** button starts it again. The
    website keeps retrying by itself with growing gaps (~1, 2, 4, 8, then
    every ~10 s, +-20%), while Connect tries at once and again ~0.5 s later
    (0.4-0.6 s): two tries within ``click_gap_s`` are a click, never an
    automatic retry.

    At login (``after_quit`` False) the dashboard may already be open and
    trying: two tries within ``confirm_s`` start EyeSync (after ``quiet_s``
    without tries, so a background tab that browsers slow to one try a minute
    doesn't).

    A website that knows the newer protocol (WEBSITE.md) asks explicitly
    instead: ``?start=1`` (Connect pressed) starts EyeSync at once and gets
    close code 4002; ``?probe=1`` never starts it and gets 4003. Both first
    receive {"type": "eyesync", "state": "starting" | "standby", "version",
    "closed_by_user"}, so the website knows EyeSync is installed.
    """
    origins = origins or DEFAULT_SITE_ORIGINS
    sel = selectors.DefaultSelector()
    bound: dict[int, list[socket.socket]] = {}
    last_try = -1e9
    armed_try: float | None = None         # first try after a quiet spell
    next_bind = 0.0
    warned = False
    try:
        while True:
            now = time.monotonic()
            if now >= next_bind:            # (re)claim ports another program held
                for port in ports:
                    if port not in bound and (socks := _listen(port)):
                        bound[port] = socks
                        for s in socks:
                            sel.register(s, selectors.EVENT_READ)
                if len(bound) < len(ports) and not warned:
                    log.warning(f"[standby] Ports {sorted(set(ports) - set(bound))} are busy; "
                                "retrying every 5 s")
                    warned = True
                next_bind = now + 5.0
            if wake is not None and wake():
                return "shortcut"
            for key, _ in sel.select(timeout=0.5 if bound else 0.0):
                try:
                    conn, _ = key.fileobj.accept()
                except OSError:
                    continue
                with conn:
                    headers, query = _parse_request(conn)
                    origin, given = headers.get("origin", ""), query.get("key", [""])[0]
                    allowed = is_website_origin(origin, origins)
                    if not allowed and given:
                        try:
                            saved = key_path.read_text(encoding="utf-8").strip()
                            allowed = bool(saved) and hmac.compare_digest(given.encode(),
                                                                          saved.encode())
                        except OSError:
                            pass
                    # The website's explicit requests (WEBSITE.md): ?start=1 = the
                    # user pressed Connect; ?probe=1 = "is EyeSync installed?".
                    # Plain tries keep the old answer, so older pages behave as before.
                    ask = ("start" if query.get("start") == ["1"] else
                           "probe" if query.get("probe") == ["1"] else "") if allowed else ""
                    answered = False
                    try:
                        if ask:
                            answered = _ws_answer(
                                conn, headers,
                                {"type": "eyesync", "version": VERSION, "closed_by_user": after_quit,
                                 "state": "starting" if ask == "start" else "standby"},
                                4002 if ask == "start" else 4003,
                                "EyeSync is starting" if ask == "start" else "EyeSync is installed "
                                "and waiting - connect with ?start=1 to start it")
                        if not answered:
                            conn.sendall(b"HTTP/1.1 503 Service Unavailable\r\n"
                                         b"Content-Length: 0\r\nConnection: close\r\n\r\n")
                    except OSError:
                        pass
                if not allowed or ask == "probe":
                    continue
                if ask == "start":
                    log.info("[standby] The website asked to start EyeSync (Connect)")
                    return "website"
                now = time.monotonic()
                if after_quit:                            # only a Connect click restarts
                    if now - last_try <= click_gap_s:
                        return "website"
                    last_try = now
                    continue
                if now - last_try >= quiet_s:
                    armed_try = now                       # opened after being closed...
                elif armed_try is not None and now - armed_try <= confirm_s:
                    return "website"                      # ...and really open: start
                else:
                    armed_try = None
                last_try = now
            if not bound:
                time.sleep(0.5)
    finally:
        for socks in bound.values():
            for s in socks:
                sel.unregister(s)
                s.close()
        sel.close()
