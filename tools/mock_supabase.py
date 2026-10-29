"""
mock_supabase.py - a stand-in for the site's Supabase project, for testing accounts locally.
=========================================================================================

    python tools/mock_supabase.py [--port 54321] [--token-ttl 3600] [--confirm-email]

Answers the handful of Supabase requests the website makes, with the same shapes and error codes,
and keeps everything in memory (nothing is written to disk, everything is gone when it stops):

    POST /auth/v1/signup                         create an account (and a session)
    POST /auth/v1/token?grant_type=password      sign in
    POST /auth/v1/token?grant_type=refresh_token new tokens (the old refresh token stops working)
    GET  /auth/v1/user                           who is signed in
    POST /auth/v1/logout                         end this session
    GET/POST/PATCH /rest/v1/profiles             the signed-in user's own row only, like the
                                                 table's Row Level Security policies
    POST /rest/v1/rpc/delete_my_account          delete the signed-in user and their row

To point a copy of the site at it, change SUPABASE_URL in that copy to http://127.0.0.1:54321 and
add the same address to its Content-Security-Policy connect-src. --token-ttl 20 makes access
tokens expire quickly, to test refreshing; --confirm-email behaves like "Confirm email" switched on.
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

LOCK = threading.Lock()
USERS: dict[str, dict] = {}        # email -> {id, email, password, created_at, user_metadata}
ACCESS: dict[str, tuple] = {}      # access token -> (user id, expires at)
REFRESH: dict[str, str] = {}       # refresh token -> user id
PROFILES: dict[str, dict] = {}     # user id -> row
TTL = 3600
CONFIRM = False
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def public_user(u: dict) -> dict:
    return {"id": u["id"], "aud": "authenticated", "role": "authenticated", "email": u["email"],
            "created_at": u["created_at"], "user_metadata": u["user_metadata"]}


def session_for(u: dict) -> dict:
    access, refresh = secrets.token_urlsafe(24), secrets.token_urlsafe(16)
    ACCESS[access] = (u["id"], time.time() + TTL)
    REFRESH[refresh] = u["id"]
    return {"access_token": access, "token_type": "bearer", "expires_in": TTL,
            "expires_at": int(time.time() + TTL), "refresh_token": refresh, "user": public_user(u)}


def user_by_id(uid: str) -> dict | None:
    return next((u for u in USERS.values() if u["id"] == uid), None)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[mock-supabase] {self.command} {self.path.split('?')[0]} -> {args[1] if len(args) > 1 else ''}", flush=True)

    # --- plumbing ---
    def cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "apikey, authorization, content-type, prefer, x-client-info")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")

    def reply(self, status: int, body=None):
        data = b"" if body is None else json.dumps(body).encode()
        self.send_response(status)
        self.cors()
        if body is not None:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def auth_error(self, status: int, code: str, msg: str):
        self.reply(status, {"code": status, "error_code": code, "msg": msg})

    def body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            data = None
        return data if isinstance(data, dict) else {}

    def caller(self) -> str | None:
        """The user id of a valid access token, or None."""
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return None
        hit = ACCESS.get(auth[7:])
        return hit[0] if hit and hit[1] > time.time() and user_by_id(hit[0]) else None

    def do_OPTIONS(self):
        self.reply(204)

    # --- requests ---
    def do_GET(self):
        url = urlsplit(self.path)
        with LOCK:
            uid = self.caller()
            if url.path == "/auth/v1/user":
                if not uid:
                    return self.auth_error(403, "bad_jwt", "invalid JWT: unable to parse or verify signature")
                return self.reply(200, public_user(user_by_id(uid)))
            if url.path == "/rest/v1/profiles":
                want = parse_qs(url.query).get("id", [""])[0].removeprefix("eq.")
                rows = [PROFILES[uid]] if uid and want == uid and uid in PROFILES else []
                return self.reply(200, rows)
        self.reply(404, {"message": "not found"})

    def do_POST(self):
        url = urlsplit(self.path)
        q = parse_qs(url.query)
        data = self.body()
        with LOCK:
            uid = self.caller()
            if url.path == "/auth/v1/signup":
                email = str(data.get("email", "")).strip().lower()
                password = str(data.get("password", ""))
                if not EMAIL_RE.match(email):
                    return self.auth_error(400, "validation_failed", "Unable to validate email address: invalid format")
                if len(password) < 6:
                    return self.auth_error(422, "weak_password", "Password should be at least 6 characters.")
                if email in USERS:
                    return self.auth_error(422, "user_already_exists", "User already registered")
                meta = data.get("data") if isinstance(data.get("data"), dict) else {}
                u = USERS[email] = {"id": str(uuid.uuid4()), "email": email, "password": password,
                                    "created_at": now_iso(), "user_metadata": meta, "confirmed": not CONFIRM}
                return self.reply(200, public_user(u) if CONFIRM else session_for(u))
            if url.path == "/auth/v1/token" and q.get("grant_type") == ["password"]:
                u = USERS.get(str(data.get("email", "")).strip().lower())
                if not u or u["password"] != data.get("password"):
                    return self.auth_error(400, "invalid_credentials", "Invalid login credentials")
                if not u["confirmed"]:
                    return self.auth_error(400, "email_not_confirmed", "Email not confirmed")
                return self.reply(200, session_for(u))
            if url.path == "/auth/v1/token" and q.get("grant_type") == ["refresh_token"]:
                owner = REFRESH.pop(str(data.get("refresh_token", "")), None)
                u = user_by_id(owner) if owner else None
                if not u:
                    return self.auth_error(400, "refresh_token_not_found", "Invalid Refresh Token: Refresh Token Not Found")
                return self.reply(200, session_for(u))
            if url.path == "/auth/v1/logout":
                if not uid:
                    return self.auth_error(403, "bad_jwt", "invalid JWT")
                ACCESS.pop(self.headers["Authorization"][7:], None)
                return self.reply(204)
            if url.path == "/rest/v1/profiles":
                if not uid or data.get("id") != uid:
                    return self.reply(403, {"code": "42501", "message": 'new row violates row-level security policy for table "profiles"'})
                row = PROFILES.get(uid, {"id": uid, "prefs": {}, "agreed": None, "created_at": now_iso()})
                row.update({k: data[k] for k in ("prefs", "agreed") if k in data}, updated_at=now_iso())
                PROFILES[uid] = row
                return self.reply(201)
            if url.path == "/rest/v1/rpc/delete_my_account":
                if not uid:
                    return self.reply(401, {"code": "42501", "message": "permission denied for function delete_my_account"})
                u = user_by_id(uid)
                USERS.pop(u["email"], None)
                PROFILES.pop(uid, None)
                for t in [t for t, (owner, _) in ACCESS.items() if owner == uid]:
                    ACCESS.pop(t)
                for t in [t for t, owner in REFRESH.items() if owner == uid]:
                    REFRESH.pop(t)
                return self.reply(204)
        self.reply(404, {"message": "not found"})

    def do_PATCH(self):
        url = urlsplit(self.path)
        data = self.body()
        with LOCK:
            uid = self.caller()
            want = parse_qs(url.query).get("id", [""])[0].removeprefix("eq.")
            if url.path == "/rest/v1/profiles" and uid and want == uid and uid in PROFILES:
                PROFILES[uid].update({k: data[k] for k in ("prefs", "agreed") if k in data}, updated_at=now_iso())
            return self.reply(204)


def main() -> None:
    global TTL, CONFIRM
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--port", type=int, default=54321)
    p.add_argument("--token-ttl", type=int, default=3600, help="seconds an access token lasts")
    p.add_argument("--confirm-email", action="store_true", help='behave like "Confirm email" switched on')
    a = p.parse_args()
    TTL, CONFIRM = a.token_ttl, a.confirm_email
    print(f"[mock-supabase] http://127.0.0.1:{a.port} (tokens last {TTL} s)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
