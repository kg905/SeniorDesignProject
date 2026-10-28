"""LAN heartbeat with one-use receiver challenges and HMAC-SHA256 authentication.

No shared clock required. This authenticates messages, not their contents' privacy.
Use an isolated WPA2/WPA3 network; use TLS/a VPN if the LAN is untrusted.
"""

from collections import OrderedDict
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import secrets
import threading
import time
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener


def validate_key(key):
    if not isinstance(key, bytes) or len(key) < 32:
        raise ValueError("ALARM_SHARED_KEY must contain at least 32 bytes; generate with secrets.token_hex(32)")
    return key


def get_key():
    return validate_key(os.environ.get("ALARM_SHARED_KEY", "").encode())


class ReceiverState:
    def __init__(self, key, *, timeout=5.0, clock=time.monotonic):
        self.key = validate_key(key)
        if timeout <= 0:
            raise ValueError("Timeout must be positive")
        self.timeout, self.clock = timeout, clock
        self.lock = threading.Lock()
        self.challenges = OrderedDict()
        self.last_message = None
        self.status = {"baby_alarm": False, "monitor_fault": True, "state": "waiting_for_camera"}

    def challenge(self):
        with self.lock:
            now = self.clock()
            # Bound unauthenticated challenge storage.
            self.challenges = OrderedDict((k, v) for k, v in self.challenges.items() if now - v <= 5)
            while len(self.challenges) >= 128:
                self.challenges.popitem(last=False)
            nonce = secrets.token_hex(32)
            self.challenges[nonce] = now
            return nonce

    def accept(self, body, signature):
        expected = hmac.new(self.key, body, hashlib.sha256).hexdigest()
        if not isinstance(signature, str) or not hmac.compare_digest(expected, signature):
            return False
        try:
            data = json.loads(body)
            if not isinstance(data, dict):
                return False
            if type(data.get("baby_alarm")) is not bool or type(data.get("monitor_fault")) is not bool:
                return False
            if data.get("state") not in {"observing", "back_observed", "check_baby", "monitor_fault"}:
                return False
            nonce = data.get("nonce")
            if not isinstance(nonce, str):
                return False
        except (ValueError, UnicodeDecodeError):
            return False
        with self.lock:
            issued = self.challenges.pop(nonce, None)
            now = self.clock()
            if issued is None or now - issued > 5:
                return False
            self.last_message = now
            self.status = {k: data[k] for k in ("baby_alarm", "monitor_fault", "state")}
            return True

    def snapshot(self):
        with self.lock:
            result = self.status.copy()
            stale = self.last_message is None or self.clock() - self.last_message >= self.timeout
        result["monitor_fault"] |= stale
        result["link_stale"] = stale
        result["audible"] = result["baby_alarm"] or result["monitor_fault"]
        return result


def make_server(host, port, receiver):
    class Handler(BaseHTTPRequestHandler):
        timeout = 2

        def setup(self):
            super().setup()
            self.connection.settimeout(2)

        def log_message(self, *_):
            pass

        def reply(self, code, data):
            body = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/challenge":
                self.reply(404, {"error": "not_found"})
                return
            self.reply(200, {"nonce": receiver.challenge()})

        def do_POST(self):
            if self.path != "/status":
                self.reply(404, {"error": "not_found"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                size = 0
            if not 0 < size <= 4096:
                self.reply(413, {"error": "size"})
                return
            try:
                body = self.rfile.read(size)
                ok = receiver.accept(body, self.headers.get("X-Signature", ""))
                self.reply(200 if ok else 403, {"accepted": ok})
            except TimeoutError:
                self.close_connection = True

    return ThreadingHTTPServer((host, port), Handler)


class AlarmClient:
    def __init__(self, url, key, *, timeout=0.5):
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Use http(s)://<alarm-host>:<port>, without embedded credentials")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("Alarm URL must be a server root")
        self.url, self.key, self.timeout = url.rstrip("/"), validate_key(key), timeout
        # LAN messages must not go to an ambient Internet egress proxy.
        self.opener = build_opener(ProxyHandler({}))

    def send(self, status):
        with self.opener.open(self.url + "/challenge", timeout=self.timeout) as r:
            challenge = json.load(r)
        payload = {k: status[k] for k in ("state", "baby_alarm", "monitor_fault")}
        payload["nonce"] = challenge["nonce"]
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        request = Request(self.url + "/status", data=body, headers={
            "Content-Type": "application/json", "X-Signature": hmac.new(self.key, body, hashlib.sha256).hexdigest(),
        })
        with self.opener.open(request, timeout=self.timeout) as r:
            if not json.load(r).get("accepted"):
                raise ConnectionError("Alarm rejected heartbeat")
