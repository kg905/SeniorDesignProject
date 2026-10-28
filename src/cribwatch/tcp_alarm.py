"""Versioned, newline-delimited JSON on TCP 5000; shares legacy replay protection."""

import hashlib
import hmac
import json
import socket
import socketserver

from .transport import validate_key

MAX_MESSAGE = 8192


def canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def make_tcp_server(host, port, state):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            self.connection.settimeout(1)
            try:
                line = self.rfile.readline(MAX_MESSAGE + 1)
                if len(line) > MAX_MESSAGE or not line.endswith(b"\n"):
                    return
                data = json.loads(line)
                if not isinstance(data, dict):
                    return
                if data.get("op") == "challenge":
                    reply = {"nonce": state.challenge()}
                elif data.get("op") == "status" and isinstance(data.get("payload"), dict) and data["payload"].get("schema_version") == 1:
                    reply = {"accepted": state.accept(canonical(data["payload"]), data.get("signature"))}
                else:
                    reply = {"accepted": False}
                self.wfile.write(canonical(reply) + b"\n")
            except (ValueError, OSError, TypeError):
                return

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
    return Server((host, port), Handler)


class TCPAlarmClient:
    def __init__(self, host, key, *, port=5000, timeout=.5):
        self.host, self.port, self.timeout = host, port, timeout
        self.key = validate_key(key)

    def exchange(self, payload):
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as connection:
            connection.settimeout(self.timeout)
            connection.sendall(canonical(payload) + b"\n")
            with connection.makefile("rb") as stream:
                line = stream.readline(MAX_MESSAGE + 1)
            if len(line) > MAX_MESSAGE or not line.endswith(b"\n"):
                raise ConnectionError("Invalid alarm reply")
            return json.loads(line)

    def send(self, status):
        nonce = self.exchange({"op": "challenge"})["nonce"]
        payload = dict(status, nonce=nonce)
        signature = hmac.new(self.key, canonical(payload), hashlib.sha256).hexdigest()
        if not self.exchange({"op": "status", "payload": payload, "signature": signature}).get("accepted"):
            raise ConnectionError("Alarm receiver rejected status")
