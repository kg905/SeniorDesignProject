import hashlib
import hmac
import json
import threading

import pytest

from cribwatch.transport import AlarmClient, ReceiverState, make_server

KEY = b"test-only-key-do-not-use-in-deployment-1234567890"


def signed(state, **overrides):
    payload = {"nonce": state.challenge(), "state": "check_baby", "baby_alarm": True, "monitor_fault": False}
    payload.update(overrides)
    body = json.dumps(payload).encode()
    return body, hmac.new(KEY, body, hashlib.sha256).hexdigest()


def test_message_authentication_replay_and_watchdog():
    now = [0.0]
    state = ReceiverState(KEY, clock=lambda: now[0])
    assert state.snapshot()["monitor_fault"]
    body, sig = signed(state)
    assert not state.accept(body, "bad")
    assert state.accept(body, sig)
    assert not state.accept(body, sig)
    assert state.snapshot()["baby_alarm"]
    assert not state.snapshot()["link_stale"]
    now[0] = 6
    assert state.snapshot()["monitor_fault"]
    assert state.snapshot()["baby_alarm"]


def test_expired_challenge_and_bad_types():
    now = [0.0]
    state = ReceiverState(KEY, clock=lambda: now[0])
    body, sig = signed(state)
    now[0] = 6
    assert not state.accept(body, sig)
    body, sig = signed(state, baby_alarm="false")
    assert not state.accept(body, sig)


def test_real_http_delivery():
    state = ReceiverState(KEY)
    server = make_server("127.0.0.1", 0, state)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        client = AlarmClient(f"http://127.0.0.1:{server.server_port}", KEY)
        client.send({"state": "check_baby", "baby_alarm": True, "monitor_fault": False})
        assert state.snapshot()["audible"]
        assert state.snapshot()["baby_alarm"]
        client.send({"state": "back_observed", "baby_alarm": False, "monitor_fault": False})
        assert not state.snapshot()["audible"]
    finally:
        server.shutdown()
        server.server_close()


def test_weak_keys_refused():
    with pytest.raises(ValueError):
        ReceiverState(b"short")
