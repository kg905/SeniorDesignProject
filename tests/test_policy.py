import pytest

from cribwatch.policy import AlarmPolicy


def test_prone_latches_and_does_not_clear_on_one_back_frame():
    p = AlarmPolicy()
    assert not p.update("prone", 0)["baby_alarm"]
    assert p.update("prone", 1)["baby_alarm"]
    assert p.update("back", 1.2, acknowledge=True)["baby_alarm"]
    p.update("back", 2.2)
    p.update("back", 3.2)
    assert p.update("back", 4.2)["can_acknowledge"]
    assert not p.update("back", 4.3, acknowledge=True)["baby_alarm"]


def test_alternating_unsafe_labels_still_trigger():
    p = AlarmPolicy()
    p.update("side", 0)
    p.update("prone", .5)
    p.update("side", 1)
    assert p.update("prone", 1.5)["baby_alarm"]


def test_missing_data_is_not_back_or_clearance():
    p = AlarmPolicy()
    p.update("unknown", 0)
    p.update("unknown", 1)
    state = p.update("unknown", 2)
    assert state["monitor_fault"] and not state["can_acknowledge"]
    assert state["state"] == "monitor_fault"


def test_elapsed_gap_cannot_create_risk_evidence():
    p = AlarmPolicy()
    p.update("prone", 0)
    state = p.update("prone", 10)
    assert state["monitor_fault"] and not state["baby_alarm"]
    assert state["posture"] == "unknown"


def test_time_reversal_rejected():
    p = AlarmPolicy()
    p.update("back", 10)
    with pytest.raises(ValueError):
        p.update("back", 9)
