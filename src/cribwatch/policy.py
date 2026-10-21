"""Conservative research alarm policy. All dwell times are engineering defaults."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PolicyConfig:
    side_seconds: float = 2.0
    prone_seconds: float = 1.0
    unknown_seconds: float = 2.0
    recovery_seconds: float = 3.0
    max_gap_seconds: float = 1.5

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("All policy timings must be finite and positive")


class AlarmPolicy:
    def __init__(self, config=None):
        self.config = config or PolicyConfig()
        self.last_time = None
        self.candidate = None
        self.since = 0.0
        self.unsafe_since = None
        self.latched = False

    def update(self, posture, now, *, acknowledge=False):
        if posture not in {"back", "side", "prone", "unknown"}:
            raise ValueError("Invalid posture")
        if not math.isfinite(now) or (self.last_time is not None and now < self.last_time):
            raise ValueError("Use finite, nondecreasing monotonic timestamps")
        gap = self.last_time is not None and now - self.last_time > self.config.max_gap_seconds
        # An observation gap cannot count as continuous reassuring or concerning evidence.
        if gap:
            posture = "unknown"
        self.last_time = now
        if posture != self.candidate or gap:
            self.candidate, self.since = posture, now
        elapsed = now - self.since
        if posture not in {"side", "prone"} or gap:
            self.unsafe_since = None
        if posture in {"side", "prone"}:
            if self.unsafe_since is None:
                self.unsafe_since = now
            delay = self.config.side_seconds if posture == "side" else self.config.prone_seconds
            if now - self.unsafe_since >= delay:
                self.latched = True
        fault = gap or (posture == "unknown" and elapsed >= self.config.unknown_seconds)
        can_ack = posture == "back" and elapsed >= self.config.recovery_seconds
        if acknowledge and can_ack:
            self.latched = False
        state = ("check_baby" if self.latched else "monitor_fault" if fault else
                 "back_observed" if can_ack else "observing")
        return {"state": state, "posture": posture, "baby_alarm": self.latched,
                "monitor_fault": fault, "can_acknowledge": can_ack,
                "candidate_seconds": round(elapsed, 3)}
