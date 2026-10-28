"""Replacement receiver: TCP 5000 NDJSON challenge/HMAC protocol, not legacy bare JSON."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cribwatch.alarm_runtime import main

if __name__ == "__main__":
    main(sys.argv[1:])
