"""Run from handoff root: --model models --models models --picamera --display --showcase."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cribwatch.showcase import main

if __name__ == "__main__":
    main(["live", *sys.argv[1:]])
