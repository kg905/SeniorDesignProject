"""Writes annotated images and JSON using the same exported model and preprocessing."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cribwatch.showcase import main

if __name__ == "__main__":
    main(["image", *sys.argv[1:]])
