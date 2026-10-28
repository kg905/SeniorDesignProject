import argparse
import json
import math


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Must be finite and positive")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description="CribWatch research prototype; not a medical device")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("reassemble")
    p.add_argument("manifest")
    p.add_argument("--parts-dir", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("download-models")
    p.add_argument("--output", default="models")
    p = commands.add_parser("manifest")
    p.add_argument("images")
    p.add_argument("--output", default="data/manifest.csv")
    p = commands.add_parser("extract")
    p.add_argument("manifest")
    p.add_argument("--models", default="models")
    p.add_argument("--output", default="data/features.csv")
    p.add_argument("--workers", type=int, default=1)
    p = commands.add_parser("audit-coco")
    p.add_argument("--dataset", action="append", required=True, help="NAME=/path/to/extracted/export (repeatable)")
    p.add_argument("--output", required=True)
    p = commands.add_parser("import-folders")
    p.add_argument("folder", help="Folder containing back/, side/, prone/ and optional unknown/")
    p.add_argument("--output", required=True)
    p.add_argument("--reference-index", help="Original audit CSV, used only to preserve source/duplicate grouping")
    p = commands.add_parser("suggest")
    p.add_argument("features")
    p.add_argument("--output", default="data/suggested.csv")
    p = commands.add_parser("train")
    p.add_argument("features")
    p.add_argument("--output", default="artifacts/run1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exploratory", action="store_true", help="Audited source-image groups/publisher labels; cannot establish unseen-baby accuracy")
    p.add_argument("--min-validation-coverage", type=float, default=.5)
    p = commands.add_parser("embed")
    p.add_argument("features")
    p.add_argument("--models", default="models")
    p.add_argument("--output", required=True)
    p = commands.add_parser("train-image")
    p.add_argument("features")
    p.add_argument("--embeddings", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p = commands.add_parser("propose-unlabeled")
    p.add_argument("features")
    p.add_argument("--embeddings", required=True)
    p.add_argument("--classifier", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("analyze-video")
    p.add_argument("video")
    p.add_argument("--classifier", required=True)
    p.add_argument("--models", default="models")
    p.add_argument("--output", default="artifacts/video.jsonl")
    p.add_argument("--fps", type=positive_float, default=5)
    p.add_argument("--allow-exploratory", action="store_true", help="Permit an exploratory model for OFFLINE analysis only")
    p = commands.add_parser("monitor")
    p.add_argument("--classifier", required=True)
    p.add_argument("--models", default="models")
    p.add_argument("--source", default="0", help="USB camera index; use analyze-video for files")
    p.add_argument("--picamera", action="store_true")
    p.add_argument("--fps", type=positive_float, default=5)
    p.add_argument("--alarm-url", help="Pi 4 server root, e.g. http://192.168.50.1:8765")
    p.add_argument("--ack-file", help="Operator creates this file to acknowledge after stable back")
    p = commands.add_parser("receiver")
    p.add_argument("--protocol", choices=["http", "tcp"], default="http", help="New single-model showcase uses tcp --port 5000")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--timeout", type=positive_float, default=5)
    p.add_argument("--gpio", type=int, help="BCM output for transistor-driven active buzzer; omitted = console only")
    p.add_argument("--mute-button", type=int, help="BCM input to GND, silences for 10s without clearing alert")
    args = parser.parse_args(argv)
    if args.command == "reassemble":
        from .transfer import reassemble
        print(json.dumps(reassemble(args.manifest, args.parts_dir, args.output), indent=2))
    elif args.command == "download-models":
        from .assets import download_models
        download_models(args.output)
        print("Verified official model asset SHA256 hashes")
    elif args.command == "manifest":
        from .data import make_manifest
        print(f"Indexed {make_manifest(args.images, args.output)} unique images; posture labels remain unreviewed")
    elif args.command == "extract":
        from .data import extract_manifest
        print(f"Extracted {extract_manifest(args.manifest, args.output, args.models, workers=args.workers)} images")
    elif args.command == "import-folders":
        from .folder_data import import_folders
        print(json.dumps(import_folders(args.folder, args.output, reference_index=args.reference_index), indent=2))
    elif args.command == "audit-coco":
        from .coco import audit_coco
        roots = dict(value.split("=", 1) for value in args.dataset)
        print(json.dumps(audit_coco(roots, args.output), indent=2))
    elif args.command == "train":
        from .learning import train
        print(json.dumps(train(args.features, args.output, args.seed, exploratory=args.exploratory,
                               min_validation_coverage=args.min_validation_coverage), indent=2))
    elif args.command == "embed":
        from .image_model import extract_embeddings
        print(f"Embedded {extract_embeddings(args.features, args.models, args.output)} images")
    elif args.command == "train-image":
        from .image_model import train_image_baseline
        print(json.dumps(train_image_baseline(args.features, args.embeddings, args.output, args.seed), indent=2))
    elif args.command == "propose-unlabeled":
        from .image_model import propose_unlabeled
        print(json.dumps(propose_unlabeled(args.features, args.embeddings, args.classifier, args.output), indent=2))
    elif args.command == "suggest":
        from .learning import suggest_labels
        print(f"Proposed {suggest_labels(args.features, args.output)} labels; they remain unreviewed")
    elif args.command == "analyze-video":
        from .video import analyze_video
        print(f"Analyzed {analyze_video(args.video, args.classifier, args.models, args.output, sample_fps=args.fps, allow_exploratory=args.allow_exploratory)} frames")
    elif args.command == "monitor":
        from .runtime import monitor
        monitor(args)
    elif args.command == "receiver":
        from .runtime import receiver
        receiver(args)


if __name__ == "__main__":
    main()
