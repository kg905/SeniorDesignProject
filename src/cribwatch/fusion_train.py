"""Train-only visual fine tuning and metric normalization; validation-only selection."""
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ColorJitter

from .data import read_csv, write_csv
from .features import FEATURES, LABELS, SCHEMA_VERSION, VISION_PREPROCESSING
from .fusion_model import ARCHITECTURE, IMAGE_SIZE, MetricFusionNet, body_evidence, encode_metrics, fit_normalizer, image_tensor
from .learning import _adapter_metadata, _select_threshold, _vision_metadata, feature_schema_for, group_split, summarize


class Frames(Dataset):
    def __init__(self, rows, normalizer, *, augment=False):
        self.rows = rows
        self.metrics = torch.from_numpy(np.stack([encode_metrics(r, normalizer) for r in rows]))
        self.augment = ColorJitter(.2, .2, .1) if augment else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image_path"]) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
            if self.augment:
                image = self.augment(image)
            tensor = image_tensor(image)
        return tensor, self.metrics[index], LABELS.index(row["posture"])


def logits_for(net, loader, device, *, ablation=None):
    net.eval()
    logits = []
    with torch.inference_mode():
        for images, metrics, _ in loader:
            logits.append(net(images.to(device), metrics.to(device), ablation=ablation).cpu())
    return torch.cat(logits).numpy()


def probabilities(logits, temperature=1.):
    return torch.softmax(torch.as_tensor(logits) / temperature, dim=1).numpy()


def predictions_for(rows, probability, threshold):
    pred = np.array(LABELS, dtype=object)[probability.argmax(axis=1)]
    allowed = np.array([body_evidence(row) for row in rows])
    pred[~allowed | (probability.max(axis=1) < threshold)] = "unknown"
    return pred


def _select_temperature(logits, rows):
    mask = np.array([body_evidence(r) for r in rows])
    if not mask.any():
        return 1.
    labels = np.array([LABELS.index(r["posture"]) for r in rows])[mask]
    return min((.5, .75, 1., 1.5, 2., 3., 4.), key=lambda t: float(
        nn.functional.cross_entropy(torch.tensor(logits[mask]) / t, torch.tensor(labels))))


def train_fusion(features_csv, output, *, epochs=10, warmup=2, batch_size=32,
                 seed=42, device="auto", exploratory=True, pretrained=True):
    torch.set_num_threads(2)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if epochs < 1 or warmup < 0 or batch_size < 1:
        raise ValueError("Choose positive epochs/batch size and nonnegative warmup")
    output = Path(output)
    options = dict(epochs=epochs, warmup=warmup, batch_size=batch_size, seed=seed, pretrained=pretrained)
    fingerprint = hashlib.sha256(Path(features_csv).read_bytes()).hexdigest()
    if (output / "test.json").exists():
        report = json.loads((output / "test.json").read_text())
        if report.get("features_sha256") != fingerprint:
            raise ValueError("Different features; preserve this trained run and use a new directory")
        if hashlib.sha256((output / "posture_fusion.pt").read_bytes()).hexdigest() != report.get("model_sha256"):
            raise ValueError("Saved fusion weights no longer match the evaluated model")
        existing = json.loads((output / "fusion.json").read_text())
        if existing.get("training_options") != options:
            raise ValueError("Training settings changed; use a new model directory rather than reusing the old score")
        if hashlib.sha256((output / "fusion.json").read_bytes()).hexdigest() != report.get("config_sha256"):
            raise ValueError("Saved configuration no longer matches its evaluation")
        print("Reusing completed, checksum-verified model:", output, flush=True)
        return report
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve the partial experiment; choose a fresh output directory")
    rows = [r for r in read_csv(features_csv) if r.get("posture") in LABELS and r.get("label_conflict", "0") != "1"
            and (r.get("training_eligible") == "1" if exploratory else r.get("reviewed") == "1")]
    if exploratory and any(r.get("label_source") not in {"human_review", "dataset_annotation"} for r in rows):
        raise ValueError("Require audited human or publisher posture labels, not model suggestions")
    if not rows:
        raise ValueError("No reviewed/audited posture rows")
    group_field = "source_group" if exploratory else "baby_id"
    if any(not r.get(group_field, "").strip() for r in rows):
        raise ValueError("Missing actual evaluation grouping")
    if not exploratory and any(not r.get("session_id") for r in rows):
        raise ValueError("Infant-independent evaluation requires real infant/session IDs")
    if not exploratory and any(r.get("subject_kind") == "doll" for r in rows):
        raise ValueError("Doll recordings cannot establish infant-independent accuracy")
    unique = {}
    for row in rows:
        if not row.get("sha256") or hashlib.sha256(Path(row["image_path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Source image missing or changed since labeling")
        if row["sha256"] in unique and unique[row["sha256"]]["posture"] != row["posture"]:
            raise ValueError("Conflicting duplicate labels")
        unique.setdefault(row["sha256"], row)
    rows = list(unique.values())
    splits = group_split(rows, seed, group_field=group_field)
    parts = [[rows[i] for i in indices] for indices in splits]
    fit_rows = [r for r in parts[0] if body_evidence(r)]
    normalizer = fit_normalizer(parts[0])
    vision_assets, adapter = _vision_metadata(rows), _adapter_metadata(rows)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    net = MetricFusionNet(pretrained=pretrained).to(device)
    optimizer = torch.optim.AdamW([
        {"params": net.visual.parameters(), "lr": 1e-4},
        {"params": list(net.metrics.parameters()) + list(net.classifier.parameters()), "lr": 8e-4}], weight_decay=1e-4)
    counts = np.array([sum(r["posture"] == label for r in fit_rows) for label in LABELS])
    if np.any(counts == 0):
        raise ValueError("Training split needs every posture")
    weights = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    train_loader = DataLoader(Frames(fit_rows, normalizer, augment=True), batch_size=batch_size,
                              shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed))
    validation_loader = DataLoader(Frames(parts[1], normalizer), batch_size=batch_size, num_workers=0)
    output.mkdir(parents=True, exist_ok=False)
    history, best = [], None
    print(f"Training one image + 12-metric network on {len(fit_rows)} body-confirmed training images ({device}); validation {len(parts[1])}, test {len(parts[2])}", flush=True)
    val_y = np.array([LABELS.index(r["posture"]) for r in parts[1]])
    for epoch in range(epochs):
        if epoch >= warmup:
            net.unfreeze_tail()
        net.train()
        losses = []
        for step, (images, metrics, labels) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net(images.to(device), metrics.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if step == 0 or (step+1) % 25 == 0:
                print(f"Epoch {epoch+1}/{epochs}, batch {step+1}/{len(train_loader)}, loss {losses[-1]:.4f}", flush=True)
        logits = logits_for(net, validation_loader, device)
        pred = predictions_for(parts[1], probabilities(logits), 0.)
        recalls = [float(np.mean(pred[val_y == i] == label)) for i, label in enumerate(LABELS)]
        val_loss = float(nn.functional.cross_entropy(torch.tensor(logits), torch.tensor(val_y)))
        key = (min(recalls), float(np.mean(recalls)), -val_loss)
        history.append({"epoch": epoch+1, "train_loss": float(np.mean(losses)), "validation_loss": val_loss,
                        "validation_minimum_recall": key[0], "validation_macro_recall": key[1]})
        print(f"Validation minimum recall {key[0]:.1%}, mean recall {key[1]:.1%}", flush=True)
        if best is None or key > best[0]:
            best = key, epoch+1, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            torch.save(best[2], output / "best-training-state.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2))
    net.load_state_dict(best[2]); net.eval()
    val_logits = logits_for(net, validation_loader, device)
    temperature = _select_temperature(val_logits, parts[1])
    val_prob = probabilities(val_logits, temperature)
    class FixedProbabilities:
        classes_ = np.array(LABELS)
        def predict_proba(self, values):
            return val_prob
    threshold, constraint_met = _select_threshold(FixedProbabilities(),
        [dict(r, pose_ok="1" if body_evidence(r) else "0") for r in parts[1]], .8)
    config = {"architecture": ARCHITECTURE, "features": list(FEATURES), "schema_version": feature_schema_for(rows),
        "preprocessing": VISION_PREPROCESSING, "image_size": IMAGE_SIZE, "normalizer": normalizer,
        "vision_assets": vision_assets, "vision_adapter": adapter, "temperature": temperature,
        "score_threshold": threshold, "features_sha256": fingerprint, "exploratory": exploratory,
        "best_epoch": best[1], "training_options": options, "initialized_from": "official ImageNet MobileNetV3 Small" if pretrained else "random smoke-test initialization"}
    torch.save(best[2], output / "posture_fusion.pt")
    config["model_sha256"] = hashlib.sha256((output / "posture_fusion.pt").read_bytes()).hexdigest()
    (output / "fusion.json").write_text(json.dumps(config, indent=2))
    config_digest = hashlib.sha256((output / "fusion.json").read_bytes()).hexdigest()
    test_loader = DataLoader(Frames(parts[2], normalizer), batch_size=batch_size, num_workers=0)
    test_logits = logits_for(net, test_loader, device)
    test_prob = probabilities(test_logits, temperature)
    test_pred = predictions_for(parts[2], test_prob, threshold)
    validation = summarize(parts[1], predictions_for(parts[1], val_prob, threshold), group_field=group_field)
    report = summarize(parts[2], test_pred, group_field=group_field)
    for current in (validation, report):
        current.update(model_sha256=config["model_sha256"], config_sha256=config_digest, features_sha256=fingerprint, architecture=ARCHITECTURE,
                       temperature=temperature, score_threshold=threshold, validation_coverage_constraint_met=constraint_met,
                       exploratory=exploratory, best_epoch=best[1], metric_input_count=12)
    report["body_evidence_coverage_ceiling"] = float(np.mean([body_evidence(r) for r in parts[2]]))
    report["unthresholded_diagnostic"] = summarize(parts[2], predictions_for(parts[2], test_prob, 0.), group_field=group_field)
    report["ablations"] = {}
    for mode in ("no_metrics", "no_image"):
        p = probabilities(logits_for(net, test_loader, device, ablation=mode), temperature)
        report["ablations"][mode] = summarize(parts[2], predictions_for(parts[2], p, threshold), group_field=group_field)
    report["ablation_note"] = "Branches zeroed in the SAME trained network; sensitivity check, not separately trained baseline models. Test ablations do not select weights."
    for name, value in (("test.json", report), ("validation.json", validation)):
        (output / name).write_text(json.dumps(value, indent=2))
    split_rows = [dict(r, split=name) for name, part in zip(("train", "validation", "test"), parts) for r in part]
    write_csv(output / "split_manifest.csv", split_rows, list(split_rows[0]))
    write_csv(output / "test_predictions.csv", [dict(r, predicted=p, **{f"score_{label}": float(v) for label, v in zip(LABELS, scores)})
        for r, p, scores in zip(parts[2], test_pred, test_prob)], list(parts[2][0]) + ["predicted", *[f"score_{label}" for label in LABELS]])
    (output / "best-training-state.pt").unlink(missing_ok=True)
    print(f"Fusion test score: {100 * report['accuracy_all_images_unknown_counts_as_error']:.1f}/100", flush=True)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--infant-independent", action="store_true")
    args = parser.parse_args()
    train_fusion(args.features, args.output, epochs=args.epochs, warmup=args.warmup,
                 batch_size=args.batch_size, device=args.device, exploratory=not args.infant_independent)
