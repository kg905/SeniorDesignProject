"""One posture network with explicit 12-metric and compact visual branches."""
import numpy as np
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .features import FEATURES, feature_vector

ARCHITECTURE = "mobilenet_v3_small_metric_fusion_v1"
IMAGE_SIZE = 160


def body_evidence(row):
    if row.get("quality_reason") in {"multiple_people", "multiple_faces", "torso_too_small", "invalid_landmarks"}:
        return False
    return (str(row.get("pose_ok")) == "1" or row.get("quality_reason") == "low_pose_quality"
            or row.get("extraction_route") in {"yolo_crop", "yolo_keypoints"})


def raw_metrics(row):
    values = feature_vector(row)
    # Poor torso coordinates are not reliable geometry. Retain native confidence
    # and face observations, but mark unreliable lengths/angles/depth missing.
    if str(row.get("pose_ok", "0")) != "1":
        values[[0, 1, 2, 3, 4, 11]] = np.nan
    return values


def fit_normalizer(rows):
    values = np.stack([raw_metrics(r) for r in rows])
    centers, scales = [], []
    for column in values.T:
        finite = column[np.isfinite(column)]
        centers.append(float(np.median(finite)) if len(finite) else 0.)
        scales.append(max(float(np.std(finite)), .01) if len(finite) else 1.)
    return {"features": list(FEATURES), "center": centers, "scale": scales,
            "fitted_on": "training_only", "input_size": 25}


def encode_metrics(row, normalizer):
    values = raw_metrics(row)
    missing = ~np.isfinite(values)
    scaled = (np.where(missing, normalizer["center"], values) - normalizer["center"]) / normalizer["scale"]
    return np.concatenate([np.clip(scaled, -10, 10), missing.astype(float),
                           [float(str(row.get("pose_ok", "0")) == "1")]]).astype(np.float32)


def image_tensor(image):
    from PIL import Image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    array = np.array(image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR), copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.)
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    return (tensor - mean) / std


class MetricFusionNet(nn.Module):
    def __init__(self, *, pretrained=False):
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        self.visual = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.metrics = nn.Sequential(nn.Linear(25, 64), nn.ReLU(), nn.Dropout(.1), nn.Linear(64, 64), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(576 + 64, 128), nn.ReLU(), nn.Dropout(.2), nn.Linear(128, 3))
        for parameter in self.visual.parameters():
            parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        # Small batches and shifted camera data should not overwrite pretrained BN statistics.
        self.visual.eval()
        return self

    def unfreeze_tail(self):
        for block in list(self.visual.children())[-3:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    def forward(self, images, metrics, *, ablation=None):
        visual = self.pool(self.visual(images)).flatten(1)
        metric = self.metrics(metrics)
        if ablation == "no_image":
            visual = torch.zeros_like(visual)
        elif ablation == "no_metrics":
            metric = torch.zeros_like(metric)
        elif self.training:
            # Encourage the metric path to remain useful instead of letting the
            # image branch dominate every training example.
            visual = visual * (torch.rand((len(visual), 1), device=visual.device) >= .2)
        return self.classifier(torch.cat([visual, metric], dim=1))
