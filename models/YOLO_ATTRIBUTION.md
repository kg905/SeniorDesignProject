# Optional YOLO + MediaPipe experiment

`yolov8n-pose.pt` is the official Ultralytics YOLOv8 nano pose checkpoint from:
https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt

Size: 6,832,633 bytes. SHA256:
`c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212`.

Upstream: https://github.com/ultralytics/ultralytics . Ultralytics code/models are
offered under AGPL-3.0 or an applicable commercial license. The upstream AGPL text
is included in YOLO_LICENSE.txt; see https://www.ultralytics.com/license for terms.
Preserve this attribution and the applicable license when redistributing weights.

This is a generic human pose checkpoint, not an infant-fine-tuned detector and
not weights from SIDS_revelation_project. The alternative uses its body box only
to recover a crop for MediaPipe after a full-frame pose failure. MediaPipe still
calculates all 12 posture features. A trained posture classifier is produced by
the separate labeled-data training run; this checkpoint does not provide one.
