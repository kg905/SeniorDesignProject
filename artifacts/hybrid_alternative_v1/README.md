# Trained YOLO + MediaPipe alternative

This is an actual exploratory training/evaluation run on **4,421 eligible publisher-labeled images from the user's original Roboflow uploads**. It does not use the manually sorted 322-image ZIP from Colab; that ZIP has not been uploaded to this workspace. Older artifacts under posture_baseline_v2 and image_research_final are preserved.

Both alternatives use exactly the same source/duplicate-group split and **882 test images**. All 12 named metric inputs are retained. Each final classifier is one histogram gradient boosting classifier (selected recipe: gradient_boosting_leaves15, score threshold 0.5). Seven classifier configurations were compared using validation only. The hybrid is the validation-selected recommendation; both exports and their honest test reports are retained.

| Measurement | Matching MediaPipe baseline | YOLO + MediaPipe alternative |
| --- | ---: | ---: |
| Usable poses across all eligible images | 55.8% | 71.1% |
| Test accuracy, unknown = error | 23.5% | 29.1% |
| Test prediction coverage | 39.3% | 49.8% |
| Test back recall | 35.1% | 38.5% |
| Test side recall | 12.7% | 20.1% |
| Test prone recall | 19.0% | 26.0% |
| Accuracy among accepted predictions only | 59.7% | 58.5% |

The hybrid recovered **146 usable cropped MediaPipe poses and 532 usable YOLO-keypoint poses** across the eligible images. That improves coverage, but accepted-prediction accuracy did not improve in this run. This remains far below 95%; prone recall is still inadequate for a reliable baby alert. No independent-infant, SIDS/airway, or actual Pi performance claim is established. Source groups are not verified infant/session identities, and these original public data have already been inspected.

The alternative starts with MediaPipe, retries a YOLO body crop, then can use YOLO 2D shoulders/hips plus actual MediaPipe dense face landmarks. The four visibility entries use native YOLO confidence on fallback frames. Head-lift depth is missing when MediaPipe cannot estimate world coordinates. These differences require the separate classifier; do not plug hybrid values into an old model. The generic official YOLOv8 nano pose weights are not infant-fine-tuned and are not weights from the suggested external repository. See [the complete feature contract](../../docs/HYBRID_ALTERNATIVE.md).

## Files

- hybrid-model/posture.joblib and test.json: new alternative and checksum-bound report.
- baseline-model/: matching freshly trained baseline for a fair comparison; does not replace older models.
- SIDS_Pi5_Hybrid_Alternative.zip: deployment bundle with ONE classifier, exact MediaPipe/YOLO assets, runtime, TCP protocol, extra requirements and attribution. Install the extra requirements as described in its HYBRID.md.
- hybrid_comparison.json: both diagnostics, validation/test reports and recorded adapter configuration.
- images.csv, source-features.csv, features-baseline.csv, features-hybrid.csv: audited labels/provenance and actual feature values. Image paths reflect this workspace; use a new manifest/import when running elsewhere.
- annotated-samples/: actual offline scans of the first test image of each true posture, including errors/unknowns if produced. These examples illustrate output, not accuracy proof. Source images and attribution are retained under data/datasets/posture in the repository.
- environment.json: actual package versions used here. Timing numbers describe this x86_64 cloud machine, not Pi 5 throughput.

Use the [separate hybrid Colab](https://colab.research.google.com/github/kg905/SeniorDesignProject/blob/main/notebooks/hybrid_mediapipe_yolo_colab.ipynb) to train on your reviewed BabyTraining.zip. For the current session, see [reuse instructions](../../docs/HYBRID_ALTERNATIVE.md). Next meaningful experiments are infant-tuned keypoint detection with suitable annotations and feature/input changes if the 12 metrics cannot distinguish back from belly. More training images alone cannot guarantee 95%.
