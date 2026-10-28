# Review of SIDS_revelation_project

Reviewed source revision [58e994d356e4b44170304d58b3ec3eaf47519b8d](https://github.com/Lori-in-the-clouds/SIDS_revelation_project/tree/58e994d356e4b44170304d58b3ec3eaf47519b8d). This is a source review, not independent reproduction of its reported results. No external code or model weights have been copied into CribWatch or executed. The inspected tree has no license file; establish permission/license terms before redistribution.

## Useful ideas

The [README](https://github.com/Lori-in-the-clouds/SIDS_revelation_project/blob/58e994d356e4b44170304d58b3ec3eaf47519b8d/README.md) describes overhead video, adaptive contrast preprocessing (CLAHE), YOLOv8 face/body feature models fine-tuned on infant datasets, engineered features, a supervised-contrastive MLP embedding, and an XGBoost classifier. It reports 95% binary accuracy across 832 samples (423 baby_safe, 409 baby_unsafe). The README calls the feature set 86-dimensional; several saved artifacts/notebooks use other dimensions, so the exact export contract must be established for reproduction.

Infant-specific keypoint training directly addresses a weakness of our generic MediaPipe detector. That is the most relevant idea to benchmark. Contrast preprocessing also deserves a controlled validation experiment, not unconditional application. Its frame overlays are useful presentation references; our current showcase already draws a pose-derived region, risk level, scores, and alarm status.

## Compatibility with the original 12 metrics

| Required metrics | Compatibility with this project's 2D YOLO outputs |
| --- | --- |
| Shoulder/hip widths, their two angles, angle difference (5) | Can be calculated from shoulders/hips with the same torso normalization and angle conventions, provided detections are reliable. |
| Four shoulder/hip visibility values (4) | YOLO keypoint confidence can fill a similar role, but is not MediaPipe visibility/presence. Treat this as a changed feature definition requiring retraining and validation. |
| face_detected (1) | Can be implemented with their face/head detections after specifying detection and person-association rules. |
| face_landmark_count (1) | Their face-feature boxes are not MediaPipe's dense face mesh. Keep a dense face landmarker or explicitly change the metric; do not substitute a box count under the same definition. |
| head_lift_ratio (1) | Their extracted keypoints are 2D. They cannot reproduce the current MediaPipe world-z proxy. Retain an appropriate 3D estimator or explicitly revise the feature; neither gives airway clearance. |

Source: [features_extractor_keypoints](https://github.com/Lori-in-the-clouds/SIDS_revelation_project/blob/58e994d356e4b44170304d58b3ec3eaf47519b8d/libraries/embeddings_utils.py#L212-L238) retains normalized x/y after confidence filtering. A hybrid could associate infant-specific body detections with MediaPipe face/3D observations, but it is not implemented in the present improvement notebook and cannot be assumed to solve missing 3D poses.

Their final classifier predicts two classes; ours needs back, side, prone and unknown rejection. Its trained classifier also expects its own engineered/embedded features, not our 12-vector. Its weights cannot be dropped into our existing artifact. Multiple extraction networks can still feed one final posture classifier; the reference's MLP is an additional learned embedding stage.

## Why 95% is not directly comparable

The inspected [best_classifiers notebook](https://github.com/Lori-in-the-clouds/SIDS_revelation_project/blob/58e994d356e4b44170304d58b3ec3eaf47519b8d/full_pipeline/best_classifiers.ipynb) constructs EmbeddingDataset with the complete feature/label arrays, calls train_embeddings, extracts embeddings, then creates Classifier. [train_embeddings](https://github.com/Lori-in-the-clouds/SIDS_revelation_project/blob/58e994d356e4b44170304d58b3ec3eaf47519b8d/libraries/EmbeddingNet_utils.py#L44-L66) trains on that dataset using label-supervised contrastive loss. [Classifier.__init__](https://github.com/Lori-in-the-clouds/SIDS_revelation_project/blob/58e994d356e4b44170304d58b3ec3eaf47519b8d/libraries/classifier_utils.py#L29-L40) performs a random 80/20 train_test_split afterward.

That shown path leaks the eventual test labels into the learned embedding and does not split by infant/session/source group. Its result can be optimistic for genuinely unseen infants. This observation is about the public workflow inspected, not independent proof of the exact provenance of every README number. Their reported binary label accuracy also does not establish three-posture recall, end-to-end failure coverage, SIDS/airway outcomes, or alarm latency. The README says the demonstration video is AI-generated using VEO; it is not physical camera/Pi validation.

## Recommended adaptation

Keep the user's 12-metric requirement and one final three-posture classifier. First run the controlled Lite/Full and tree-classifier comparison already added in [IMPROVE_MODEL.md](IMPROVE_MODEL.md). Next benchmark an infant-tuned YOLO pose extractor on the SAME reviewed images and groups; examine keypoint correctness and prone coverage, not just whether any points are returned. Before deploying a hybrid, specify all 12 metric definitions, missing-value handling, person association, asset hashes, preprocessing, and Pi speed. Retrain the final classifier and evaluate every learned stage using train-only fitting, validation-only selection, and a fresh infant/session-held-out final test. Preserve the existing TCP alarm and live display interfaces.
