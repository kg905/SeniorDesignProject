# Uploaded fusion deployment: 49.2% on 512 test images

[Download SIDS_Pi5_Fusion_Geometry.zip](SIDS_Pi5_Fusion_Geometry.zip) using **Download raw** on GitHub. This is the exact user-uploaded archive received November 7, 2026. Its original code, weights, assets and bundled instructions are preserved unchanged. It is a separate release from the older 267-image and 882-image experiments.

Use the [updated Pi 5 / Pi 4 quickstart](../../../docs/PI_FUSION_QUICKSTART.md) for this fusion package. The bundled README also describes the older metric-only path; the quickstart identifies the correct fusion files and dependencies.

## Recorded evaluation

These results come from the bundled `models/test.json`, not a new training run or hardware test. F1 values are calculated from that report's confusion matrix; see [evaluation_metrics.json](evaluation_metrics.json).

| Evaluation | Images | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| All test images; unknown predictions count as errors | 512 | 49.2% | 0.580 | 0.588 |
| Posture assigned only | 348 | 72.4% | 0.716 | 0.731 |

Prediction coverage is **68.0%**: 348 assignments, including 252 correct and 96 incorrect. The remaining 164 predictions are unknown. Full-test recall is back **57.4%**, side **46.7%**, prone **42.8%**. The saved acceptance threshold is 0.50. The report does not establish the 95% target or accuracy on unseen infants.

The earlier 73.3% accepted-only result was from a different 267-image run. Do not apply it to this uploaded model or compare cohorts as a controlled before/after experiment.

## Integrity and contents

- Archive: 18,490,018 bytes; SHA256 `5eb641aa59751c38fdbf031204e82d16cb2d00c5d1fbeff02ff24d3f03d964b3`.
- Posture weights: SHA256 `98dbc3ec04e1db58611bc84fc99ede7191d738a28952fa1065654a65d3421301`.
- Tensor-only posture network, schema-3 feature contract, MobileNet image branch, 12-metric branch, generic YOLOv8 Pose and MediaPipe assets, Pi 5 runtime and Pi 4 TCP receiver.
- ZIP CRC, all 43 entries in bundled `SHA256SUMS.json`, and weights/config/report binding verified during intake. No uploaded script was executed to verify these hashes.
- [Original test report](test.json), [original model configuration](fusion.json), and [release receipt](release.json) are available separately for inspection.

Local smoke checks also passed: the bundled Pi 4 receiver starts its help path without third-party packages, the Pi 5 live/offline command options load, and the fusion weights plus detector assets load successfully. A blank frame returns `unknown` / `no_pose`. The new notebook F1 cell reproduces the metrics above from reconstructed confusion-matrix counts and handles all-unknown/empty inputs. These checks do not measure new posture accuracy or test physical Pi hardware.

`BabyTraining-V4.zip` is the separate training dataset, not part of this deployment. Its attachment exceeds the environment's 32 MiB transfer limit and has not been copied to GitHub. The user is providing it through Drive. Its ZIP-to-model provenance has not been independently checked here.

Keep this archive together; copying only a `.pt` file into an older deployment can change or break feature compatibility. Uploading this package does not modify its measured performance. Actual Pi camera latency, networking, GPIO and audible alarm tests remain necessary.
