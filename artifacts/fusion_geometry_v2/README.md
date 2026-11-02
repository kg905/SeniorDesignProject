# New geometry and posture model: measured results

This is a real training/evaluation run on the original uploaded publisher-labeled datasets. It is separate from the user's [31.5% Colab model](../team_test_runs/colab-31p5-4094df16/README.md), which used a different 267-image test cohort. The reviewed images and actual doll camera clips from that run are not available here. Old models and the exact uploaded ZIP are preserved.

All models below share the same **882 test images** and source-group assignments. Unknowns count as errors. Every learned stage fits training groups only; validation chooses recipes/epochs/calibration/thresholds. Test results are reported after selection. These publicly inspected data remain exploratory and do not establish accuracy on unseen infants.

| Model | Overall accuracy | Back recall | Side recall | Prone recall |
| --- | ---: | ---: | ---: | ---: |
| Older matched hybrid | 29.1% | 38.5% | 20.1% | 26.0% |
| Directed metrics only | 34.6% | 45.2% | 23.2% | 32.2% |
| Image + original metrics | 48.4% | 59.8% | 40.5% | 40.9% |
| Image + directed metrics | 50.1% | 55.3% | 43.3% | 50.4% |

The validation-selected alternative is **Image + directed metrics**. It uses one small neural network with an explicit 12-metric branch, missing-value indicators and a visual branch. Schema 3 preserves directed transverse angles relative to the torso; old models keep their original axial image angles. YOLOv8 and MediaPipe provide the measured coordinates. These projected angles are not calibrated 3D roll or airway clearance.

In the same trained directed network, zeroing the metric branch reduced test accuracy to **43.4%**; zeroing the visual branch yielded **16.2%**. These are sensitivity checks, not independently trained replacement baselines. The separately trained metrics-only model is reported above.

**95% is not achieved.** The body's detection/evidence gate covers only **78.1% of this test cohort**, which bounds achievable all-image accuracy under that gate even for a perfect downstream classifier. The next necessary detector improvement requires verified shoulder/hip annotations and representative overhead doll recordings. A provided manual annotation/export route can fine-tune a four-keypoint YOLOv8 extractor; no such tuning result is claimed without verified annotations.

## Use the new model

- **SIDS_Pi5_Fusion_Geometry.zip**: separate deployment with the directed fused classifier, matching YOLO/MediaPipe assets, detailed geometry display, camera capture/annotation utilities and unchanged TCP 5000 alarm contract. Read its README.md, FUSION.md and HYBRID.md; extra hybrid dependencies are required.
- **directed_fusion_model/**: safe tensor-only posture weights, fusion.json feature/calibration metadata, checksum-bound evaluation and grouped predictions.
- **directed_metrics_model/**: separately trained metrics-only comparison.
- **model/**: first neural comparison with original axial metrics, retained for reproducibility.
- **annotated_samples/** and sample_selection.json: first usable-pose test example per posture plus the first detection failure. Actual predictions/measurements are drawn, including missing values. These still-image previews are not a live-camera performance test.
- **metric_distributions.png** and metric_audit.json: descriptive training-only distributions and availability. These are not safe ranges. Their labels originate from the supplied datasets.
- **comparison.json**: validation-only recommendation and all measured reports.

[Open the new Colab](https://colab.research.google.com/github/kg905/SeniorDesignProject/blob/main/notebooks/fusion_geometry_colab.ipynb) or read [the workflow and Pi recording instructions](../../docs/FUSION_GEOMETRY_V2.md). To adapt to the actual doll, record labeled back/side/prone clips from the confirmed overhead camera, with matching JSONL and real session IDs. Terminal photos cannot establish whether landmarks are correctly located.

Pi 5/Pi 4 operation and continuous event timing still require physical testing. The categories describe experimental posture risk; no SIDS-prevention, blocked-airway probability, or 95%-validated safety claim is made.
