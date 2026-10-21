# Uploaded dataset results

**The 95% target was not achieved. Neither trained baseline is suitable for a baby safety alarm.**

All 12 uploaded files were received. Both original archives were reconstructed with per-part SHA256 and ZIP CRC checks: 43,360,504 bytes and 199,283,513 bytes. The data contain 5,814 images: 1,000 baby/face-box images and 4,814 posture-box images.

The posture export already labels supine/back, side, and prone. The baby/face export does not label posture. Publisher annotations are imported with provenance, not marked as independently verified. Export metadata declares Public Domain/CC0 and CC BY 4.0 respectively; underlying permissions were not independently audited.

## Audit and data preparation

- 5,767 unique file hashes; 47 exact duplicate files.
- 188 conservative duplicate/source groups span the supplied splits. New exploratory splits keep groups together.
- 44 declared-source groups have contradictory posture labels, affecting 368 images; these are quarantined. Near-duplicate frames can show genuine posture changes and are grouped, not automatically quarantined.
- 4,421 images remain eligible under the publisher labels before exact-file deduplication.
- No baby identities or recording identities were supplied. Source-image grouping does not prove independence between babies, sessions, rooms, or cameras.
- Original supplied files/images were preserved. No labels were inferred from filename words such as Supine or Sideways.

## Feature extraction and training

All 5,814 images were processed with real MediaPipe pose and face inference. The 12 requested metrics and quality/failure reasons were saved. To match the posture export, inference consistently stretches RGB input to 640x640 (feature schema 2). There is no claim that stretched pixels preserve physical distances.
Usable torso landmarks: 3,259/5,814. All labeled failures count as unknown/error in evaluation; they are not silently dropped.

| Model | All-image accuracy (unknown = error) | Coverage | Accuracy on accepted only | Diagnostic without score rejection |
| --- | ---: | ---: | ---: | ---: |
| 12 metrics | 18.8% | 32.3% | 58.0% | 28.6% |
| Image + metrics | 33.9% | 52.1% | 65.1% | 55.5% |

Both models were evaluated on 885 publisher-labeled images in 341 source-image groups. Model/head/score rejection were selected using validation only. These are exploratory publisher-label results, not an untouched independent clinical test. The image model uses a frozen generic MobileNetV3 embedding plus the 12 metrics and a small learned head; it is not an infant-specific fine-tuned neural network.

Metrics-only pose failures cap attainable coverage well below 95%. The image model improves raw posture classification but remains far below the target. Additional hyperparameter sweeps against these observed results would not establish independent validation.

See `posture_confusion_matrices.png`, `metric_distributions.png`, each run's `test.json`, and `model_comparison.csv`. Descriptive 5th/50th/95th training percentiles are in `training_feature_ranges.csv`; their overlap prevents turning them into universal safe/unsafe ranges.

## Label review queue

A review queue contains 1,025 unknown-label images, including all 1,000 images from the baby/face dataset. Suggestions for that 1,000-image dataset: {'back': 321, 'prone': 153, 'side': 133, 'unknown': 393}. These remain proposals: posture stays unknown and human_verified/reviewed stay 0 until an independent reviewer confirms them.

The 1,000-image face dataset must not be evaluated using model-generated posture proposals as ground truth. Conflicting publisher labels need adjudication, not automatic majority voting. The visual spot review also contained ambiguous views and examples that warrant checking the publisher label.

## What remains necessary

1. Independently adjudicate posture/face/context labels, including conflicts and ambiguous cases. Preserve uncertainty.
2. Obtain identified, permitted recordings from the intended crib/camera setup and create a held-out infant/session cohort before further tuning. Do not invent IDs to pass the training guard.
3. Develop and evaluate an infant-specific keypoint or fine-tuned image model; the two tested baselines are inadequate. This requires additional model-development work and a fresh independent test set, not a chosen numerical safety cutoff.
4. Evaluate continuous-event sensitivity, latency, false alarms/hour, occlusions/IR, and camera/network/power failures on Pi hardware. No actual Pi hardware or continuous infant recordings were available here.
5. Treat airway obstruction, awake/sleep state, physical head height and SIDS prevention as unvalidated: none has ground-truth labels in these exports. A visible face never automatically establishes safety.

Both artifacts are marked exploratory and are rejected by live monitoring. Offline video analysis requires `--allow-exploratory`; that flag is deliberately absent from the live monitor. Source, dataset exports and final artifacts can be versioned together through Git/Git LFS. GitHub publication does not establish model safety, publish the cloud environment, or verify restoration into a fresh cloud task.
