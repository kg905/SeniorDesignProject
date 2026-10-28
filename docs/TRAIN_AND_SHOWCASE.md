# Label first, train one classifier, then demonstrate a live risk overlay

**Manual folder alternative:** the user now prefers File Explorer sorting. Use [MANUAL_FOLDERS.md](MANUAL_FOLDERS.md) and [manual_folders_colab.ipynb](../notebooks/manual_folders_colab.ipynb) to upload one sorted ZIP directly. The review-interface instructions below remain optional; previous Drive reviews are preserved.

**Agreed architecture: camera → MediaPipe pose/face → your 12 metrics → ONE posture classifier → posture-risk level → timed alarm.** We are keeping MediaPipe. MobileNet/image embeddings are not used by this selected path. The original ZIPs and 5,814 extracted images are already in this repository; no more ZIP uploads/splitting are needed.

## Step 1 — Open the label-and-train notebook

[Open notebooks/label_and_train_colab.ipynb in Colab](https://colab.research.google.com/github/kg905/SeniorDesignProject/blob/main/notebooks/label_and_train_colab.ipynb). Use a CPU runtime. Run the setup cells, mount your private Drive folder, and prepare the review CSV. This uses the datasets you already supplied, not replacement data.

The notebook keeps your saved reviews in Drive and the originals unchanged. It prioritizes conflicting groups, unlabeled images, then other publisher labels. It does not automatically mark machine suggestions as correct.

## Step 2 — You separate/review the images

The default reviewer does not require widgets. It displays one photo, then asks for **b** (back), **s** (side), **p** (belly/prone), **u** (unknown/unclear), **n** (no baby), or **q** (stop). Press **Enter**: each label is saved immediately. After 12 images, rerun the cell to continue. There is no Save Batch button in this mode. An unlabeled/unsaved photo remains pending. `no_baby` examples are for background/fault tests, not a fourth posture classifier.

Optional face information can be typed after the label, e.g. `p visible` or `p occluded`. Existing real baby/session IDs are preserved; leave them blank unless actually known. The optional dropdown reviewer still supports those fields and Save reviewed batch, but the default image/input reviewer avoids blank-widget failures in Colab.

You do not need to move files or create train/test folders. Your labels are recorded in a CSV. Optionally export reviewed copies into back/side/prone folders, but these are class folders, not evaluation splits. The first export includes human-reviewed non-conflicting labels only. Ambiguous images remain excluded.

Review the 368 images with conflicting source labels together. The notebook's resolution helper clears a publisher conflict only when the whole original-source group has been human-reviewed and agrees on one known label. Incomplete/disagreeing groups stay quarantined; do not bulk-clear flags. Identical-image label disagreements must be resolved before export. The 1,000 baby/face images have no posture truth until you review them.

The default image/input reviewer never shows publisher/model suggestions. For an independent second reviewer use a separate fresh review copy; the optional widget also supports `blind=True`. Resolve test-label disagreements before showing reviewers model predictions.

**Stop after labeling/export if you want us to retrain next.** Share the reviewed CSV plus any additional images it references through an authorized private route. Labels for the already-uploaded images can be matched using their SHA256; the same images need not be uploaded again. Do not publish private infant identities or recordings in this public GitHub repository.

## Step 3 — Extract the SAME 12 MediaPipe metrics

Only after review, the next notebook section processes the reviewed manifest with the pinned official `pose_landmarker_lite.task` and `face_landmarker.task`:

```bash
cribwatch extract /private/reviewed.csv --models models --output /private/features.csv --workers 2
```

Feature order and definitions are in `src/cribwatch/features.py`. Preprocessing remains schema 2: uint8 RGB stretched to 640x640, projected widths normalized by torso length, axial angles, four visibility scores, face status/count, and a camera-depth head proxy. The last is not actual head height/airway clearance. No clinical safe/unsafe ranges are invented from training percentiles.

**Check detection coverage before expecting 95%.** Usable torso landmarks existed in only about 56% of the original images. The metrics-only classifier returns unknown when pose quality fails. If more than 5% of evaluation images fail extraction, 95% all-image accuracy is mathematically unattainable with that gate, even with a perfect classifier. Do not remove those images from the reported denominator to hide the problem.

Improve the intended camera framing, image quality and representative data first. A fuller infant-appropriate landmark model may need separate evaluation if Lite still fails; the packaged workflow deliberately preserves the existing verified assets/preprocessing. Better posture labels alone cannot repair every landmark failure.

## Step 4 — Split without leaking related images

The reviewed CSV determines which images are eligible. Code creates approximately 60% train / 20% validation / 20% test splits; you should not shuffle frames manually into folders.

- For initial improvement with unknown identities, use source/duplicate groups and mark the run **exploratory**. This can measure improvement on the supplied labels but cannot establish unseen-infant accuracy.
- For the final claim, collect a fresh permissioned cohort with real pseudonymous baby and recording IDs. Keep every image of a baby together across sessions; keep recording and known duplicate/source groups from crossing splits. Do not invent identities for scraped images.
- Have the reserved test cohort independently reviewed; keep it out of all model/threshold selection. The existing public test results have already been examined, so regrouping them does not create an untouched final cohort.

Add representative Arducam daytime/night/IR, clothing, motion and occlusion examples. Use a doll for camera/alarm setup, not infant-recognition validation. Do not position an infant unsafely for data collection. At least 15 groups is only the software's basic prerequisite, not a sufficient validation study size.

## Step 5 — Train ONE classifier

The selected implementation trains a class-balanced 200-tree Random Forest with median imputation/missing indicators. MediaPipe provides the inputs; it is not a second competing posture classifier.

```bash
# Initial research run: human-reviewed data, source groups, unknown identities.
cribwatch train /private/features.csv --exploratory --output /private/run-001
# Final cohort with reviewed labels AND actual baby/session IDs: omit --exploratory.
```

The confidence threshold is selected using validation data, not the final test. Training percentiles describe the data, not medical cutoffs. If validation remains poor, inspect label/feature errors, improve data/extraction and revise classifier settings using validation only. More trees/epochs cannot guarantee 95%.

## Step 6 — Read the evaluation before making a claim

Training writes `test.json`, confusion matrix, per-class recall, coverage, accepted-only accuracy, and group-bootstrap uncertainty. All-image accuracy counts unknown predictions as errors. To state our 95% posture point target, require **at least 95% overall AND at least 95% recall in each back/side/prone class**, on a fresh reviewed infant-independent cohort. State the confidence interval; a 95% point estimate is not a 95% lower confidence bound.

If the target is not reached, report the actual score and improve the training/validation process. Do not repeatedly optimize against the already observed test set and call it independent. Neither risk/airway accuracy nor SIDS prevention follows from posture accuracy.

Then evaluate continuous recordings separately for missed posture episodes, delay to alarm, unknown time and false alarms/hour. Empty crib, camera obstruction, multiple people and camera/network failures must not silently become lower risk. Set event/nuisance-alarm targets before examining final recordings.

## Step 7 — Package the hardware teammate's files

```bash
python deploy/single-model/build_handoff.py --model /private/run-001 \
  --output /private/SIDS_Pi5_Deployment.zip
```

This includes the one trained `posture_model.joblib`, its checksum-bound report, actual MediaPipe assets, descriptive feature percentiles, runtime source, launchers and checksums. It refuses missing/mismatched exports. No model files or 95% reports are fabricated. The PDF's V10 notebook was not supplied: this preserves the repository's actual feature/classifier/policy code, not unseen V10 artifacts named `feature_ranges.joblib` or `risk_cutoffs.joblib`.

## Step 8 — Pi 5 live scan: box + risk level

Follow [PI5_HANDOFF.md](PI5_HANDOFF.md). Keep the teammate's working `sids_env` intact; use a separate environment with MediaPipe 1.1.1 and sklearn 1.7.2. Do not install the downloaded CPython 3.11 MediaPipe 0.10.18 wheel into Python 3.13.

On a Pi desktop/HDMI display, from the deployment ZIP folder:

```bash
python monitor_pi5.py --model models --models models --picamera --display --showcase \
  --alarm-risk moderate --log demo-recordings/run-001.jsonl --record demo-recordings/run-001.mp4
```

Each classified frame shows a **box estimated from visible body landmarks**, posture, **lower/moderate/high/unknown posture risk**, raw model score, timestamp, fault/link condition and alarm state. The box is not an independently validated baby detector or airway localization; it can miss occluded parts. When pose quality is unreliable there is no trusted box and the result is unknown.

Press **s** to save an annotated photo and JSON; **q** stops. The overlay belongs to the exact frame classified, not a newer frame. Repeated display of one frame cannot accumulate fresh alarm evidence. Stale frames or hung inference show unknown/fault instead of retaining reassuring readings. MP4 is opt-in constant-rate output; use JSONL timestamps for actual timing/performance.

`--showcase` explicitly labels a below-target research demonstration. It does not validate the model. Without it, the new display checks for a matching report that meets the posture point target. The earlier 18.8% metrics model can be used to exercise the display explicitly as an experimental baseline; do not present it as the improved model.

## Step 9 — Trigger the parents' alarm at the chosen risk level

Follow [ALARM_PROTOCOL.md](ALARM_PROTOCOL.md), adopting the provided replacement receiver on both devices. Use **TCP 5000**, not UDP 5005 or the old HTTP 8765 path. The PDF's receiver source was not supplied, so wire compatibility with it has not been assumed.

- `--alarm-risk moderate` (default): sustained side/moderate OR prone/high can alarm.
- `--alarm-risk high`: only sustained prone/high triggers the posture alarm.
- Unknown/lost monitoring remains a separate fault regardless of the chosen posture trigger.

Prototype delays are side 2 seconds and prone 1 second. These are engineering settings, not medical thresholds. The alert remains latched until explicit acknowledgement after stable back for 3 seconds. Muting does not clear it.

With the same protected shared key loaded on both devices:

```bash
# Pi 4 (console-only first; add properly wired GPIO after testing)
python3 alarm_receiver.py --gpio 17 --mute-button 27
# Pi 5: add the Pi 4's actual private address
python monitor_pi5.py --model models --models models --picamera --display --showcase \
  --alarm-host 192.168.50.1 --alarm-risk moderate --ack-file /tmp/cribwatch-ack
```

## Step 10 — Rehearse and present evidence separately

Compare fixed permitted reference images in Colab/Pi using the same classifier/asset hashes. Test actual camera color/orientation, box alignment, pose coverage, score/state output, timing, latching, acknowledgement, mute, lost camera/inference/Wi-Fi, reboot recovery, actual sound and battery duration. Measure Pi FPS/temperature rather than guessing.

For the showcase, show both the live scan and the separate held-out evaluation report, including errors/uncertainty. A few successful scans do not prove 95% accuracy. No actual Pi tests or improved reviewed-data accuracy have been completed in this workspace.


## If an already-open Colab notebook shows no labeling interface

Replace its step 4 cell with the following. This fast-forwards the code checkout, skips unnecessary LFS downloads, and resumes the existing Drive CSV. It does not recreate/delete your labels or reinstall dependencies.

```python
import os, subprocess
subprocess.run(
    ["git", "-C", "/content/SeniorDesignProject", "pull", "--ff-only"],
    env=dict(os.environ, GIT_LFS_SKIP_SMUDGE="1"), check=True
)
from cribwatch.review_console import review_console
review_console(REVIEW, batch_size=12)
```

The photo and the text input appear directly below this cell. Type a label and press Enter; q stops. No custom widget manager is needed. If REVIEW is undefined after a runtime restart, rerun sections 1 and 3 first without deleting the saved review CSV.
