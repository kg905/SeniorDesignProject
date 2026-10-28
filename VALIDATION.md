# Implementation validation

## Uploaded-data follow-up (current)

- Both original COCO exports were reconstructed from all 12 attachments with chunk SHA256/ZIP CRC verification. The earlier large-file access blocker is resolved.
- Real MediaPipe pose/face extraction completed for **all 5,814 uploaded images**. 3,259 passed the torso-quality gate. Metadata, failure reasons, preprocessing/schema and annotation provenance are retained; expensive inference has checkpoints.
- The final audit separates exact/same-source annotation conflicts from near-duplicate posture changes. It found 188 conservative groups spanning original splits and 44 conflicting declared-source groups affecting 368 images. Publisher posture labels exist for the 4,814-image export; the 1,000-image baby/face export has no posture labels. No individual baby/session IDs were supplied.
- Two exploratory models were trained and evaluated on the same source-group holdout (885 images, 341 groups). Both fell far short of 95%; see `DATASET_RESULTS.md`. Model suggestions remain unknown/unreviewed ground truth. Live monitoring rejects exploratory artifacts.
- **35 automated tests passed**, including COCO provenance, duplicate leakage, legitimate posture changes, conflicts, path traversal, exploratory identity/accuracy safeguards, image-head training, blank-frame handling and proposal provenance, in addition to the original geometry/policy/transport tests.
- The first parallel extraction completed but emitted MediaPipe cleanup errors at process exit because native thread dispatch had shut down before Python atexit callbacks. Cleanup now uses multiprocessing finalizers; a subsequent real two-worker, 12-image extraction completed without those exceptions. Completed feature outputs were intact and reused only for unchanged image hashes/schema/preprocessing.
- The updated Colab notebook's component functions were exercised locally; hosted Colab UI is not verified. No hardware/clinical readiness or environment publication/fresh-task restoration is claimed.

Earlier checks and limitations below are historical; where they state uploads were inaccessible or no infant model was trained, the follow-up above supersedes those statements.

Validated in the provided Debian 13 x86-64 cloud environment with Python 3.12, MediaPipe 1.1.1, OpenCV contrib 4.12.0.88, NumPy 2.2.6, and scikit-learn 1.7.2.

| Check | Result and scope |
| --- | --- |
| Automated tests | **24 passed**: feature geometry/aspect ratio/quality, unknown handling, time-based policy/latching/acknowledgement/gaps, mixed side/prone continuity, authentication/replay/stale links, actual local HTTP exchange, independent baby split, training/serialization, label proposals remaining unreviewed, deduplication and archive traversal rejection |
| Real pose/face inference | Official MediaPipe `pose.jpg` and `portrait.jpg` samples processed. Both produced usable torso geometry; portrait produced 478 face landmarks. Approximately 33–39 ms per sampled image on this cloud machine; not a Pi benchmark |
| Blank input | Actual MediaPipe inference returned `no_pose`, with unknown measurements and no reassuring back classification |
| Real image-to-feature pipeline | Two official adult sample images indexed and extracted to a manifest/12-feature CSV; face landmark output checked. No infant performance implied |
| Video-file pipeline | A 40-frame generated bench clip (official sample plus blank frames) processed through real MediaPipe, a deliberately synthetic test classifier, temporal policy, and JSONL output. Blank segment ended unknown with monitor fault. Synthetic model and temporary inputs were discarded |
| Repeatable setup | Cloud setup script rerun successfully; official model SHA256 checks repeated. Missing `libGLESv2.so.2` resolved with a workspace-local package extracted through signed Debian package metadata; TLS/package/model verification retained |
| Dependency consistency | `python -m pip check`: no broken requirements |
| ARM64 package availability | Downloaded matching Python 3.11 ARM64 wheels for MediaPipe, NumPy, scikit-learn and OpenCV contrib; **not executed on a Pi** |
| Colab notebook | Python cells parsed; underlying extraction/training/video functions exercised locally. **Not run in a hosted Colab browser** |
| systemd templates | `systemd-analyze verify` passed with cloud user/paths substituted. Original templates require the documented Pi user/paths, absent in the cloud. Hardware service startup remains untested |

## Unresolved data and hardware validation

- Subsequent uploaded exports `baby.v5-dataset70_20_10.coco.zip` and `baby-detection.v16i.coco.zip` both exceeded the execution attachment-transfer limit of 32 MiB per file. Both transfer attempts were rejected before local download. Neither archive's size beyond that limit, contents, labels, or license could be inspected. A local browser helper now creates valid ZIP chunks below the limit; it requires all parts plus its manifest to be uploaded before reconstruction/inspection can proceed.
- Four additional transfer tests passed (exact reconstruction, preservation of existing output, corruption rejection/partial-file cleanup, missing/path-escaping parts, and incomplete size metadata). Separately, the actual browser helper's JavaScript generated four valid ZIP chunks, which Python reconstructed byte for byte; the resulting archive passed ZIP integrity validation. These are transfer checks, not dataset analysis.
- After the user reported blocked downloads in the chat preview, the helper was updated to use direct download links and optional folder saving. A real headless Chromium test served the helper locally, prepared a 25 MiB test ZIP, downloaded two chunk ZIPs plus the manifest through their links, and reconstructed the original SHA256 exactly. JavaScript errors were absent. A simulated folder-permission rejection displayed an error while preserving direct links. Actual native folder-picker permission and chat-preview downloads were not validated; the helper instructs users to open the downloaded HTML outside the chat preview.
- No infant datasets were accessible: Roboflow Universe returned HTTP 403; metadata API returned HTTP 401 requiring authentication. Source labels, image counts, permissions and licenses were not inspected. No actual image sorting or infant model fitting occurred.
- No learned infant feature ranges, back/side/prone accuracy, 95% result, airway assessment, or medical outcome is established. Tests with synthetic features validate software only.
- Actual Pi 4/Pi 5 camera, GPIO, buzzer, battery, Wi-Fi access point, reboot behavior, latency, sustained FPS, thermal performance, and physical alarm delivery require the bench checks in `deploy/raspberry-pi.md`.
- The implementation has no independent receiver power-loss detector and no supervised-play/sleep classifier. Face visibility and the camera-depth proxy never suppress a prone alert.
- The 12-feature baseline may be insufficient. Evaluate it on reviewed, identity-separated data before selecting a more capable image/keypoint model or making performance claims.

Next data step: supply authorized exported images and baby/session metadata (or suitable identified recordings), confirm labels through review, then run the notebook. Clinical use requires a separate validation/regulatory process beyond this project prototype.


## 2026-10-27: selected MediaPipe/12-metric classifier and live showcase

User confirmed keeping MediaPipe plus all twelve metrics and ONE posture classifier. The unfinished image-only/TensorFlow path was removed from this update. Original MobileNet/image-baseline reports remain historical results, not the selected deployment.

- Read the attached Pi 5 handoff. It reports Arducam IMX290, Debian 13 ARM64, Python 3.13.5, and TCP 5000 on Pi 4. V10 notebook/receiver source were not provided; exact compatibility with unseen V10 artifacts is not claimed.
- Added label-first Colab workflow using the original 5,814 supplied images, persistent reviews, conservative conflict adjudication, reviewed-only export, optional class folders, and SHA256 rebasing of shared CSVs. Unknowns are not promoted to labels. Actual human review is still pending.
- Preserved the 12-feature schema/preprocessing and one Random Forest. Related identity/session/source groups stay together; new training reports bind the serialized classifier SHA256. No improved infant model was trained or 95% result established by this change.
- Added MediaPipe-derived normalized pose box, posture-risk overlay, optional scan JPEG/JSON and annotated MP4, fresh-frame/inference watchdog, and explicit moderate/high alarm trigger setting. A box is a landmark-derived region, not a separate validated baby/airway detector; some occluded body parts may lie outside it.
- Local real inference on three previously accepted baseline examples exercised pose/face extraction, the existing serialized classifier, all 12 metrics, pose boxes and risk overlays. Those selected examples are a functional check, not an accuracy estimate.
- Local six-second replay-camera harness exercised real MediaPipe + baseline classifier -> overlay/recording -> authenticated TCP receiver -> latched alert. It wrote 30 status samples, observed an alarm and a connected receiver, then stale-image unknown/fault and an explicit shutdown fault. This was a software bench replay, not a live Pi/infant hardware test.
- Full Python suite: **44 passed**. New coverage includes box geometry/rendering, risk-trigger selection, latch continuity, repeated-frame rejection, persistent stale faults, variable-resolution video recording, human-label provenance/conflict handling/CSV rebasing, actual TCP authentication/replay/version/timeout, and handoff ZIP contents/checksums/standalone imports. Handoff packaging tests use explicitly artificial features, not validated infant artifacts.
- `pip check` passed in the existing MediaPipe/sklearn environment. Notebook source cells parsed; hosted Colab UI/training execution remains unverified. The Pi 4 receiver launcher runs with Python standard library only when GPIO is omitted.
- Official PyPI metadata lists MediaPipe 1.1.1 `py3-none-manylinux_2_28_aarch64` wheel. It does not justify installing the older CPython 3.11-only wheel on Python 3.13. Actual ARM64 import, camera/color, GPIO, latency/temperature, private Wi-Fi, audible output and battery/failure tests remain with the hardware teammate.

95% requires reviewed labels, much better pose coverage than the original ~56%, a fresh infant/session-independent cohort, honest unknown-as-error reporting/per-class recall/intervals, and separate continuous-event testing. No clinical/SIDS-prevention claim is made. The GitHub update supplies the workflow and integration code, not completed label review or trained-accuracy validation.


## Colab review interface fix

The user reported a completed labeling cell with no visible images, including after enabling custom widgets. The old reviewer displayed child widgets inside an Output capture before displaying the parent; a hosted Colab frontend failure was not reproduced locally. Removed nested Output capture from the optional widget and added the default `review_console` image/input reviewer, which uses normal notebook image output and Python input instead of the widget renderer. It saves each explicit label immediately, resumes by reading reviewed flags, does not label an interrupted/current photo, checks image hashes, and reloads the CSV before saving so stale batches cannot erase labels saved elsewhere. The dropdown reviewer reports errors visibly and preserves progress.

The full suite passed: **49 tests**, including five new targeted tests for label/quit/invalid-input handling, restart/resume, modified-image rejection, preservation of other saved rows, and the explicit widget image tree/save callback. The notebook's default section 4 and troubleshooting instructions use the no-widget path. Actual hosted Colab/browser rendering remains to be checked by the user; no claim is made that it was remotely tested here. Original images and review files were not modified by this fix.


## Manual File Explorer folder intake

User prefers manually copying labeled images into computer folders and uploading a sorted ZIP instead of using the review UI. Added docs/MANUAL_FOLDERS.md, notebooks/manual_folders_colab.ipynb and cribwatch import-folders. Labels come only from explicit user class-folder assignments (back/side/prone, belly alias; unknown/no-baby excluded from posture training), not publisher/model predictions or source filename words. Exact/pixel/declared-original conflicting labels are quarantined; source/near-duplicate/optional real identity/recording links are kept together. An optional metadata.csv can carry real identities; none are invented. Original files are not moved and outputs refuse overwrite.

A local four-image ZIP fixture from existing images plus a blank scene passed safe extraction -> manual folder import -> original SHA256 matching (three images) -> actual MediaPipe extraction of all twelve features. The blank unknown scene stayed excluded and failed the pose gate. This was an intake/feature plumbing check, not new reviewed infant ground truth, training, or an accuracy result. Sparse Git source/reference checkout was exercised and confirmed no raw data/datasets download while the original audit and model pointers were available. Actual hosted Colab UI/upload and a new user-labeled model remain unrun.

Full suite: **55 tests passed**, including six new tests for folder labels overriding misleading filenames/publisher labels, duplicate-label quarantine, related-source grouping across genuine posture changes, ZIP wrapper/Apple metadata, real identity metadata, malformed layout/corrupt files/overwrite protection. pip check and notebook source parsing passed. The MediaPipe + 12 metrics + one Random Forest, scan overlay and TCP 5000 handoff are unchanged. A small pilot helps iteration; neither 95% nor clinical/SIDS performance is established.


## Local Colab upload and active-run storage

The user reported browser upload reaching 100%, then OSError errno 107 (Transport endpoint is not connected) while files.upload opened BabyTraining.zip inside a Google Drive-backed incoming directory. Updated manual_folders_colab section 3 to use local /content/cribwatch-runs for uploads, extraction, feature output and training, avoiding active writes to the disconnected FUSE mount. An optional already-in-Drive source is explicitly remounted and copied locally. The source ZIP is checked for size/CRC and fingerprinted before processing. Prior review files and Drive runs are not deleted.

Section 8 now explicitly requires downloading the trained handoff before runtime reset, and offers optional Drive backup only after local processing completes. Local persistence is not implied. The manual guide includes a recovery cell for an already-open notebook; the user must reselect the same sorted ZIP, not recreate or re-sort it.

Full suite: **60 tests passed**, including five new upload tests for local destination/cwd restoration, a previous working directory disappearing during transfer, copying existing source ZIPs locally, incomplete/invalid ZIP rejection, and refusing a Drive work_root. Notebook source parsed. Hosted Colab/browser upload, actual remount behavior, the user's ZIP contents, and its later model accuracy were not inspected or verified here. Feature/model/alarm logic is unchanged.
