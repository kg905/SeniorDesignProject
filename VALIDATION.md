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
