# Manual folders in File Explorer → upload one ZIP → train in Colab

This route replaces the notebook labeling interface. You choose labels by copying images into folders on your computer. Colab reads those folders directly. The model remains **MediaPipe → 12 metrics → ONE Random Forest classifier → posture-risk/box/alarm**.

## 1. Start with a manageable pilot

Aim for **about 100 clear, varied images per posture** (300 total), or 100–200 each if manageable. This is for a first experiment, not proof of 95%. Prefer distinct source photos/recordings over many flips, augmentations or neighboring frames of one photo/video. Some images will fail the pose-quality gate, so a small pilot may need more examples before training can split all classes.

Use your original ZIPs. If needed, download them from [data/uploads](https://github.com/kg905/SeniorDesignProject/tree/main/data/uploads) using the file page's Download raw button. Start with `baby-detection.v16i.coco.zip` (the posture export), and add useful full-body images from the other export later. Extract the original archive locally. Its existing train/valid/test folders are just places to find images; you are creating a new reviewed pilot, not preserving those leaky supplied splits.

## 2. Create these folders on your computer

```text
BabyTraining/
  back/
  side/
  prone/
  unknown/        # optional
```

In Windows File Explorer create BabyTraining, open it, then create the four subfolders. Use large thumbnails/Preview pane to inspect each photo. **COPY** chosen images into exactly one class folder; keep original filenames and image contents. Do not rename, crop or resize them just for sorting. Do not copy annotations/README files or entire original train/test directories.

| Folder | Human-assigned meaning |
| --- | --- |
| back | Torso on its back, chest upward |
| side | Torso resting on its left or right side |
| prone | Torso on belly, chest downward; head can be raised or turned |
| unknown | Ambiguous, cropped, obscured or uncertain; not used as posture training truth |

Do not label safe/unsafe based on face visibility or choose a label from the filename alone. The importer uses the folder you choose, not the publisher/model posture label. `belly` is accepted as an alias for prone, but use `prone` for consistency. Optional `no_baby` holds empty-scene fault examples, not a fourth posture class.

If identical filenames from different original folders would overwrite one another, keep them in separate subfolders inside the chosen class rather than replacing a file. Images from the same known recording may be kept in a consistently named recording subfolder across classes; this groups the recording, not a guessed baby identity.

## 3. ZIP the parent folder

Windows: go up one level, right-click **BabyTraining**, then **Compress to ZIP file** (or Send to → Compressed (zipped) folder). Name it `BabyTraining.zip`. On macOS use Compress BabyTraining. The notebook accepts a ZIP with the parent BabyTraining folder or with the class folders directly inside.

Do **not** create train/validation/test folders. We split after import, keeping known duplicates/related sources together.

## 4. Open the new manual-folder Colab

[Open notebooks/manual_folders_colab.ipynb](https://colab.research.google.com/github/kg905/SeniorDesignProject/blob/main/notebooks/manual_folders_colab.ipynb).

1. Use a CPU runtime; run setup cells individually. No label widgets/text-label prompts are used.
2. Upload and process in local Colab storage under `/content/cribwatch-runs`. Drive is optional and is not used for active upload/training writes. **Download section 8 results before resetting/closing the runtime; local storage is temporary.**
3. In the upload section, select **BabyTraining.zip**, not the original unsorted dataset ZIPs. Alternatively set `ZIP_IN_DRIVE` if it is already in Drive; that option reconnects Drive and copies the ZIP locally before processing.
4. The notebook extracts into a new directory and prints class counts, eligible unique counts, duplicate/conflict information, and how many selected images match the original audit.
5. It uses your class folders as human-provided labels. Unknowns stay excluded. Identical/declared-original copies with conflicting folder labels are quarantined; inspect conflicts.json and fix the computer folders before uploading a corrected ZIP.
6. Original file hashes match to the existing dataset audit to preserve known source/augmentation grouping even if filenames differ. New/unmatched images also get conservative exact/pixel/perceptual grouping. None of this establishes infant independence.
7. Continue to verify the two MediaPipe assets, extract the original 12 features, inspect pose coverage and train ONE classifier. The notebook then displays actual results and optional annotated sample scans, and builds a deployment ZIP only from the real trained/evaluated run.

Setup uses a sparse code/reference checkout; it does not download all 5,814 original images again. The original images you select are in your uploaded ZIP. Training does not automatically publish those images to GitHub.

Section 5 now fetches the two task assets directly from official MediaPipe storage with checksum verification and network timeouts. A 132-byte `.task` file is a Git LFS pointer, not a usable model. Matching pointers can be replaced by the verified downloads. Extraction reports initialization, the first completed image, every 10 images, and completion; it checkpoints every completed image. If an old section 5 was waiting on Git LFS, stop that cell, keep the runtime and current RUN/MANIFEST, pull the latest code with LFS smudging disabled, reload `cribwatch.assets` and `cribwatch.data`, and run the updated section 5. Do not repeat upload/import or reset the runtime to fix a model download.

## 5. Keep accuracy claims separate from the pilot

Default evaluation uses source groups and remains **exploratory**. Code targets about 60/20/20 train/validation/test; you do not manually shuffle adjacent frames between them. Unknown/failed-pose predictions count as errors, not discarded test examples. If pose quality fails more than 5% of final test images, 95% all-image accuracy is unattainable under this gate without improving extraction.

The existing trainer needs at least 15 groups and at least two groups per class in every split; 15 alone is not sufficient for every dataset arrangement. Use varied sources in all classes. If training cannot form the splits, add genuinely varied examples rather than inventing identities or disabling the checks.

To support a final 95% posture claim, collect a fresh independently reviewed cohort with actual baby/session IDs. Optional `metadata.csv` in BabyTraining can provide `relative_path,baby_id,session_id` (e.g. `back/photo.jpg,actual_pseudonymous_id,actual_recording_id`). These must be real metadata, not per-image invented IDs. The notebook offers baby-ID evaluation only when those are genuinely available. A target requires ≥95% overall and per-class recall, reported with uncertainty; still-image results do not validate alarm latency/false alarms/hour or medical risk.

The supplied public data/test results have already been inspected. Regrouping/retraining that data cannot establish an untouched external test. A high model score or a few successful live scans does not prove 95% accuracy. No new accuracy result is claimed by adding this importer.

## 6. Add more later

Add/correct images in the same **computer** class folders, make an updated ZIP of the entire BabyTraining folder, and start a new notebook run. Each run reimports your current folder labels and leaves prior local runs intact while that runtime exists. Keep downloaded results on your computer before ending/resetting the runtime. There is no online retraining merely because files were added. Keep original filenames/bytes and known recording grouping. Do not repeatedly tune against a final test cohort already inspected.

For Pi deployment, live pose boxes/risk categories and TCP 5000 alarm behavior, follow [PI5_HANDOFF.md](PI5_HANDOFF.md) and [ALARM_PROTOCOL.md](ALARM_PROTOCOL.md). The new intake route changes labeling convenience, not the agreed feature/risk/model architecture.


## Recovery: Drive “Transport endpoint is not connected” during upload

That means the Drive filesystem disconnected while the older notebook tried to write the ZIP into its Drive run directory. It does not indicate incorrect posture folders. The updated notebook keeps uploads and the entire active run local, and uses Drive only for an explicitly requested source read or final optional backup.

In an already-open old notebook, replace section 3 with this cell, then select the same sorted ZIP again. You do not need to re-sort images, reinstall packages or recreate the ZIP.

```python
import os, sys, json, subprocess
from pathlib import Path
ROOT = Path('/content/SeniorDesignProject')
os.chdir(ROOT)
subprocess.run(['git', 'pull', '--ff-only'],
               env=dict(os.environ, GIT_LFS_SKIP_SMUDGE='1'), check=True)
import importlib
sys.path.insert(0, str(ROOT / 'src'))
importlib.invalidate_caches()
from google.colab import files
from cribwatch.colab_io import local_zip_run
RUN, ZIP_PATH = local_zip_run()
print('Verified local ZIP:', ZIP_PATH)
print('Local run:', RUN)
```

Continue at section 4. The run now lives under /content/cribwatch-runs, so download the trained deployment ZIP in section 8 before disconnecting/resetting the runtime. Source photos remain on your computer. An old “Also in Drive” print label in a previously opened cell does not change the actual local path. The updated section 8 offers an optional explicit Drive backup after local processing; Drive success is never assumed from its earlier Mounted message.


If an already-open session says `No module named cribwatch`, add `sys.path.insert(0, '/content/SeniorDesignProject/src')` before importing project modules. The updated notebook includes this explicitly so a runtime restart is not needed merely to activate an editable-install path. Dependencies from section 2 must still be installed.
