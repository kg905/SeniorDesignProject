# Improve the current 12-metric model

The pilot supplied through Colab has 179 usable poses among 305 eligible images (58.7%), including 48/101 prone images. The screenshot reports 14.5% test accuracy and 21% prediction coverage. These are user-supplied results, not a run reproduced in this cloud workspace. The manually sorted ZIP is not available here. The first experiment must separate extraction failures from classification failures.

## Keep the current Colab session

Do not repeat upload/import or reset the runtime. After your extraction/training cells have stopped, add this new code cell. It pulls the updated source, reuses your completed Lite features, extracts Full separately, compares seven classifier configurations, and evaluates one selected pipeline. Its output is a separate improvement directory; your existing model remains available.

```python
import os, sys, subprocess, importlib
from pathlib import Path
ROOT = Path('/content/SeniorDesignProject')
sys.path.insert(0, str(ROOT / 'src'))
subprocess.run(['git', 'pull', '--ff-only'], cwd=ROOT,
               env=dict(os.environ, GIT_LFS_SKIP_SMUDGE='1', GIT_TERMINAL_PROMPT='0'),
               check=True, timeout=120)
for name in ['assets', 'data', 'learning', 'optimize']:
    importlib.reload(importlib.import_module('cribwatch.' + name))
from cribwatch.optimize import run_optimization
BASELINE_FEATURES = globals().get('BASELINE_FEATURES', FEATURES)
IMPROVEMENT = Path(RUN) / 'improvement-v1'
result = run_optimization(
    MANIFEST, IMPROVEMENT, ROOT / 'models',
    existing_features=BASELINE_FEATURES, variants=('lite', 'full'),
)
MODEL = Path(result['model_dir'])
FEATURES = Path(result['features'])
report = result['report']
print('Selected model:', MODEL)
print('Accuracy:', f"{report['accuracy_all_images_unknown_counts_as_error']:.1%}")
print('Coverage:', f"{report['coverage']:.1%}")
for label, value in report['per_class'].items():
    print(label, 'recall:', f"{value['recall']:.1%}")
print('Rejected/accepted:', report['rejection_reasons'])
print('Comparison:', result['comparison_path'])
```

Continue with the existing notebook's section 7 for annotated scans and section 8 for the deployment ZIP. Those cells use the newly assigned MODEL. The ZIP includes exactly one classifier and whichever pose asset was used to train it. Download before resetting Colab. To inspect the detailed comparison/errors and training/validation failure images, use section 6 of the improvement notebook below, or share comparison.json and best-model/test.json for diagnosis. Do not tune on test image examples.

For a fresh session, use [improve_model_colab.ipynb](https://colab.research.google.com/github/kg905/SeniorDesignProject/blob/main/notebooks/improve_model_colab.ipynb). Sections 1–4 upload your same sorted BabyTraining.zip; no re-sorting is needed.

## What the comparison changes

- MediaPipe Lite versus Full, with unchanged 640×640 preprocessing, geometry, visibility/presence gate, and 12 input metrics. Heavy is optional in a new output directory, not enabled by default because speed must also be tested.
- Seven classifier configurations: three Random Forest, two Extra Trees, and two histogram gradient boosting configurations. Boosting tests the tree-boosting idea without adding XGBoost/PyTorch dependencies. Only one configuration is exported.
- Fit training rows only. Imputation is fitted on training data. Retain identical source/duplicate grouping and split seed across extractors. Select classifier and extractor by validation: highest minimum class recall, then mean class recall, then fewer side/prone images predicted back, then overall accuracy. Score thresholds still use validation only, the existing 0.5–1.0 search, and the existing minimum-coverage rule. No gate is lowered to inflate coverage.
- Candidate stages do not generate test predictions or test reports. After choosing the extractor, train the selected recipe and evaluate its test split. This does not make a previously viewed pilot test set untouched or establish independent infant performance.
- Report per-posture pose coverage, extraction failure reasons, median/p95 inference times on this machine, final rejection reasons, the confidence-ungated diagnostic, class recall, and checksum-bound reports.
- Pin exact pose/face asset hashes in the classifier, enforce them at live/offline inference, and bundle the matching extractor. Lite/face hashes are pre-pinned. Full/Heavy assets come from versioned official Google HTTPS URLs and their hashes are recorded at download time, not pre-pinned in the repository.

The comparison reuses completed CSVs/checkpoints and completed validation stages when the same inputs and output directory are used. Changed labels/configuration require a new directory. Keep the same BASELINE_FEATURES value on reruns. If a variant fails, comparison.json records its error; a remaining working variant can finish, but this is not evidence that the failed variant performed poorly. If none finish, errors.json explains the failures. Partial model-export writes may require a fresh improvement directory; completed feature extraction can still be supplied separately.

## What to do with the result

If pose coverage is still low, address extraction before labeling hundreds more photos. Review training/validation failures and test an infant-tuned keypoint detector or collect suitable keypoint annotations. More general-purpose classifier settings cannot recover an undetected pose. If pose coverage improves but posture accuracy remains poor, examine feature overlap and collect varied, representative reviewed examples from the actual camera setup. The 12 metrics may not separate back from prone reliably.

Measure speed, frame freshness, alarm delays, and false alarms on the actual Pi 5/Pi 4 system. Colab extraction times do not prove Pi throughput. Full/Heavy can be slower and must meet the existing stale-frame limits; increasing those limits merely to hide slow inference is not validation. No model accuracy or hardware improvement is claimed by these code changes.

See [the external project review](REFERENCE_PROJECT_REVIEW.md) for how the suggested infant-YOLO pipeline differs from the required metrics.
