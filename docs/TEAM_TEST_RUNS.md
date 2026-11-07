# Team test packages

The latest uploaded package is **[fusion-49p2-98dbc3ec](../artifacts/team_test_runs/fusion-49p2-98dbc3ec/README.md)**: the exact `SIDS_Pi5_Fusion_Geometry.zip`, verified checksums, model configuration, original report and F1 calculations. Its report contains 512 test images: 49.2% overall accuracy, 68.0% coverage and 72.4% accuracy on assigned postures. Macro F1 is 0.580 for all images and 0.716 for assigned postures only. Follow the [updated Pi 5 / Pi 4 quickstart](PI_FUSION_QUICKSTART.md). This is a different test cohort from the earlier packages, not a controlled before/after comparison.

The expanded dataset is `BabyTraining-V4.zip`, enclosing `Baby Training - V1/back`, `prone`, `side` and `unknown`. Its chat attachment exceeds the 32 MiB transfer limit and is not stored in this repository yet; the user is providing it in Drive. [Dataset usage](../data/team_datasets/README.md). The fusion notebook now includes section 6A for F1/precision/recall and accepted-only coverage reporting.

The earlier uploaded Colab package is **[colab-31p5-4094df16](../artifacts/team_test_runs/colab-31p5-4094df16/README.md)**. Download its ZIP using Download raw, unpack into a separate folder, and follow its run instructions. It contains the user's trained hybrid with 31.5% test accuracy on 267 images. Its exact archive and classifier checksums are recorded; this package is preserved independently of ongoing new-model development.

The older [original-data hybrid](../artifacts/hybrid_alternative_v1/README.md) remains available for comparison. It is a different trained artifact and reports 29.1% accuracy on a different evaluation cohort. Do not compare these percentages as if they used the same test images or mix one package's classifier with another's feature code/assets.

New geometry overlays and a new model are under development. Their source, reports and packages will receive separate versioned paths. A file being uploaded or a display being improved does not change a trained model's measured accuracy.

The geometry companion is now available next to the exact uploaded ZIP; it uses the same 31.5% classifier with updated drawing/capture code. The separately trained [new geometry + image/metric model](../artifacts/fusion_geometry_v2/README.md) and [its Colab](https://colab.research.google.com/github/kg905/SeniorDesignProject/blob/main/notebooks/fusion_geometry_colab.ipynb) are also available. Its 50.1% result is on the original-data 882-image cohort and is not a score on the user's 267-image Colab cohort. Neither model meets 95%.
