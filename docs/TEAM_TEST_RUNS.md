# Team test packages

The current uploaded Colab package is **[colab-31p5-4094df16](../artifacts/team_test_runs/colab-31p5-4094df16/README.md)**. Download its ZIP using Download raw, unpack into a separate folder, and follow its run instructions. It contains the user's trained hybrid with 31.5% test accuracy on 267 images. Its exact archive and classifier checksums are recorded; this package is preserved independently of ongoing new-model development.

The older [original-data hybrid](../artifacts/hybrid_alternative_v1/README.md) remains available for comparison. It is a different trained artifact and reports 29.1% accuracy on a different evaluation cohort. Do not compare these percentages as if they used the same test images or mix one package's classifier with another's feature code/assets.

New geometry overlays and a new model are under development. Their source, reports and packages will receive separate versioned paths. A file being uploaded or a display being improved does not change a trained model's measured accuracy.

The geometry companion is now available next to the exact uploaded ZIP; it uses the same 31.5% classifier with updated drawing/capture code. The separately trained [new geometry + image/metric model](../artifacts/fusion_geometry_v2/README.md) and [its Colab](https://colab.research.google.com/github/kg905/SeniorDesignProject/blob/main/notebooks/fusion_geometry_colab.ipynb) are also available. Its 50.1% result is on the original-data 882-image cohort and is not a score on the user's 267-image Colab cohort. Neither model meets 95%.
