# Fusion Geometry 2 evaluation

[Download SIDS_Pi5_Fusion_Geometry2.zip](https://media.githubusercontent.com/media/kg905/SeniorDesignProject/main/artifacts/team_test_runs/fusion-49p1-744cc500/SIDS_Pi5_Fusion_Geometry2.zip) · [SHA-256 file](SIDS_Pi5_Fusion_Geometry2.zip.sha256) · [Pi 5 / Pi 4 quickstart](../../../docs/PI_FUSION_QUICKSTART.md)

## Recorded evaluation

The uploaded model's [original test report](test.json) covers 875 images in 335 source groups. The table uses the report's confusion matrix. Macro and weighted F1 cover back, side, and prone. Unknown predictions count as misses for the true posture in the all-image row. [Derived metrics](evaluation_metrics.json)

| Evaluation | Images | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| All test images | 875 | 49.1% | 0.592 | 0.597 |
| Assigned postures only | 565 | 76.1% | 0.756 | 0.763 |

The model assigned a posture to 565 images (64.6% coverage) and returned unknown for 310. It classified 430 images correctly and labeled 29 of 543 side or prone images as back. Full-test recall was 54.5% for back, 49.6% for side, and 42.3% for prone. The group-bootstrap 95% interval for all-image accuracy was 44.4%–54.0%.

The 95% research target was unmet. The [validation report](validation.json) covers 846 images and records that the validation coverage constraint was unmet. The reports use source-image groups. They provide no evaluation of performance on unseen infants, continuous alarm events, or clinical outcomes. The [previous team fusion package](../fusion-49p2-98dbc3ec/README.md) was evaluated on a different test cohort.

## Integrity

The exact uploaded ZIP is 18,490,899 bytes. Its SHA-256 is `61866eb10bf2a7b41db64c52cfe74c016aa2bdcf947ca4b282fa605f84db46ac`. ZIP CRCs and all 43 entries in the bundled checksum manifest passed. Both evaluation reports match the packaged weights and configuration hashes, and the configuration matches the bundled detector hashes. [Model configuration](fusion.json) · [Release receipt](release.json)
