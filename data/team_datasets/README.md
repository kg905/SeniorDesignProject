# Expanded team training dataset

The current fusion notebook expects the user-supplied **BabyTraining-V4.zip** with this structure:

```text
BabyTraining-V4.zip
└── Baby Training - V1/
    ├── back/
    ├── prone/
    ├── side/
    └── unknown/
```

The notebook already discovers this enclosing folder automatically. Back, side and prone supply posture training labels; the unknown folder remains excluded from posture training.

**Dataset ZIP status:** not stored in this repository yet. The chat attachment exceeds the execution environment's 32 MiB transfer limit; the user is placing it in Drive. This README is a record of the expected layout, not a copy of the dataset or a claim that its images have been inspected.

In [the fusion Colab](https://colab.research.google.com/github/kg905/SeniorDesignProject/blob/main/notebooks/fusion_geometry_colab.ipynb), either select the ZIP from your computer in section 3, or set:

```python
ZIP_IN_DRIVE = '/content/drive/MyDrive/BabyTraining-V4.zip'
```

Adjust the path if it is inside another Drive folder. The notebook mounts Drive and copies the ZIP into a fresh local run before extraction; training does not write active files onto the Drive mount. Section 4 prints the actual folder counts. Section 6 trains; section 6A evaluates F1/precision/recall; section 8 exports the matching Pi runtime and trained model.

The Pis do not need the raw training dataset. Use the deployment ZIP on the devices, as explained in the [Pi quickstart](../../docs/PI_FUSION_QUICKSTART.md).
