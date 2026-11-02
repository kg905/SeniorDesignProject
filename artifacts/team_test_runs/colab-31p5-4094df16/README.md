# Current team test package: user's Colab-trained hybrid

Download **[SIDS_Pi5_Hybrid_Alternative.zip](SIDS_Pi5_Hybrid_Alternative.zip)** with GitHub's **Download raw** button. This is the exact ZIP uploaded by the user, preserved byte for byte. It is separate from the older publisher-label hybrid and from the new geometry/fusion work in progress.

| Measurement | This uploaded model |
| --- | ---: |
| Overall model score (test accuracy; unknown = error) | 31.5/100 |
| Test images | 267 |
| Prediction coverage | 51.3% |
| Back recall | 45.8% |
| Side recall | 13.8% |
| Prone recall | 28.3% |

The report is readable in [test.json](test.json). This is an experimental test package, not a 95%-validated model. Classifier: histogram gradient boosting, 7-leaf setting; acceptance threshold 0.55. YOLOv8 nano pose and MediaPipe provide the feature inputs.

## Identify the exact package

- ZIP SHA256: `a1cb912329d68c211aefb974397798c7afc2da6037f504a7b60bbc34692a97aa`
- Classifier SHA256: `4094df169c6275f5857ba94f8b136a32730221bbf309b2dda39148cb365500d5`
- Release metadata: [release.json](release.json).

All ZIP entries passed CRC checks, every entry listed in SHA256SUMS.json matched, and the classifier matched its checksum-bound test report. Intake verified files; it did not execute the uploaded model/code or test physical Pis.

## Group test instructions

1. Unzip into a **new** folder, such as `~/senior_design/team_test_4094df16`. Keep older deployment folders. Run commands from this unpacked folder so the frozen bundled source and matching model are used.
2. Follow the ZIP's **README.md** for Pi 5/Picamera2 setup, and **HYBRID.md** for extra Torch/Ultralytics requirements. Keep the bundled pinned versions and the single OpenCV contrib distribution. The Pi 4 receiver does not need the vision dependencies.
3. Check saved photos first:

```bash
python test_model.py --model models --models models --showcase \
  --output test-photos-001 /path/to/back.jpg /path/to/side.jpg /path/to/prone.jpg
```

4. Run from the Pi desktop for live camera, risk display, and an annotated recording:

```bash
python monitor_pi5.py --model models --models models --picamera --display --showcase \
  --alarm-risk moderate --log test-runs/run-001.jsonl --record test-runs/run-001.mp4
```

Use fresh output filenames per run. For a physical alarm, follow bundled PROTOCOL.md, run its Pi 4 receiver on TCP 5000, provision the same protected ALARM_SHARED_KEY on both devices, then add `--alarm-host <Pi4-IP>` to the Pi 5 command. A connected link alone does not prove that a posture alert or buzzer works. Verify actual sound and monitor-fault behavior in a controlled doll test.

5. Record the **actual** doll posture, camera placement, lighting, and whether people/hands entered the view for each trial. Keep video/JSONL together. Confidence and valid-detection rate are not posture accuracy; compare predictions against the actual labels. Capture the camera frame itself rather than only a terminal photograph, especially for failed prone examples.

The screenshots' custom “FINAL TEST SUMMARY” script is not part of this uploaded ZIP. Keep a copy of that script with any results if it is used; do not assume it shares the bundled monitor's fault logic. This package predates the new detailed geometry overlay.
