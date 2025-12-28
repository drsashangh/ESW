# KY-038 Cough Detection Model

Train a simple ML model using the COUGHVID public dataset to emulate a KY-038 (analog microphone) cough detector, then run real-time inference over serial using readings printed by an Arduino/ESP32 sketch.

## 1. Prepare Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r data_collection/ky038_cough_model/requirements.txt
```

## 2. Training

By default, the script looks at `data_collection/COUGHVID-dataset/public_dataset`.

```bash
python data_collection/ky038_cough_model/preprocess_and_train.py \
  --label_threshold 0.5 \
  --model_type gb \
  --max_files 1000  # optional for a quick run
```

Outputs model artifact: `data_collection/ky038_cough_model/models/ky038_cough_model.joblib`

Adjust `--label_threshold` if you want to be stricter (e.g., 0.7).

## 3. Flash / Run Sensor Sketch

Use (or adapt) `CODES/sensor_codes/microphone.ino` on your board. Ensure it prints lines like:

```
Sound Level: 1234
```

and at a regular interval (e.g., every 100 ms).

## 4. Real-Time Inference

```bash
python data_collection/ky038_cough_model/realtime_infer_serial.py \
  --port /dev/ttyUSB0 \
  --model data_collection/ky038_cough_model/models/ky038_cough_model.joblib \
  --window_s 1.0 \
  --sample_period_s 0.1 \
  --prob_threshold 0.5
```

Outputs lines like:

```
prob=0.812  COUGH
prob=0.132  no_cough
```

### 4b. ThingSpeak mode (ESP32 publishes, PC polls)

1) Flash `CODES/sensor_codes/microphone_rms.ino` (ESP32):
   - Fill in `WIFI_SSID`, `WIFI_PASS`, `THINGSPEAK_WRITE_API_KEY`.
   - Default publish interval is 15 s (ThingSpeak free tier limit). The sketch still prints 100 ms RMS locally over Serial for debugging, but only publishes the average RMS every 15 s to `field1`.

2) Install extra dep for polling:

```bash
pip install -r data_collection/ky038_cough_model/requirements.txt
```

3) Run the ThingSpeak polling inference:

```bash
python data_collection/ky038_cough_model/realtime_infer_thingspeak.py \
  --channel_id YOUR_CHANNEL_ID \
  --read_api_key YOUR_READ_API_KEY  # omit if channel is public \
  --model data_collection/ky038_cough_model/models/ky038_cough_model.joblib \
  --field 4 \
  --window_s 30 \
  --sample_period_s 15 \
  --min_samples 2 \
  --prob_threshold 0.6
```

Notes:
- Free channels accept one update every ~15 seconds, so detection latency is coarse in ThingSpeak mode. For near‑real‑time (sub‑second), prefer the serial mode.
- The `--field` flag selects which ThingSpeak field carries the RMS values (e.g., `--field 4` if you followed the ESP32 sketch defaults).
- Use `--min_samples` to control how many numeric readings must be present before the model fires (defaults to 2 to match a 30 s window with 15 s sampling). Increase if you want wider windows; decrease (to 1) if you prefer predictions even with a single point in the buffer.
- The script builds a window from the last N ThingSpeak samples (`window_s / sample_period_s`).
- We publish to `field1`; ensure no other device overwrites it.

## 5. Notes & Next Steps

- Features approximate the envelope stats a KY-038 would yield; consider adding band-pass filtering or spectral features (MFCCs) for improved accuracy.
- Calibrate `prob_threshold` using validation ROC curve to balance sensitivity vs specificity.
- To reduce false positives, increase window length (e.g., 1.5s in serial mode, or >30 s in ThingSpeak mode) or thresholds used internally for peak counts.
- Add logging or buffering to capture raw windows for later retraining.

### ESP32 includes in VS Code
If you see `cannot open source file "WiFi.h"` in VS Code, select an ESP32 board in the Arduino extension or install the ESP32 Arduino core. The code compiles on ESP32; the squiggles are just IntelliSense not using ESP32 headers.

## 6. Troubleshooting

| Issue | Tip |
|-------|-----|
| Model very biased | Raise/Lower `--label_threshold` or use class_weight adjustments. |
| Serial freezes | Confirm correct port and baud; ensure no other process holds the port. |
| Low accuracy | Increase `--max_files` to use more training samples; try `--model_type logreg`. |

## 7. License

This directory's scripts are provided as-is; dataset licensing per COUGHVID terms.

## 8. Sharing/Publishing to GitHub

Large audio and dataset files should not be committed. Provide code + lightweight docs, and distribute the trained model artifact separately. Common strategies:

**Option A – GitHub Release asset (recommended)**  
Train locally, attach `ky038_cough_model.joblib` to a Release. Users download it:
```bash
MODEL_URL="https://github.com/<owner>/<repo>/releases/download/v1.0.0/ky038_cough_model.joblib"
DEST="data_collection/ky038_cough_model/models/ky038_cough_model.joblib"
mkdir -p "$(dirname "$DEST")" && curl -L "$MODEL_URL" -o "$DEST"
```
Pros: clean history, versioned; avoids Git LFS limits.

**Option B – Git LFS**  
Track only model artifacts:
```bash
git lfs install
git lfs track "*.joblib"
git add .gitattributes
git add data_collection/ky038_cough_model/models/ky038_cough_model.joblib
git commit -m "Add cough model via LFS"
git push origin main
```
Pros: clone includes model; Cons: bandwidth quota.

**Option C – Re-train on first use**  
Ship code only; first run executes training script. Pros: minimal repo; Cons: user needs dataset & time.

The root `.gitignore` intentionally excludes `models/` and large audio (`*.wav`). Remove or adjust ignores if you adopt LFS and want artifacts tracked.

Add a model card (see `MODEL_CARD.md`) detailing data sources, feature extraction, intended use, and limitations.

### Minimum Files to Publish
Include:
1. `preprocess_and_train.py`
2. `fastapi_server.py` (if using buffered API)
3. `export_edge_model.py` (for microcontroller deployment)
4. `realtime_infer_serial.py` / `realtime_infer_thingspeak.py`
5. `requirements.txt`
6. `README.md` & `MODEL_CARD.md`

### Versioning the Model
Increment semantic version (e.g., v1.0.0) when:
- Training data changes materially
- Feature set changes
- Algorithm class changes (e.g., logreg → gradient boost)

Record metrics (precision/recall, ROC AUC) in Release notes for transparency.

### Security & Privacy
Do not publish raw cough audio if it might contain PII or background speech. Share only derived features when needed.

### Integrity Checks
Provide an optional SHA256 for the model file:
```bash
sha256sum ky038_cough_model.joblib > ky038_cough_model.joblib.sha256
```
Users verify:
```bash
sha256sum -c ky038_cough_model.joblib.sha256
```

---
For deployment in containers, mount the model path or download at container start rather than baking large artifacts into the image.
