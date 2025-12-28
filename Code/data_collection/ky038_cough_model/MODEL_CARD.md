# Model Card: KY-038 Cough Detection

- Model: Scikit-learn pipeline (e.g., GradientBoostingClassifier or LogisticRegression)
- Target: Probability of cough event in a short audio window sampled from a KY-038 (analog mic) surrogate
- Input: Short WAV clips (mono, 16-bit PCM), featureized as envelope/RMS statistics; optional peak counts and percentiles
- Output: Probability p(cough); thresholded for decision

## Intended Use
- Assistive detection of cough segments in embedded or edge settings. Not a medical device. Not for diagnosis or triage.

## Data
- Source: COUGHVID public dataset (refer to original license and terms).
- We do not redistribute the dataset in this repo.
- Preprocessing: Resample/mono if needed, compute 100 ms RMS windows; derive statistics (mean/std/max/percentiles, skew/kurtosis, peak counts).

## Training & Evaluation
- Split: Train/val on public subset; test set held out by filename.
- Metrics (example): Report precision/recall and ROC AUC at chosen threshold. Calibrate threshold to desired sensitivity/specificity.
- Class imbalance: Consider class_weight or threshold adjustments.

## Limitations & Risks
- Background speech/noise may trigger false positives.
- Microphone and room acoustics matter; KY-038 variability requires on-site calibration.
- Not robust to severe domain shifts (e.g., different sampling rates, clipping).

## Deployment Notes
- Edge export: `export_edge_model.py` can convert a logistic-regression variant to a C header for MCU inference.
- FastAPI: `fastapi_server.py` provides a buffered upload endpoint for WAV clips.
- Model artifact: Prefer publishing via GitHub Release; avoid committing large binaries to git.

## Versioning
- Semantic version: bump when data/feature set/algorithm changes.
- Provide SHA256 for model files in Releases to enable integrity checks.
