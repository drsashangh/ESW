#!/usr/bin/env python3
import argparse
import json
import os
from glob import glob
from typing import List, Tuple

import numpy as np
import soundfile as sf
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis

DATA_DIR_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "COUGHVID-dataset", "public_dataset")
MODEL_DIR_DEFAULT = os.path.join(os.path.dirname(__file__), "models")


def window_rms(x: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    if len(x) < frame_len:
        # pad to at least one frame
        pad = frame_len - len(x)
        x = np.pad(x, (0, pad), mode='constant')
    n_frames = 1 + (len(x) - frame_len) // hop_len
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, frame_len),
        strides=(x.strides[0] * hop_len, x.strides[0])
    )
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    return rms


def extract_clip_features(audio: np.ndarray, sr: int, win_s: float = 0.1, hop_s: float = 0.1) -> Tuple[np.ndarray, dict]:
    win = max(1, int(sr * win_s))
    hop = max(1, int(sr * hop_s))
    # rectified amplitude envelope approximation via RMS over 100 ms windows
    rms = window_rms(audio, win, hop)

    # robust normalization within clip (scale invariance to sensor gain)
    med = np.median(rms)
    mad = np.median(np.abs(rms - med)) + 1e-9
    rms_norm = (rms - med) / mad

    # summary stats that a KY-038-like envelope would reveal
    feats = {
        'len': len(rms_norm),
        'mean': float(np.mean(rms_norm)),
        'std': float(np.std(rms_norm)),
        'max': float(np.max(rms_norm)),
        'median': float(np.median(rms_norm)),
        'p90': float(np.percentile(rms_norm, 90)),
        'p95': float(np.percentile(rms_norm, 95)),
        'p99': float(np.percentile(rms_norm, 99)),
        'skew': float(skew(rms_norm)),
        'kurt': float(kurtosis(rms_norm)),
    }

    # threshold-based activity metrics
    thr1 = 2.0  # ~2 MAD above median
    thr2 = 4.0
    frac_thr1 = float(np.mean(rms_norm > thr1))
    frac_thr2 = float(np.mean(rms_norm > thr2))

    # peak counts (coughs are short high peaks in envelope)
    peaks, _ = find_peaks(rms_norm, height=thr1, distance=max(1, int(0.3 / hop_s)))
    peak_count = int(len(peaks))

    feats.update({
        'frac_thr1': frac_thr1,
        'frac_thr2': frac_thr2,
        'peak_count': peak_count,
        'max_over_median': float((np.max(rms) + 1e-9) / (np.median(rms) + 1e-9)),
    })

    # vectorize in stable order
    order = [
        'len','mean','std','max','median','p90','p95','p99','skew','kurt',
        'frac_thr1','frac_thr2','peak_count','max_over_median'
    ]
    vec = np.array([feats[k] for k in order], dtype=np.float32)
    # guard against NaNs/Infs from statistics on near-constant arrays
    vec = np.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6)
    return vec, feats


def load_label(json_path: str, threshold: float) -> int:
    with open(json_path, 'r') as f:
        meta = json.load(f)
    val = meta.get('cough_detected')
    if val is None:
        raise ValueError(f"Missing 'cough_detected' in {json_path}")
    try:
        score = float(val)
    except Exception:
        # sometimes it might be '0'/'1' strings; coerce
        score = float(str(val))
    return int(score >= threshold)


def make_dataset(data_dir: str, label_threshold: float, max_files: int = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    wavs = sorted(glob(os.path.join(data_dir, '*.wav')))
    if max_files is not None:
        wavs = wavs[:max_files]
    X, y, ids = [], [], []
    for wav_path in wavs:
        base = os.path.splitext(os.path.basename(wav_path))[0]
        json_path = os.path.join(data_dir, base + '.json')
        if not os.path.exists(json_path):
            continue
        try:
            audio, sr = sf.read(wav_path)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            # optional: high-pass filter to remove very low freq rumble - not strictly necessary for envelope RMS
            xvec, _ = extract_clip_features(audio, sr, win_s=0.1, hop_s=0.1)
            label = load_label(json_path, label_threshold)
            X.append(xvec)
            y.append(label)
            ids.append(base)
        except Exception as e:
            print(f"Skip {wav_path}: {e}")
    if not X:
        raise RuntimeError(f"No data found in {data_dir}")
    return np.vstack(X), np.array(y, dtype=np.int64), ids


def build_model(model_type: str = 'gb') -> Pipeline:
    if model_type == 'logreg':
        clf = LogisticRegression(max_iter=500, class_weight='balanced')
    else:
        clf = GradientBoostingClassifier()
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
        ('scaler', StandardScaler()),
        ('clf', clf)
    ])
    return pipe


def main():
    ap = argparse.ArgumentParser(description='Train KY-038-like cough detector using COUGHVID labels')
    ap.add_argument('--data_dir', default=DATA_DIR_DEFAULT, help='Path to COUGHVID public_dataset folder')
    ap.add_argument('--label_threshold', type=float, default=0.5, help='Threshold on cough_detected to mark positive')
    ap.add_argument('--model_out', default=os.path.join(MODEL_DIR_DEFAULT, 'ky038_cough_model.joblib'))
    ap.add_argument('--model_type', choices=['gb','logreg'], default='gb')
    ap.add_argument('--test_size', type=float, default=0.2)
    ap.add_argument('--val_size', type=float, default=0.1)
    ap.add_argument('--max_files', type=int, default=None, help='Optionally cap number of files for quick runs')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    print(f"Loading dataset from {args.data_dir} ...")
    X, y, ids = make_dataset(args.data_dir, args.label_threshold, args.max_files)
    print(f"Samples: {len(y)}, Positives: {int(y.sum())} ({100*y.mean():.1f}%)")

    # split train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=args.test_size + args.val_size, stratify=y, random_state=42)
    rel_val = args.val_size / (args.test_size + args.val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1-rel_val, stratify=y_temp, random_state=42)

    model = build_model(args.model_type)
    model.fit(X_train, y_train)

    def eval_split(name, Xs, ys):
        pred = model.predict(Xs)
        prob = model.predict_proba(Xs)[:,1] if hasattr(model, 'predict_proba') else None
        print(f"\n{name} report:")
        print(classification_report(ys, pred, digits=4))
        if prob is not None:
            try:
                auc = roc_auc_score(ys, prob)
                print(f"ROC AUC: {auc:.4f}")
            except Exception:
                pass

    eval_split('Validation', X_val, y_val)
    eval_split('Test', X_test, y_test)

    dump(model, args.model_out)
    print(f"Saved model to {args.model_out}")

if __name__ == '__main__':
    main()
