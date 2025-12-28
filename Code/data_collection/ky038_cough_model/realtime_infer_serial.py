#!/usr/bin/env python3
import argparse
import re
import sys
import time
from collections import deque

import numpy as np
from joblib import load
import serial

# parse lines like: "Sound Level: 1234"
LINE_RE = re.compile(r"Sound\s*Level\s*:\s*(\d+)")


def compute_window_features(values: np.ndarray) -> np.ndarray:
    # mimic training features when only a short sliding window is available
    x = values.astype(np.float32)
    if x.size < 5:
        x = np.pad(x, (0, 5 - x.size))
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)) + 1e-9)
    xn = (x - med) / mad

    feats = []
    feats.append(len(xn))
    feats.append(float(np.mean(xn)))
    feats.append(float(np.std(xn)))
    feats.append(float(np.max(xn)))
    feats.append(float(np.median(xn)))
    feats.append(float(np.percentile(xn, 90)))
    feats.append(float(np.percentile(xn, 95)))
    feats.append(float(np.percentile(xn, 99)))

    # simple skew/kurt approximations for short windows can be noisy
    from scipy.stats import skew, kurtosis
    feats.append(float(skew(xn)))
    feats.append(float(kurtosis(xn)))

    feats.append(float(np.mean(xn > 2.0)))
    feats.append(float(np.mean(xn > 4.0)))

    # peak proxy: count local maxima above 2.0
    c = 0
    for i in range(1, len(xn) - 1):
        if xn[i] > 2.0 and xn[i] > xn[i - 1] and xn[i] > xn[i + 1]:
            c += 1
    feats.append(float(c))

    feats.append(float((np.max(x) + 1e-9) / (np.median(x) + 1e-9)))

    vec = np.array(feats, dtype=np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6)
    return vec


def main():
    ap = argparse.ArgumentParser(description='Real-time KY-038 cough detection over serial')
    ap.add_argument('--port', required=True, help='Serial port (e.g., /dev/ttyUSB0 or COM3)')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--model', default='models/ky038_cough_model.joblib')
    ap.add_argument('--window_s', type=float, default=1.0, help='Sliding window in seconds')
    ap.add_argument('--sample_period_s', type=float, default=0.1, help='Period of analogRead prints on the MCU (e.g., 0.1s)')
    ap.add_argument('--prob_threshold', type=float, default=0.5, help='Decision threshold on model probability')
    args = ap.parse_args()

    try:
        model = load(args.model)
    except Exception as e:
        print(f"Failed to load model at {args.model}: {e}")
        sys.exit(1)

    ser = serial.Serial(args.port, args.baud, timeout=1)
    print(f"Opened {ser.port} @ {ser.baudrate}")

    maxlen = max(1, int(args.window_s / args.sample_period_s))
    buf = deque(maxlen=maxlen)

    try:
        while True:
            line = ser.readline().decode(errors='ignore').strip()
            if not line:
                continue
            m = LINE_RE.search(line)
            if not m:
                continue
            val = int(m.group(1))
            buf.append(val)

            if len(buf) == maxlen:
                feat = compute_window_features(np.array(list(buf)))
                try:
                    if hasattr(model, 'predict_proba'):
                        prob = float(model.predict_proba(feat.reshape(1, -1))[:, 1])
                    else:
                        # fallback using decision function sigmoid
                        from math import exp
                        d = float(model.decision_function(feat.reshape(1, -1)))
                        prob = 1.0 / (1.0 + exp(-d))
                    is_cough = prob >= args.prob_threshold
                    print(f"prob={prob:.3f}\t{('COUGH' if is_cough else 'no_cough')}")
                except Exception as e:
                    print(f"Inference error: {e}")
            # small sleep to avoid busy loop if data rate is lower than expected
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        ser.close()


if __name__ == '__main__':
    main()
