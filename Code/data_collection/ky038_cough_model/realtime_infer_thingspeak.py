#!/usr/bin/env python3
import argparse
import time
import requests
import numpy as np
from joblib import load

# ThingSpeak: use the last N points from field1 and run model on a sliding window.


def nan_to_num_vec(x, dtype=np.float64):
    return np.nan_to_num(np.asarray(x, dtype=dtype), nan=0.0, posinf=1e12, neginf=-1e12)


def compute_window_features(values: np.ndarray) -> np.ndarray:
    x = nan_to_num_vec(values, dtype=np.float64)
    if x.size < 5:
        pad_width = 5 - x.size
        pad_mode = 'edge' if x.size > 0 else 'constant'
        x = np.pad(x, (0, pad_width), mode=pad_mode)
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

    from scipy.stats import skew, kurtosis
    feats.append(float(skew(xn)))
    feats.append(float(kurtosis(xn)))

    feats.append(float(np.mean(xn > 2.0)))
    feats.append(float(np.mean(xn > 4.0)))

    c = 0
    for i in range(1, len(xn) - 1):
        if xn[i] > 2.0 and xn[i] > xn[i - 1] and xn[i] > xn[i + 1]:
            c += 1
    feats.append(float(c))

    feats.append(float((np.max(x) + 1e-9) / (np.median(x) + 1e-9)))

    vec = nan_to_num_vec(feats, dtype=np.float64)
    return vec.astype(np.float32)


def fetch_latest_field(channel_id: str, field: int, read_api_key: str, results: int = 10):
    url = f"https://api.thingspeak.com/channels/{channel_id}/fields/{field}.json?results={results}"
    if read_api_key:
        url += f"&api_key={read_api_key}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    feeds = data.get('feeds', [])
    values = []
    key = f'field{field}'
    for item in feeds:
        v = item.get(key)
        try:
            values.append(float(v))
        except Exception:
            pass
    return values


def main():
    ap = argparse.ArgumentParser(description='Real-time cough detection polling from ThingSpeak field1')
    ap.add_argument('--channel_id', required=True, help='ThingSpeak channel ID')
    ap.add_argument('--read_api_key', default='', help='ThingSpeak READ API key (if channel is private)')
    ap.add_argument('--model', default='models/ky038_cough_model.joblib')
    ap.add_argument('--field', type=int, default=1, help='ThingSpeak field number containing RMS values (1-8)')
    ap.add_argument('--window_s', type=float, default=30.0, help='Sliding window seconds (ThingSpeak free tier ~15s/sample)')
    ap.add_argument('--sample_period_s', type=float, default=15.0, help='Sampling period of ThingSpeak updates (seconds)')
    ap.add_argument('--prob_threshold', type=float, default=0.6)
    ap.add_argument('--poll_period_s', type=float, default=7.0, help='How often to poll ThingSpeak (seconds)')
    ap.add_argument('--results', type=int, default=10, help='How many latest samples to fetch per poll')
    ap.add_argument('--min_samples', type=int, default=2, help='Minimum numeric samples required before inference')
    args = ap.parse_args()

    model = load(args.model)
    needed = max(1, int(round(args.window_s / args.sample_period_s)))
    min_required = max(1, args.min_samples)

    print("Polling ThingSpeak... Press Ctrl+C to stop.")
    try:
        while True:
            values = fetch_latest_field(args.channel_id, args.field, args.read_api_key, results=args.results)
            if not values:
                print("No data yet... (ThingSpeak returned no numeric entries)")
                time.sleep(args.poll_period_s)
                continue

            numeric_values = [v for v in values if not np.isnan(v)]
            if len(numeric_values) < min_required:
                preview = numeric_values[-3:] if numeric_values else []
                print(f"Not enough samples yet (have {len(numeric_values)}, need >= {min_required}). Recent values: {preview}")
                time.sleep(args.poll_period_s)
                continue

            # Keep only the last needed samples
            window = np.array(numeric_values[-max(needed, min_required):], dtype=np.float64)
            feat = compute_window_features(window)

            try:
                if hasattr(model, 'predict_proba'):
                    prob = float(model.predict_proba(feat.reshape(1, -1))[0, 1])
                else:
                    from math import exp
                    d = float(model.decision_function(feat.reshape(1, -1)))
                    prob = 1.0 / (1.0 + exp(-d))
                is_cough = prob >= args.prob_threshold
                print(f"prob={prob:.3f}\t{('COUGH' if is_cough else 'no_cough')}\twindow={window.tolist()}")
            except Exception as e:
                print(f"Inference error: {e}")

            time.sleep(args.poll_period_s)
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == '__main__':
    main()
