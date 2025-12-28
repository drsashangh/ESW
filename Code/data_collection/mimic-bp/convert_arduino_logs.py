#!/usr/bin/env python3
"""
Convert Arduino CSV logs (timestamp, ecg, ppg, spo2) into per-subject .npy segment arrays compatible
with the `mimic-bp` processing pipeline.

Expected input:
 - A folder of CSV files. Filenames should include a subject identifier (e.g. p000123_session1.csv).
 - Each CSV has columns: timestamp, ecg, ppg, spo2 (header optional). Timestamp may be milliseconds or seconds.

Output:
 - For each input file: saved numpy arrays in an output directory mirroring `ecg/`, `ppg/`, `spo2/` as
   per-subject files named like `p000123_ecg.npy`, `p000123_ppg.npy`, `p000123_spo2.npy`.
 - Arrays are 2D: (n_segments, samples_per_segment). By default segments are 30s long and fs=125 Hz.

Usage (example):
  python3 convert_arduino_logs.py --indir raw_logs/ --outdir mimic-bp/ecg_ppg --fs 125 --seg_len 30

Notes:
 - This is a best-effort converter: if timestamps are irregular, it will resample using linear interpolation
   to the target sampling rate `--fs` before segmenting.
 - If your Arduino logged synchronized ECG+PPG on the same board, ensure those columns correspond to the
   same timestamps. If separate files were recorded, you should align them by timestamp before running this
   script.
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
from pathlib import Path


def infer_columns(df):
    cols = [c.lower() for c in df.columns]
    mapping = {}
    if 'timestamp' in cols:
        mapping['t'] = df.columns[cols.index('timestamp')]
    elif 'time' in cols:
        mapping['t'] = df.columns[cols.index('time')]
    # ECG
    for name in ['ecg', 'lead', 'ecg1']:
        if name in cols:
            mapping['ecg'] = df.columns[cols.index(name)]
            break
    # PPG
    for name in ['ppg', 'pleth', 'ir', 'red']:
        if name in cols:
            mapping['ppg'] = df.columns[cols.index(name)]
            break
    # SpO2
    for name in ['spo2', 'sp02', 'o2']:
        if name in cols:
            mapping['spo2'] = df.columns[cols.index(name)]
            break
    return mapping


def resample_signal(t, x, fs_target):
    # t in seconds
    if len(t) < 2:
        return np.array([]), np.array([])
    t0, t1 = t[0], t[-1]
    n_samples = int(np.floor((t1 - t0) * fs_target)) + 1
    if n_samples <= 1:
        return np.array([]), np.array([])
    t_new = np.linspace(t0, t1, n_samples)
    x_new = np.interp(t_new, t, x)
    return t_new, x_new


def segment_signal(x, samples_per_seg):
    n = len(x)
    n_segs = n // samples_per_seg
    if n_segs == 0:
        return np.empty((0, samples_per_seg))
    x = x[:n_segs * samples_per_seg]
    return x.reshape(n_segs, samples_per_seg)


def parse_subject_from_fname(fname):
    # Expect pXXXXX or pXXXXX_ pattern
    m = re.search(r'(p\d{5})', fname)
    if m:
        return m.group(1)
    # fallback: use filename without ext
    return Path(fname).stem


def process_file(path, outdir, fs=125, seg_len=30):
    df = pd.read_csv(path)
    mapping = infer_columns(df)
    if 't' not in mapping:
        # if no timestamp column, try to assume uniform sampling and create timestamps
        t = np.arange(len(df)) / fs
    else:
        t_raw = df[mapping['t']].values.astype(float)
        # guess units: if large (>1e5) treat as ms
        if np.median(np.diff(t_raw)) > 1.0:
            t = (t_raw - t_raw[0]) / 1000.0
        else:
            t = (t_raw - t_raw[0]).astype(float)

    subj = parse_subject_from_fname(str(path.name))
    samples_per_seg = int(seg_len * fs)

    saved = {}

    for key in ['ecg', 'ppg', 'spo2']:
        if key not in mapping:
            continue
        x_raw = df[mapping[key]].values.astype(float)
        t_res, x_res = resample_signal(t, x_raw, fs)
        if len(x_res) == 0:
            continue
        segs = segment_signal(x_res, samples_per_seg)
        if segs.shape[0] == 0:
            continue
        # Save as {subject}_{key}.npy
        out_subdir = Path(outdir) / key
        out_subdir.mkdir(parents=True, exist_ok=True)
        out_fname = out_subdir / f"{subj}_{key}.npy"
        np.save(out_fname, segs)
        saved[key] = str(out_fname)

    return subj, saved


def main():
    p = argparse.ArgumentParser(description="Convert Arduino CSV logs to per-subject .npy segments")
    p.add_argument('--indir', required=True, help='Input directory with CSV log files')
    p.add_argument('--outdir', required=True, help='Output base directory to write ecg/, ppg/, spo2/')
    p.add_argument('--fs', type=int, default=125, help='Target sampling rate (Hz)')
    p.add_argument('--seg_len', type=float, default=30.0, help='Segment length (seconds)')
    p.add_argument('--ext', default='.csv', help='File extension to look for')
    args = p.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    files = sorted(indir.glob(f'*{args.ext}'))
    if len(files) == 0:
        print('No files found in', indir)
        return

    summary = {}
    for f in files:
        subj, saved = process_file(f, outdir, fs=args.fs, seg_len=args.seg_len)
        summary[subj] = saved
        print(f'Processed {f.name} -> subject={subj}, saved: {list(saved.keys())}')

    # write a small summary csv
    import json
    with open(outdir / 'conversion_summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2)
    print('Done. Summary written to', outdir / 'conversion_summary.json')


if __name__ == '__main__':
    main()
