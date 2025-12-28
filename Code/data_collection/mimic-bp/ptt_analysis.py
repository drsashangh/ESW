"""Compute Pulse Transit Time (PTT) per-segment and analyze correlation with SBP/DBP.

This script expects synchronized ECG and PPG numpy arrays per patient in the same
layout used by the extractor (e.g., `ppg/p000123_ppg.npy`, `ecg/p000123_ecg.npy`,
and `labels/p000123_labels.npy`). Each waveform is shape (30, N) for 30 segments.

Outputs:
  - ptt_results.csv: per-segment PTT_mean/PTT_std and SBP/DBP
  - ptt_stats.json: correlation statistics per patient and overall

Usage:
  python3 ptt_analysis.py --dbpath /path/to/mimic-bp --outdir ./ptt_results --fs 125

If you plan to collect new data from Arduino, see the NOTES below for sampling and sync.
"""

import os
import argparse
import numpy as np
import pandas as pd
import json

from scipy.signal import butter, filtfilt, find_peaks


def bandpass(x, fs, low, high, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, x)


def detect_rpeaks(ecg, fs):
    ecg_f = bandpass(ecg, fs, 5.0, 40.0, order=2)
    env = np.abs(ecg_f)
    win = max(1, int(0.150 * fs))
    ma = np.convolve(env, np.ones(win)/win, mode='same')
    th = np.mean(ma) + 0.5 * np.std(ma)
    peaks, _ = find_peaks(ma, height=th, distance=int(0.3*fs))
    return peaks


def detect_ppg_foots(ppg, fs):
    ppg_f = bandpass(ppg, fs, 0.5, 8.0, order=2)
    peaks, _ = find_peaks(ppg_f, distance=int(0.3*fs))
    foots = []
    for p in peaks:
        start = max(0, p - int(0.4*fs))
        if start >= p:
            foots.append(p)
            continue
        idx = np.argmin(ppg_f[start:p+1])
        foots.append(start + idx)
    return np.array(peaks), np.array(foots)


def compute_ptt_for_segment(ecg_seg, ppg_seg, fs):
    try:
        rpeaks = detect_rpeaks(ecg_seg, fs)
        peaks, foots = detect_ppg_foots(ppg_seg, fs)
    except Exception:
        return None, None

    ptt_list = []
    for r in rpeaks:
        later_foots = foots[foots > r]
        if later_foots.size == 0:
            continue
        ptt = (later_foots[0] - r) / fs
        ptt_list.append(ptt)

    if len(ptt_list) == 0:
        return None, None
    return float(np.mean(ptt_list)), float(np.std(ptt_list))


def main(dbpath, outdir, fs=125):
    os.makedirs(outdir, exist_ok=True)

    # find patients by labels
    label_paths = []
    for root in (dbpath, os.path.join(dbpath, 'labels')):
        if os.path.isdir(root):
            for p in os.listdir(root):
                if p.endswith('_labels.npy'):
                    label_paths.append(os.path.join(root, p))

    patients = [os.path.basename(p).split('_labels.npy')[0] for p in label_paths]
    patients = sorted(list(set(patients)))

    rows = []
    per_patient_stats = {}

    for patient in patients:
        labels_fn = None
        for cand in (os.path.join(dbpath, 'labels', f'{patient}_labels.npy'), os.path.join(dbpath, f'{patient}_labels.npy')):
            if os.path.isfile(cand):
                labels_fn = cand
                break
        if labels_fn is None:
            continue
        labels = np.load(labels_fn)

        # load ecg and ppg
        ecg_fn = None
        ppg_fn = None
        for cand in (os.path.join(dbpath, 'ecg', f'{patient}_ecg.npy'), os.path.join(dbpath, f'{patient}_ecg.npy')):
            if os.path.isfile(cand):
                ecg_fn = cand
                break
        for cand in (os.path.join(dbpath, 'ppg', f'{patient}_ppg.npy'), os.path.join(dbpath, f'{patient}_ppg.npy')):
            if os.path.isfile(cand):
                ppg_fn = cand
                break

        if ecg_fn is None or ppg_fn is None:
            # skip patients without synchronized signals
            continue

        ecg = np.load(ecg_fn)
        ppg = np.load(ppg_fn)

        ptt_vals = []
        sbp_vals = []
        dbp_vals = []

        num_segments = labels.shape[0]
        for idx in range(num_segments):
            ecg_seg = ecg[idx]
            ppg_seg = ppg[idx]
            sbp, dbp = labels[idx]
            ptt_mean, ptt_std = compute_ptt_for_segment(ecg_seg, ppg_seg, fs)
            rows.append({'patient': patient, 'segment': int(idx), 'PTT_mean': ptt_mean, 'PTT_std': ptt_std, 'SBP': float(sbp), 'DBP': float(dbp)})
            if ptt_mean is not None:
                ptt_vals.append(ptt_mean)
                sbp_vals.append(sbp)
                dbp_vals.append(dbp)

        # per-patient correlation if enough
        stats = {}
        if len(ptt_vals) >= 5:
            try:
                from scipy.stats import pearsonr
                r_sbp, _ = pearsonr(ptt_vals, sbp_vals)
                r_dbp, _ = pearsonr(ptt_vals, dbp_vals)
            except Exception:
                r_sbp = float(np.corrcoef(ptt_vals, sbp_vals)[0,1])
                r_dbp = float(np.corrcoef(ptt_vals, dbp_vals)[0,1])
            stats['r_ptt_sbp'] = float(r_sbp)
            stats['r_ptt_dbp'] = float(r_dbp)
        else:
            stats['r_ptt_sbp'] = None
            stats['r_ptt_dbp'] = None

        per_patient_stats[patient] = stats

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, 'ptt_results.csv'), index=False)
    with open(os.path.join(outdir, 'ptt_stats.json'), 'w') as f:
        json.dump(per_patient_stats, f, indent=2)

    # overall correlation across all segments where PTT_mean available
    df_valid = df.dropna(subset=['PTT_mean'])
    out = {}
    if not df_valid.empty:
        try:
            from scipy.stats import pearsonr
            r_sbp, _ = pearsonr(df_valid['PTT_mean'].values, df_valid['SBP'].values)
            r_dbp, _ = pearsonr(df_valid['PTT_mean'].values, df_valid['DBP'].values)
        except Exception:
            r_sbp = float(np.corrcoef(df_valid['PTT_mean'].values, df_valid['SBP'].values)[0,1])
            r_dbp = float(np.corrcoef(df_valid['PTT_mean'].values, df_valid['DBP'].values)[0,1])
        out['overall_r_ptt_sbp'] = float(r_sbp)
        out['overall_r_ptt_dbp'] = float(r_dbp)
    else:
        out['overall_r_ptt_sbp'] = None
        out['overall_r_ptt_dbp'] = None

    with open(os.path.join(outdir, 'ptt_overall_stats.json'), 'w') as f:
        json.dump(out, f, indent=2)

    print('Wrote PTT results to', outdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbpath', required=True, help='Path to mimic-bp folder')
    parser.add_argument('--outdir', default='./ptt_results', help='Output directory')
    parser.add_argument('--fs', type=int, default=125, help='sampling frequency')
    args = parser.parse_args()
    main(args.dbpath, args.outdir, fs=args.fs)

"""
NOTES on data collection for reliable PTT:
- Sample ECG and PPG on the same microcontroller/ADC so samples are synchronized.
- Use sampling rate >=125 Hz; 250-500 Hz is preferred for higher timing precision.
- Ensure timestamps or sample counters don't drift between channels.
- Save per-patient files as numpy arrays with shape (segments, samples_per_segment), and labels as (segments, 2) for SBP/DBP.
"""
