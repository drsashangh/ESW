"""Extract features from ECG + hw-827 PPG per 30s segment.

Saves a CSV with one row per patient-segment containing engineered features and SBP/DBP targets.

Usage:
    python extract_features_hw827.py --dbpath /absolute/path/to/mimic-bp --out features.csv

This script is conservative: it checks for available files per patient (ecg, ppg, labels)
and falls back gracefully when a channel is missing.

Dependencies: numpy, scipy, pandas
Optional (better): tqdm
"""
import os
import argparse
from glob import glob
import numpy as np
import pandas as pd
import ast

try:
    from scipy.signal import butter, filtfilt, find_peaks
except Exception:
    raise ImportError('scipy is required: pip install scipy')

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x):
        return x


def bandpass(x, fs, low=0.5, high=40.0, order=3):
    nyq = 0.5 * fs
    lowb = low / nyq
    highb = high / nyq
    b, a = butter(order, [lowb, highb], btype='band')
    return filtfilt(b, a, x)


def detect_rpeaks(ecg, fs):
    """Simple R-peak detector using bandpass + peak detection.

    Returns sample indices of R-peaks.
    """
    # bandpass to emphasize QRS
    ecg_f = bandpass(ecg, fs, low=5.0, high=40.0, order=2)
    # absolute and squared
    env = np.abs(ecg_f)
    # moving average to smooth
    win = int(0.150 * fs)
    if win < 1:
        win = 1
    ma = np.convolve(env, np.ones(win) / win, mode='same')
    # dynamic threshold
    th = np.mean(ma) + 0.5 * np.std(ma)
    peaks, _ = find_peaks(ma, height=th, distance=int(0.3 * fs))
    return peaks


def detect_ppg_peaks_and_foot(ppg, fs):
    """Return indices of systolic peaks and approximate foot points (onset).

    Foot detection is approximated by finding local minima before peaks.
    """
    ppg_f = bandpass(ppg, fs, low=0.5, high=8.0, order=2)
    # find peaks
    peaks, _ = find_peaks(ppg_f, distance=int(0.3 * fs))
    # approximate foot as the local minimum in window before peak
    foots = []
    for p in peaks:
        start = max(0, p - int(0.4 * fs))
        if start >= p:
            foots.append(p)
            continue
        idx = np.argmin(ppg_f[start:p+1])
        foots.append(start + idx)
    return np.array(peaks), np.array(foots)


def segment_features(ecg_seg, ppg_seg, spo2_seg, labels_seg, fs=125):
    """Compute feature dict for a single 30s segment.

    Inputs:
        ecg_seg: 1D numpy array or None
        ppg_seg: 1D numpy array or None
        labels_seg: (sbp, dbp)
    """
    feat = {}
    sbp, dbp = labels_seg
    feat['SBP'] = float(sbp)
    feat['DBP'] = float(dbp)

    N = None
    if ecg_seg is not None:
        N = len(ecg_seg)
    elif ppg_seg is not None:
        N = len(ppg_seg)

    if N is None:
        return None

    # time axis
    t = np.arange(N) / fs

    # SpO2 features intentionally removed for a cleaner ECG/PPG-only pipeline

    # ECG-derived features
    if ecg_seg is not None:
        try:
            rpeaks = detect_rpeaks(ecg_seg, fs)
        except Exception:
            rpeaks = np.array([])
        rr = np.diff(rpeaks) / fs if len(rpeaks) >= 2 else np.array([])
        hr = 60.0 / rr if rr.size > 0 else np.array([])
        feat['ECG_beats'] = int(len(rpeaks))
        feat['HR_mean'] = float(np.nanmean(hr)) if hr.size > 0 else np.nan
        feat['HR_std'] = float(np.nanstd(hr)) if hr.size > 0 else np.nan
        feat['RR_mean'] = float(np.nanmean(rr)) if rr.size > 0 else np.nan
        feat['RR_std'] = float(np.nanstd(rr)) if rr.size > 0 else np.nan
        feat['HRV_rmssd'] = float(np.sqrt(np.mean(np.diff(rr)**2))) if rr.size > 1 else np.nan
    else:
        feat['ECG_beats'] = np.nan
        feat['HR_mean'] = np.nan
        feat['HR_std'] = np.nan
        feat['RR_mean'] = np.nan
        feat['RR_std'] = np.nan
        feat['HRV_rmssd'] = np.nan

    # PPG-derived features and PTT if both ECG and PPG exist
    if ppg_seg is not None:
        peaks, foots = detect_ppg_peaks_and_foot(ppg_seg, fs)
        feat['PPG_beats'] = int(len(peaks))
        if len(peaks) > 0:
            amps = ppg_seg[peaks] - ppg_seg[foots]
            feat['PPG_amp_mean'] = float(np.nanmean(amps))
            feat['PPG_amp_std'] = float(np.nanstd(amps))
            # width at half amplitude (approx)
            widths = []
            for i, p in enumerate(peaks):
                foot = foots[i]
                half = ppg_seg[foot] + 0.5 * (ppg_seg[p] - ppg_seg[foot])
                # find left and right crossing
                left_idx = foot
                while left_idx < p and ppg_seg[left_idx] < half:
                    left_idx += 1
                right_idx = p
                while right_idx < len(ppg_seg)-1 and ppg_seg[right_idx] > half:
                    right_idx += 1
                widths.append((right_idx - left_idx) / fs)
            feat['PPG_width_mean'] = float(np.nanmean(widths)) if widths else np.nan
        else:
            feat['PPG_amp_mean'] = np.nan
            feat['PPG_amp_std'] = np.nan
            feat['PPG_width_mean'] = np.nan
    else:
        feat['PPG_beats'] = np.nan
        feat['PPG_amp_mean'] = np.nan
        feat['PPG_amp_std'] = np.nan
        feat['PPG_width_mean'] = np.nan

    # PTT (ECG -> PPG foot) if both available
    if ecg_seg is not None and ppg_seg is not None:
        try:
            rpeaks = detect_rpeaks(ecg_seg, fs)
            peaks, foots = detect_ppg_peaks_and_foot(ppg_seg, fs)
            # compute PTTs: for each rpeak find first ppg foot after it
            ptt_list = []
            for r in rpeaks:
                # find foots > r
                later = foots[foots > r]
                if later.size == 0:
                    continue
                ptt = (later[0] - r) / fs
                ptt_list.append(ptt)
            if len(ptt_list) > 0:
                feat['PTT_mean'] = float(np.mean(ptt_list))
                feat['PTT_std'] = float(np.std(ptt_list))
            else:
                feat['PTT_mean'] = np.nan
                feat['PTT_std'] = np.nan
        except Exception:
            feat['PTT_mean'] = np.nan
            feat['PTT_std'] = np.nan
    else:
        feat['PTT_mean'] = np.nan
        feat['PTT_std'] = np.nan

    return feat


def load_patient_arrays(dbpath, patient):
    """Attempt to load arrays for patient. Returns dict with keys: ecg, ppg, labels
    Each waveform expected shape (30, N) where N is samples per segment.
    """
    out = {'ecg': None, 'ppg': None, 'labels': None}
    # possible filenames
    # try top-level file names first
    top_ecg = os.path.join(dbpath, f'{patient}_ecg.npy')
    top_ppg = os.path.join(dbpath, f'{patient}_ppg.npy')
    top_labels = os.path.join(dbpath, f'{patient}_labels.npy')

    if os.path.isfile(top_ecg):
        out['ecg'] = np.load(top_ecg)
    if os.path.isfile(top_ppg):
        out['ppg'] = np.load(top_ppg)
    if os.path.isfile(top_labels):
        out['labels'] = np.load(top_labels)

    # if not found, try common subfolders: ppg/, abp/, resp/, labels/
    if out['ppg'] is None:
        candidate = os.path.join(dbpath, 'ppg', f'{patient}_ppg.npy')
        if os.path.isfile(candidate):
            out['ppg'] = np.load(candidate)
    # abp files live under abp/ (not used as input but helpful)
    if out.get('abp') is None:
        candidate_abp = os.path.join(dbpath, 'abp', f'{patient}_abp.npy')
        if os.path.isfile(candidate_abp):
            out['abp'] = np.load(candidate_abp)
    if out['ecg'] is None:
        candidate = os.path.join(dbpath, 'ecg', f'{patient}_ecg.npy')
        if os.path.isfile(candidate):
            out['ecg'] = np.load(candidate)
    # SpO2 loading removed
    if out['labels'] is None:
        candidate = os.path.join(dbpath, 'labels', f'{patient}_labels.npy')
        if os.path.isfile(candidate):
            out['labels'] = np.load(candidate)

    return out


def _read_subject_ids(files):
    ids = []
    for fn in files:
        with open(fn, 'r') as f:
            txt = f.read().strip()
        # support Python-list literal or newline-separated
        if txt.startswith('[') and txt.endswith(']'):
            try:
                lst = ast.literal_eval(txt)
                ids.extend([str(x).strip() for x in lst])
            except Exception:
                # fallback: split by comma
                txt2 = txt.strip('[]')
                parts = [p.strip().strip('"\'"\'') for p in txt2.split(',') if p.strip()]
                ids.extend(parts)
        else:
            for line in txt.splitlines():
                pid = line.strip()
                if pid:
                    ids.append(pid)
    return ids


def main(dbpath, out_csv, fs=125, subject_list_files=None):
    # get patient ids either from files or by scanning *_labels.npy files
    patients = []
    if subject_list_files:
        patients = _read_subject_ids(subject_list_files)
    else:
        # scan for labels at top-level or under labels/ subfolder
        for path in glob(os.path.join(dbpath, '*_labels.npy')) + glob(os.path.join(dbpath, 'labels', '*_labels.npy')):
            basename = os.path.basename(path)
            pid = basename.split('_labels.npy')[0]
            patients.append(pid)

    patients = sorted(list(set(patients)))

    rows = []
    for patient in tqdm(patients):
        arrs = load_patient_arrays(dbpath, patient)
        labels = arrs['labels']
        if labels is None:
            # nothing to do
            continue
        # number of segments
        num_segments = labels.shape[0]
        for idx in range(num_segments):
            ecg_seg = arrs['ecg'][idx] if (arrs['ecg'] is not None) else None
            ppg_seg = arrs['ppg'][idx] if (arrs['ppg'] is not None) else None
            feat = segment_features(ecg_seg, ppg_seg, spo2_seg=None, labels_seg=labels[idx, :], fs=fs)
            if feat is not None:
                feat['patient'] = patient
                feat['segment'] = int(idx)
                rows.append(feat)

    if not rows:
        print('No features extracted — check dataset path and files')
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f'Wrote features to {out_csv} — {len(df)} rows')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbpath', required=True, help='Path to mimic-bp folder containing *_ecg.npy, *_ppg.npy, *_labels.npy')
    parser.add_argument('--out', required=True, help='CSV output file for features')
    parser.add_argument('--fs', type=int, default=125, help='Sampling frequency (Hz)')
    parser.add_argument('--train-list', help='Optional train_subjects.txt (can be given multiple times)', action='append')
    args = parser.parse_args()
    subject_files = args.train_list if args.train_list else None
    main(args.dbpath, args.out, fs=args.fs, subject_list_files=subject_files)
