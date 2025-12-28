#!/usr/bin/env python3
"""
End-to-end pipeline for non-invasive BP modeling with the MIMIC-like HW-827 dataset.

Steps:
    1) Feature extraction from ECG/PPG arrays -> features.csv
  2) Hyperparameter tuning with patient-wise splits -> tuned models + predictions
  3) (Optional) Ensemble with RF + tuned model -> ensemble metrics
  4) Per-patient calibration on tuned predictions -> metrics before/after

Usage:
  python run_pipeline.py \
    --dbpath data_collection/mimic-bp \
    --outdir data_collection/mimic-bp/models_with_tuning \
    --train-list data_collection/mimic-bp/train_subjects.txt \
    --test-list data_collection/mimic-bp/test_subjects.txt \
    --val-list data_collection/mimic-bp/val_subjects.txt

Notes:
  - This script calls into the existing scripts in this folder to preserve behavior.
  - The feature extractor tolerates missing channels and computes robust summary features.
  - Calibration simulates cuff-based personalization using the first k samples per patient.
"""
import os
import json
import argparse
from types import SimpleNamespace


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def step_features(dbpath: str, out_csv: str, fs: int = 125, list_files=None):
    # Import the extractor as a module and call its main()
    import extract_features_hw827 as extract
    extract.main(dbpath, out_csv, fs=fs, subject_list_files=list_files)


def step_tune(features_csv: str, train_list, test_list, outdir: str, n_iter: int = 20, n_jobs: int = 4, prefer_xgb: bool = True):
    import hyperparam_tune as tune
    args = SimpleNamespace(
        features=features_csv,
        train_list=train_list,
        test_list=test_list,
        outdir=outdir,
        n_iter=max(5, int(n_iter)),
        n_jobs=int(n_jobs),
        prefer_xgb=bool(prefer_xgb),
    )
    tune.main(args)


def step_ensemble(features_csv: str, train_list, test_list, tuned_dir: str, outdir: str):
    import ensemble_eval as ens
    args = SimpleNamespace(
        features=features_csv,
        train_list=train_list,
        test_list=test_list,
        tuned_dir=tuned_dir,
        outdir=outdir,
    )
    ens.main(args)


def step_calibrate_tuned(tuned_outdir: str, calib_outdir: str, k: int = 3, mode: str = 'offset'):
    """Run per-patient calibration using tuned_predictions_patientwise.csv."""
    import calibrate_predictions as calib
    preds_csv = os.path.join(tuned_outdir, 'tuned_predictions_patientwise.csv')
    if not os.path.isfile(preds_csv):
        raise FileNotFoundError(f"Missing tuned predictions at {preds_csv}")
    calib.main(preds_csv, calib_outdir, k=int(k), mode=mode)


def parse_args():
    ap = argparse.ArgumentParser(description='Run full BP modeling pipeline (features -> tuned models -> calibration).')
    ap.add_argument('--dbpath', required=True, help='Dataset folder with *_ecg.npy, *_ppg.npy, *_labels.npy, etc.')
    ap.add_argument('--outdir', required=True, help='Output directory for models and reports')
    ap.add_argument('--features-csv', default=None, help='If provided, skip extraction and reuse this features CSV')
    ap.add_argument('--fs', type=int, default=125, help='Sampling frequency for feature extraction')
    ap.add_argument('--train-list', required=True, nargs='+', help='One or more train subject list files')
    ap.add_argument('--test-list', required=True, nargs='+', help='One or more test subject list files')
    ap.add_argument('--val-list', nargs='*', default=None, help='Optional val subject list files (unused by tuning)')
    ap.add_argument('--no-ensemble', action='store_true', help='Skip the ensemble step and only run tuning')
    ap.add_argument('--k', type=int, default=3, help='Calibration: number of per-patient samples (cuff readings)')
    ap.add_argument('--calib-mode', choices=['offset', 'linear'], default='offset', help='Calibration mode')
    ap.add_argument('--n-iter', type=int, default=20, help='Randomized search iterations for tuning')
    ap.add_argument('--n-jobs', type=int, default=4, help='Parallel jobs for tuning and RF')
    ap.add_argument('--prefer-xgb', action='store_true', help='Prefer XGBoost if available')
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    # 1) Features
    features_csv = args.features_csv or os.path.join(args.outdir, 'features_hw827.csv')
    if args.features_csv is None:
        print('Extracting features ->', features_csv)
        # Extract features for both train and test subjects to enable evaluation
        subject_files = list(args.train_list) + list(args.test_list)
        step_features(args.dbpath, features_csv, fs=args.fs, list_files=subject_files)
    else:
        print('Reusing features from', features_csv)

    # 2) Hyperparameter tuning
    tuned_dir = os.path.join(args.outdir, 'models_tuned')
    ensure_dir(tuned_dir)
    print('Tuning models ->', tuned_dir)
    step_tune(features_csv, args.train_list, args.test_list, tuned_dir, n_iter=args.n_iter, n_jobs=args.n_jobs, prefer_xgb=args.prefer_xgb)

    # 3) Optional ensemble
    if not args.no_ensemble:
        ens_out = os.path.join(args.outdir, 'models_ensemble')
        ensure_dir(ens_out)
        print('Ensembling tuned + RF ->', ens_out)
        step_ensemble(features_csv, args.train_list, args.test_list, tuned_dir=tuned_dir, outdir=ens_out)

    # 4) Per-patient calibration on tuned predictions
    calib_out = os.path.join(args.outdir, 'calibration')
    ensure_dir(calib_out)
    print('Calibrating tuned predictions ->', calib_out)
    step_calibrate_tuned(tuned_dir, calib_out, k=args.k, mode=args.calib_mode)

    print('\nPipeline complete. Artifacts saved under:', args.outdir)


if __name__ == '__main__':
    main()
