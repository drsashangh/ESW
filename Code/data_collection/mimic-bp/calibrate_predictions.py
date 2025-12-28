"""Apply per-subject calibration to baseline predictions and evaluate improvements.

This script expects the file `baseline_predictions_patientwise.csv` produced by
`train_baseline.py` to exist and contain columns: patient, SBP_true, SBP_pred, DBP_true, DBP_pred.

It simulates calibration by using the first `k` samples per patient as cuff readings
to fit either an offset-only (shift) or linear correction (scale+shift), then evaluates
on the remaining samples.

Usage:
    python calibrate_predictions.py --preds models_patientwise/baseline_predictions_patientwise.csv --k 3 --mode offset

Outputs:
    - calibrated_predictions.csv
    - calibration_metrics.json
"""
import argparse
import os
import pandas as pd
import numpy as np
import json

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_metrics(y_true, y_pred):
    err = y_pred - y_true
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pct_within_5 = float(np.mean(np.abs(err) <= 5) * 100.0)
    pct_within_10 = float(np.mean(np.abs(err) <= 10) * 100.0)
    pct_within_15 = float(np.mean(np.abs(err) <= 15) * 100.0)
    return {'MAE': float(mae), 'RMSE': float(rmse), 'pct_within_5': pct_within_5, 'pct_within_10': pct_within_10, 'pct_within_15': pct_within_15}


def calibrate_patient(df_pat, k=3, mode='offset'):
    # df_pat is rows for one patient in the test set in chronological order
    n = len(df_pat)
    if n <= k:
        # not enough samples to calibrate -> return predictions unchanged for evaluation
        return df_pat.copy(), None

    calib = df_pat.iloc[:k]
    test = df_pat.iloc[k:]

    # SBP calibration
    Xc = calib[['SBP_pred']].values.reshape(-1, 1)
    yc = calib['SBP_true'].values
    if mode == 'offset':
        b = (yc - Xc.ravel()).mean()
        def apply_sbp(x):
            return x + b
    else:
        model = LinearRegression().fit(Xc, yc)
        def apply_sbp(x):
            return model.predict(np.array(x).reshape(-1, 1))

    # DBP calibration
    Xc_db = calib[['DBP_pred']].values.reshape(-1, 1)
    yc_db = calib['DBP_true'].values
    if mode == 'offset':
        bdb = (yc_db - Xc_db.ravel()).mean()
        def apply_dbp(x):
            return x + bdb
    else:
        model_db = LinearRegression().fit(Xc_db, yc_db)
        def apply_dbp(x):
            return model_db.predict(np.array(x).reshape(-1, 1))

    # apply to test rows
    test2 = test.copy()
    test2['SBP_pred_calib'] = apply_sbp(test2['SBP_pred'].values)
    test2['DBP_pred_calib'] = apply_dbp(test2['DBP_pred'].values)

    # return calibration-applied test rows and calibration params
    params = {'mode': mode}
    if mode == 'offset':
        params.update({'SBP_offset': float(b), 'DBP_offset': float(bdb)})
    else:
        params.update({'SBP_coef': float(model.coef_[0]), 'SBP_intercept': float(model.intercept_), 'DBP_coef': float(model_db.coef_[0]), 'DBP_intercept': float(model_db.intercept_)})

    return test2, params


def main(preds_csv, outdir, k=3, mode='offset'):
    df = pd.read_csv(preds_csv)
    if 'patient' not in df.columns:
        raise ValueError('predictions CSV must contain a patient column')

    os.makedirs(outdir, exist_ok=True)

    all_tests = []
    calib_params = {}

    for patient, g in df.groupby('patient'):
        g_sorted = g.reset_index(drop=True)
        test2, params = calibrate_patient(g_sorted, k=k, mode=mode)
        if test2 is None:
            continue
        all_tests.append(test2)
        calib_params[patient] = params

    if not all_tests:
        print('No patients had enough samples for calibration with k=', k)
        return

    df_calib = pd.concat(all_tests, axis=0).reset_index(drop=True)

    # compute metrics before calibration (on same test subset)
    # need to align rows: use original test rows (after first k per patient)
    orig_tests = []
    for patient, g in df.groupby('patient'):
        g_sorted = g.reset_index(drop=True)
        if len(g_sorted) <= k:
            continue
        orig_tests.append(g_sorted.iloc[k:])
    df_orig = pd.concat(orig_tests, axis=0).reset_index(drop=True)

    sbp_before = compute_metrics(df_orig['SBP_true'].values, df_orig['SBP_pred'].values)
    dbp_before = compute_metrics(df_orig['DBP_true'].values, df_orig['DBP_pred'].values)

    sbp_after = compute_metrics(df_calib['SBP_true'].values, df_calib['SBP_pred_calib'].values)
    dbp_after = compute_metrics(df_calib['DBP_true'].values, df_calib['DBP_pred_calib'].values)

    out_metrics = {'before': {'SBP': sbp_before, 'DBP': dbp_before}, 'after': {'SBP': sbp_after, 'DBP': dbp_after}, 'k': k, 'mode': mode}

    df_calib.to_csv(os.path.join(outdir, 'calibrated_predictions.csv'), index=False)
    with open(os.path.join(outdir, 'calibration_metrics.json'), 'w') as f:
        json.dump(out_metrics, f, indent=2)
    with open(os.path.join(outdir, 'calibration_params.json'), 'w') as f:
        json.dump(calib_params, f, indent=2)

    print('Calibration completed. Metrics saved to', outdir)
    print('SBP before:', sbp_before)
    print('SBP after :', sbp_after)
    print('DBP before:', dbp_before)
    print('DBP after :', dbp_after)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', required=True, help='baseline_predictions_patientwise.csv')
    parser.add_argument('--outdir', default='./calib_results', help='where to save calibrated predictions and metrics')
    parser.add_argument('--k', type=int, default=3, help='number of calibration samples per patient')
    parser.add_argument('--mode', choices=['offset', 'linear'], default='offset', help='calibration model')
    args = parser.parse_args()
    main(args.preds, args.outdir, k=args.k, mode=args.mode)
