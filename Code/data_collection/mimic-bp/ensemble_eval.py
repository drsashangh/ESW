#!/usr/bin/env python3
"""Simple ensemble: average tuned XGBoost (from hyperparam_tune) and a RandomForest trained on train set.

Produces ensemble predictions on test patients and saves metrics and predictions.
"""
import argparse
import os
import json
import pandas as pd
import numpy as np
from joblib import load, dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def prepare(df, stats=None):
    drop_cols = ['patient', 'segment'] if 'patient' in df.columns else []
    y_sbp = df['SBP'].values
    y_dbp = df['DBP'].values
    X = df.drop(columns=['SBP', 'DBP'] + drop_cols)
    X = X.select_dtypes(include=[np.number])
    if stats is None:
        stats = X.mean()
    X = X.fillna(stats)
    return X.values, y_sbp, y_dbp, X.columns.tolist(), stats


def compute_metrics(y, yhat):
    mae = mean_absolute_error(y, yhat)
    rmse = np.sqrt(mean_squared_error(y, yhat))
    return {'MAE': float(mae), 'RMSE': float(rmse)}


def main(args):
    df = pd.read_csv(args.features)
    # read ids
    def read_ids(fn):
        with open(fn, 'r') as f:
            txt = f.read().strip()
        if txt.startswith('[') and txt.endswith(']'):
            import ast
            try:
                lst = ast.literal_eval(txt)
                return set([str(x).strip() for x in lst])
            except Exception:
                txt = txt.strip('[]')
                parts = [p.strip().strip("'\"") for p in txt.split(',') if p.strip()]
                return set(parts)
        else:
            return set([l.strip() for l in txt.splitlines() if l.strip()])

    train_ids = set()
    test_ids = set()
    for fn in args.train_list:
        train_ids.update(read_ids(fn))
    for fn in args.test_list:
        test_ids.update(read_ids(fn))

    df_train = df[df['patient'].isin(train_ids)].reset_index(drop=True)
    df_test = df[df['patient'].isin(test_ids)].reset_index(drop=True)

    X_tr, y_sbp_tr, y_dbp_tr, feat_names, stats = prepare(df_train)
    X_te, y_sbp_te, y_dbp_te, _, _ = prepare(df_test, stats)

    os.makedirs(args.outdir, exist_ok=True)

    # load tuned xgb models if available
    xgb_sbp = None
    xgb_dbp = None
    try:
        xgb_sbp = load(os.path.join(args.tuned_dir, 'tuned_model_sbp.joblib'))
        xgb_dbp = load(os.path.join(args.tuned_dir, 'tuned_model_dbp.joblib'))
        print('Loaded tuned models from', args.tuned_dir)
    except Exception:
        print('Could not load tuned models from', args.tuned_dir)

    # train RF on train set
    rf_sbp = RandomForestRegressor(n_estimators=400, max_depth=15, n_jobs=-1)
    rf_dbp = RandomForestRegressor(n_estimators=400, max_depth=15, n_jobs=-1)
    print('Training RF on train set...')
    rf_sbp.fit(X_tr, y_sbp_tr)
    rf_dbp.fit(X_tr, y_dbp_tr)

    # predictions
    preds = {}
    if xgb_sbp is not None:
        preds['sbp_xgb'] = xgb_sbp.predict(X_te)
    preds['sbp_rf'] = rf_sbp.predict(X_te)
    if xgb_dbp is not None:
        preds['dbp_xgb'] = xgb_dbp.predict(X_te)
    preds['dbp_rf'] = rf_dbp.predict(X_te)

    # simple average ensemble
    sbp_preds = []
    dbp_preds = []
    if 'sbp_xgb' in preds:
        sbp_preds.append(preds['sbp_xgb'])
    sbp_preds.append(preds['sbp_rf'])
    if 'dbp_xgb' in preds:
        dbp_preds.append(preds['dbp_xgb'])
    dbp_preds.append(preds['dbp_rf'])

    sbp_ens = np.mean(np.vstack(sbp_preds), axis=0)
    dbp_ens = np.mean(np.vstack(dbp_preds), axis=0)

    # compute metrics
    sbp_metrics = compute_metrics(y_sbp_te, sbp_ens)
    dbp_metrics = compute_metrics(y_dbp_te, dbp_ens)

    print('Ensemble SBP metrics:', sbp_metrics)
    print('Ensemble DBP metrics:', dbp_metrics)

    out_df = pd.DataFrame({
        'patient': df_test['patient'] if 'patient' in df_test.columns else None,
        'SBP_true': y_sbp_te,
        'SBP_pred': sbp_ens,
        'DBP_true': y_dbp_te,
        'DBP_pred': dbp_ens,
    })
    out_df.to_csv(os.path.join(args.outdir, 'ensemble_predictions_patientwise.csv'), index=False)

    with open(os.path.join(args.outdir, 'ensemble_metrics.json'), 'w') as f:
        json.dump({'SBP': sbp_metrics, 'DBP': dbp_metrics}, f, indent=2)

    # save RF models
    try:
        dump(rf_sbp, os.path.join(args.outdir, 'rf_model_sbp.joblib'))
        dump(rf_dbp, os.path.join(args.outdir, 'rf_model_dbp.joblib'))
    except Exception:
        pass

    print('Saved ensemble predictions and metrics to', args.outdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--train-list', required=True, nargs='+')
    parser.add_argument('--test-list', required=True, nargs='+')
    parser.add_argument('--tuned-dir', default='data_collection/mimic-bp/models_tuned_x80')
    parser.add_argument('--outdir', default='data_collection/mimic-bp/models_ensemble')
    args = parser.parse_args()
    main(args)
