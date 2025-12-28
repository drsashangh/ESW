#!/usr/bin/env python3
"""Hyperparameter tuning for SBP/DBP regressors excluding SpO2 features.

This mirrors hyperparam_tune.py but removes all columns beginning with 'SpO2_'.

Usage:
  python hyperparam_tune_no_spo2.py --features features_hw827.csv \
    --train-list train_subjects.txt --test-list test_subjects.txt \
    --outdir models_tuned_no_spo2 --prefer-xgb --n-iter 25
"""
import argparse
import json
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


def read_ids(fn):
    with open(fn, 'r') as f:
        txt = f.read().strip()
    if txt.startswith('[') and txt.endswith(']'):
        try:
            import ast
            lst = ast.literal_eval(txt)
            return set([str(x).strip() for x in lst])
        except Exception:
            txt = txt.strip('[]')
            parts = [p.strip().strip("'\"") for p in txt.split(',') if p.strip()]
            return set(parts)
    else:
        ids = [l.strip() for l in txt.splitlines() if l.strip()]
        return set(ids)


def prepare(df, stats=None):
    # drop patient/segment and SpO2 columns
    drop_cols = []
    if 'patient' in df.columns:
        drop_cols.append('patient')
    if 'segment' in df.columns:
        drop_cols.append('segment')
    # remove SpO2 columns
    df = df[[c for c in df.columns if not c.startswith('SpO2_')]]

    y_sbp = df['SBP'].values
    y_dbp = df['DBP'].values
    X = df.drop(columns=['SBP', 'DBP'] + drop_cols)
    X = X.select_dtypes(include=[np.number])
    if stats is None:
        stats = X.mean()
    X = X.fillna(stats)
    return X.values, y_sbp, y_dbp, X.columns.tolist(), stats


def eval_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return {'MAE': float(mae), 'RMSE': float(rmse)}, y_pred


def main(args):
    df = pd.read_csv(args.features)
    # read lists
    train_ids = set()
    test_ids = set()
    for fn in args.train_list:
        train_ids.update(read_ids(fn))
    for fn in args.test_list:
        test_ids.update(read_ids(fn))

    df_train = df[df['patient'].isin(train_ids)].reset_index(drop=True)
    df_test = df[df['patient'].isin(test_ids)].reset_index(drop=True)
    if df_train.empty or df_test.empty:
        raise RuntimeError('Train or test split is empty. Check subject lists.')

    X_tr, y_sbp_tr, y_dbp_tr, feat_names, stats = prepare(df_train)
    X_te, y_sbp_te, y_dbp_te, _, _ = prepare(df_test, stats)
    groups = df_train['patient'].values

    os.makedirs(args.outdir, exist_ok=True)

    # choose estimator + param space
    if XGB_AVAILABLE and args.prefer_xgb:
        estimator = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=args.n_jobs)
        param_dist = {
            'n_estimators': [100, 200, 400],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.5, 0.7, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
        }
    else:
        estimator = RandomForestRegressor(n_jobs=args.n_jobs)
        param_dist = {
            'n_estimators': [100, 200, 400],
            'max_depth': [6, 10, 15, None],
            'min_samples_split': [2, 4, 8],
            'min_samples_leaf': [1, 2, 4]
        }

    cv = GroupKFold(n_splits=3)
    rs_sbp = RandomizedSearchCV(estimator, param_distributions=param_dist, n_iter=args.n_iter,
                                cv=cv, scoring='neg_mean_absolute_error', verbose=2, n_jobs=args.n_jobs, random_state=42)
    print('Randomized search (SBP, no SpO2)...')
    rs_sbp.fit(X_tr, y_sbp_tr, groups=groups)
    best_sbp = rs_sbp.best_estimator_
    sbp_metrics, sbp_pred = eval_metrics(best_sbp, X_te, y_sbp_te)
    print('Best params SBP:', rs_sbp.best_params_)
    print('SBP metrics:', sbp_metrics)

    rs_dbp = RandomizedSearchCV(estimator, param_distributions=param_dist, n_iter=args.n_iter,
                                cv=cv, scoring='neg_mean_absolute_error', verbose=2, n_jobs=args.n_jobs, random_state=43)
    print('Randomized search (DBP, no SpO2)...')
    rs_dbp.fit(X_tr, y_dbp_tr, groups=groups)
    best_dbp = rs_dbp.best_estimator_
    dbp_metrics, dbp_pred = eval_metrics(best_dbp, X_te, y_dbp_te)
    print('Best params DBP:', rs_dbp.best_params_)
    print('DBP metrics:', dbp_metrics)

    # save predictions/metrics
    out_df = pd.DataFrame({
        'patient': df_test['patient'],
        'SBP_true': y_sbp_te,
        'SBP_pred': sbp_pred,
        'DBP_true': y_dbp_te,
        'DBP_pred': dbp_pred,
    })
    out_df.to_csv(os.path.join(args.outdir, 'tuned_predictions_no_spo2_patientwise.csv'), index=False)

    with open(os.path.join(args.outdir, 'tuned_metrics_no_spo2.json'), 'w') as f:
        json.dump({'SBP': sbp_metrics, 'DBP': dbp_metrics,
                   'SBP_best_params': getattr(rs_sbp, 'best_params_', None),
                   'DBP_best_params': getattr(rs_dbp, 'best_params_', None)}, f, indent=2)

    try:
        dump(best_sbp, os.path.join(args.outdir, 'tuned_model_sbp_no_spo2.joblib'))
        dump(best_dbp, os.path.join(args.outdir, 'tuned_model_dbp_no_spo2.joblib'))
        print('Saved tuned models to', args.outdir)
    except Exception:
        print('Could not save tuned models.')

    print('Done.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True)
    p.add_argument('--train-list', required=True, nargs='+')
    p.add_argument('--test-list', required=True, nargs='+')
    p.add_argument('--outdir', default='models_tuned_no_spo2')
    p.add_argument('--n-iter', type=int, default=20)
    p.add_argument('--n-jobs', type=int, default=4)
    p.add_argument('--prefer-xgb', action='store_true')
    a = p.parse_args()
    a.n_iter = max(5, a.n_iter)
    main(a)
