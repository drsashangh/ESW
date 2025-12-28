#!/usr/bin/env python3
"""Hyperparameter tuning (randomized search) for SBP/DBP regressors using patient-wise Groups.

Performs RandomizedSearchCV with GroupKFold on the training patients (from provided train-list files).
Fits best model on full train set and evaluates on the test set.

Usage:
  python3 hyperparam_tune.py --features features.csv --train-list train_subjects.txt --test-list test_subjects.txt --outdir ./models_tuned

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


def prepare(Xdf, stats=None):
    drop_cols = ['patient', 'segment'] if 'patient' in Xdf.columns else []
    y_sbp = Xdf['SBP'].values
    y_dbp = Xdf['DBP'].values
    X = Xdf.drop(columns=['SBP', 'DBP'] + drop_cols)
    X = X.select_dtypes(include=[np.number])
    if stats is None:
        stats = X.mean()
    X = X.fillna(stats)
    return X.values, y_sbp, y_dbp, X.columns.tolist(), stats


def eval_and_save(model, X_test, y_test, outdir, prefix):
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
        raise RuntimeError('Train or test split is empty. Check subject lists provided.')

    X_tr, y_sbp_tr, y_dbp_tr, feat_names, stats = prepare(df_train)
    X_te, y_sbp_te, y_dbp_te, _, _ = prepare(df_test, stats)

    groups = df_train['patient'].values

    os.makedirs(args.outdir, exist_ok=True)

    # choose estimator
    if XGB_AVAILABLE and args.prefer_xgb:
        estimator = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=args.n_jobs)
        param_dist = {
            'n_estimators': [50, 100, 200, 400],
            'max_depth': [3, 5, 6, 8],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.5, 0.8, 1.0],
            'reg_alpha': [0, 0.1, 1.0],
        }
    else:
        estimator = RandomForestRegressor(n_jobs=args.n_jobs)
        param_dist = {
            'n_estimators': [100, 200, 400],
            'max_depth': [6, 10, 15, None],
            'min_samples_split': [2, 4, 8],
            'min_samples_leaf': [1, 2, 4]
        }

    # Randomized search
    cv = GroupKFold(n_splits=3)
    rs = RandomizedSearchCV(estimator, param_distributions=param_dist, n_iter=args.n_iter, cv=cv, scoring='neg_mean_absolute_error', verbose=2, n_jobs=args.n_jobs, random_state=42)

    print('Running randomized search on SBP...')
    rs.fit(X_tr, y_sbp_tr, groups=groups)
    print('Best params (SBP):', rs.best_params_)
    best_sbp = rs.best_estimator_

    # evaluate on test
    sbp_metrics, sbp_pred = eval_and_save(best_sbp, X_te, y_sbp_te, args.outdir, 'sbp')

    print('Running randomized search on DBP...')
    rs2 = RandomizedSearchCV(estimator, param_distributions=param_dist, n_iter=args.n_iter, cv=cv, scoring='neg_mean_absolute_error', verbose=2, n_jobs=args.n_jobs, random_state=43)
    rs2.fit(X_tr, y_dbp_tr, groups=groups)
    print('Best params (DBP):', rs2.best_params_)
    best_dbp = rs2.best_estimator_
    dbp_metrics, dbp_pred = eval_and_save(best_dbp, X_te, y_dbp_te, args.outdir, 'dbp')

    # save predictions and metrics
    out_df = pd.DataFrame({
        'patient': df_test['patient'] if 'patient' in df_test.columns else None,
        'SBP_true': y_sbp_te,
        'SBP_pred': sbp_pred,
        'DBP_true': y_dbp_te,
        'DBP_pred': dbp_pred,
    })
    out_df.to_csv(os.path.join(args.outdir, 'tuned_predictions_patientwise.csv'), index=False)

    with open(os.path.join(args.outdir, 'tuned_metrics.json'), 'w') as f:
        json.dump({'SBP': sbp_metrics, 'DBP': dbp_metrics, 'SBP_best_params': getattr(rs, 'best_params_', None), 'DBP_best_params': getattr(rs2, 'best_params_', None)}, f, indent=2)

    # save models
    try:
        dump(best_sbp, os.path.join(args.outdir, 'tuned_model_sbp.joblib'))
        dump(best_dbp, os.path.join(args.outdir, 'tuned_model_dbp.joblib'))
        print('Saved tuned models to', args.outdir)
    except Exception:
        print('Could not save models with joblib.dump')

    print('Done. Metrics written to', os.path.join(args.outdir, 'tuned_metrics.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--train-list', required=True, nargs='+')
    parser.add_argument('--test-list', required=True, nargs='+')
    parser.add_argument('--outdir', default='models_tuned')
    parser.add_argument('--n-iter', type=int, default=20)
    parser.add_argument('--n-jobs', type=int, default=4)
    parser.add_argument('--prefer-xgb', action='store_true', help='Prefer XGBoost when available')
    args = parser.parse_args()
    # normalize args
    args.n_iter = max(5, args.n_iter)
    args.n_jobs = args.n_jobs
    main(args)
