"""Train a simple XGBoost baseline on the CSV produced by extract_features_hw827.py

Usage:
    python train_baseline.py --features features.csv --outdir ./models

Dependencies: pandas, scikit-learn, xgboost (or fallback to RandomForest if xgboost missing)
"""
import os
import argparse
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from joblib import dump
except Exception:
    dump = None


def prepare_data(df, training_stats=None):
    # drop non-feature cols
    drop_cols = ['patient', 'segment'] if 'patient' in df.columns else []
    y_sbp = df['SBP'].values
    y_dbp = df['DBP'].values
    X = df.drop(columns=['SBP', 'DBP'] + drop_cols)
    # numeric only
    X = X.select_dtypes(include=[np.number])

    # impute NaN with training-set column mean. If training_stats provided, use that.
    if training_stats is None:
        stats = X.mean()
    else:
        stats = training_stats
    X = X.fillna(stats)
    return X.values, y_sbp, y_dbp, X.columns.tolist(), stats


def train_and_evaluate(X_train, y_train, X_test, y_test, model_type='xgb'):
    if model_type == 'xgb' and XGB_AVAILABLE:
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, max_depth=6, learning_rate=0.05)
        # Some xgboost installations do not accept early_stopping_rounds in the sklearn-fit API.
        # Use a plain fit call for compatibility here.
        try:
            model.fit(X_train, y_train)
        except TypeError:
            # fallback: use xgb.train with DMatrix if needed (keep simple here)
            model.fit(X_train, y_train)
    else:
        # light fallback using sklearn
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return model, mae, rmse, y_pred



def compute_metrics(y_true, y_pred):
    err = y_pred - y_true
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    try:
        from scipy.stats import pearsonr
        r, _ = pearsonr(y_true, y_pred)
    except Exception:
        r = np.corrcoef(y_true, y_pred)[0, 1]
    ba_mean = np.mean(err)
    ba_sd = np.std(err)
    pct_within_5 = np.mean(np.abs(err) <= 5) * 100.0
    pct_within_10 = np.mean(np.abs(err) <= 10) * 100.0
    pct_within_15 = np.mean(np.abs(err) <= 15) * 100.0
    return {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'PearsonR': float(r),
        'BA_mean': float(ba_mean),
        'BA_sd': float(ba_sd),
        'pct_within_5': float(pct_within_5),
        'pct_within_10': float(pct_within_10),
        'pct_within_15': float(pct_within_15),
    }


def main(features_csv, outdir, train_list=None, val_list=None, test_list=None, random_state=42):
    df = pd.read_csv(features_csv)

    # If patient columns and lists provided, perform patient-wise splits
    patients = df['patient'].unique() if 'patient' in df.columns else None
    if patients is not None and (train_list or val_list or test_list):
        def read_ids(fn):
            if fn is None:
                return set()
            with open(fn, 'r') as f:
                txt = f.read().strip()
            # Accept either newline-separated IDs or a Python-style list literal
            if txt.startswith('[') and txt.endswith(']'):
                try:
                    import ast
                    lst = ast.literal_eval(txt)
                    return set([str(x).strip() for x in lst])
                except Exception:
                    # fallback to naive splitting
                    txt = txt.strip('[]')
                    parts = [p.strip().strip("'\"") for p in txt.split(',') if p.strip()]
                    return set(parts)
            else:
                ids = [l.strip() for l in txt.splitlines() if l.strip()]
                return set(ids)

        train_ids = set()
        val_ids = set()
        test_ids = set()
        if train_list:
            for fn in train_list:
                train_ids.update(read_ids(fn))
        if val_list:
            for fn in val_list:
                val_ids.update(read_ids(fn))
        if test_list:
            for fn in test_list:
                test_ids.update(read_ids(fn))

        # filter df rows
        df_train = df[df['patient'].isin(train_ids)].reset_index(drop=True)
        df_val = df[df['patient'].isin(val_ids)].reset_index(drop=True)
        df_test = df[df['patient'].isin(test_ids)].reset_index(drop=True)
        # If any set is empty, fall back to random split
        if df_train.empty or df_test.empty:
            print('One of train/test splits is empty using provided lists; falling back to random split')
            df_train, df_test = train_test_split(df, test_size=0.2, random_state=random_state)
            df_val = pd.DataFrame()
    else:
        # random patient-wise split if patient column exists
        if patients is not None:
            tr_p, te_p = train_test_split(patients, test_size=0.2, random_state=random_state)
            df_train = df[df['patient'].isin(tr_p)].reset_index(drop=True)
            df_test = df[df['patient'].isin(te_p)].reset_index(drop=True)
            df_val = pd.DataFrame()
        else:
            # no patient column: simple random split rows
            df_train, df_test = train_test_split(df, test_size=0.2, random_state=random_state)
            df_val = pd.DataFrame()

    os.makedirs(outdir, exist_ok=True)

    # prepare training features and save training stats for imputation
    X_tr, y_sbp_tr, y_dbp_tr, feat_names, train_stats = prepare_data(df_train)
    X_te, y_sbp_te, y_dbp_te, _, _ = prepare_data(df_test, training_stats=train_stats)

    print('Training SBP model...')
    sbp_model, _, _, sbp_pred = train_and_evaluate(X_tr, y_sbp_tr, X_te, y_sbp_te, model_type='xgb')
    sbp_metrics = compute_metrics(y_sbp_te, sbp_pred)
    print('SBP metrics:', sbp_metrics)

    print('Training DBP model...')
    dbp_model, _, _, dbp_pred = train_and_evaluate(X_tr, y_dbp_tr, X_te, y_dbp_te, model_type='xgb')
    dbp_metrics = compute_metrics(y_dbp_te, dbp_pred)
    print('DBP metrics:', dbp_metrics)

    # save predictions and metrics
    results = pd.DataFrame({
        'patient': df_test['patient'] if 'patient' in df_test.columns else None,
        'SBP_true': y_sbp_te,
        'SBP_pred': sbp_pred,
        'DBP_true': y_dbp_te,
        'DBP_pred': dbp_pred,
    })
    results.to_csv(os.path.join(outdir, 'baseline_predictions_patientwise.csv'), index=False)

    import json
    with open(os.path.join(outdir, 'metrics_patientwise.json'), 'w') as f:
        json.dump({'SBP': sbp_metrics, 'DBP': dbp_metrics}, f, indent=2)

    if dump is not None:
        try:
            dump(sbp_model, os.path.join(outdir, 'model_sbp.joblib'))
            dump(dbp_model, os.path.join(outdir, 'model_dbp.joblib'))
            print('Saved models to', outdir)
        except Exception:
            print('joblib.dump failed; models not saved.')
    else:
        print('joblib not available; skipping model save.')

    # save simple results
    results = pd.DataFrame({
        'SBP_true': y_sbp_te,
        'SBP_pred': sbp_pred,
        'DBP_true': y_dbp_te,
        'DBP_pred': dbp_pred,
    })
    results.to_csv(os.path.join(outdir, 'baseline_predictions.csv'), index=False)

    if dump is not None:
        try:
            dump(sbp_model, os.path.join(outdir, 'model_sbp.joblib'))
            dump(dbp_model, os.path.join(outdir, 'model_dbp.joblib'))
            print('Saved models to', outdir)
        except Exception:
            print('joblib.dump failed; models not saved.')
    else:
        print('joblib not available; skipping model save.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True, help='CSV output from extract_features_hw827.py')
    parser.add_argument('--outdir', default='./models', help='Directory to save models and predictions')
    parser.add_argument('--train-list', help='Path to train_subjects.txt (can be given multiple times)', action='append')
    parser.add_argument('--val-list', help='Path to val_subjects.txt (can be given multiple times)', action='append')
    parser.add_argument('--test-list', help='Path to test_subjects.txt (can be given multiple times)', action='append')
    args = parser.parse_args()
    main(args.features, args.outdir, train_list=args.train_list, val_list=args.val_list, test_list=args.test_list)
