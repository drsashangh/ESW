#!/usr/bin/env python3
"""
Fever Spike Prediction Model
============================
Predicts whether temperature will spike above 38°C in the next 15 minutes
using heart rate, activity, and temperature trends.

Usage:
    python fever_model/train_fever_model.py

Output:
    fever_model/fever_spike_model.joblib
"""

import os
import pandas as pd
import numpy as np
from datetime import timedelta
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score

# ==================== Configuration ==================== #
DATA_PATH = "predictor-conditions/notebooks/data/processed/fever_training_data.csv"
MODEL_OUTPUT = "fever_model/fever_spike_model.joblib"
FEVER_THRESHOLD = 37.2  # Celsius - elevated temp threshold (adjusted for dataset)
LOOKBACK_MINUTES = 5    # How much history to use for features
FORECAST_MINUTES = 15   # How far ahead to predict
SAMPLE_INTERVAL_SEC = 20  # Approximate sampling interval

# ==================== Feature Engineering ==================== #

def safe_trend(values: np.ndarray, time_axis: np.ndarray) -> float:
    """Compute slope of values over time."""
    if len(values) < 2 or np.std(values) < 1e-6:
        return 0.0
    try:
        return float(np.polyfit(time_axis, values, 1)[0])
    except:
        return 0.0


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    """Compute correlation with NaN handling."""
    if len(a) < 2:
        return 0.0
    corr = a.corr(b)
    return float(corr) if not np.isnan(corr) else 0.0


def build_features(window_df: pd.DataFrame) -> dict:
    """Build features from a time window of sensor data."""
    features = {}
    
    # Time axis for trend calculations (in seconds)
    time_axis = (window_df.index - window_df.index[0]).total_seconds().values
    
    # Temperature features
    temp = window_df['temperature']
    features['temp_mean'] = temp.mean()
    features['temp_std'] = temp.std() if len(temp) > 1 else 0
    features['temp_min'] = temp.min()
    features['temp_max'] = temp.max()
    features['temp_last'] = temp.iloc[-1]
    features['temp_trend'] = safe_trend(temp.values, time_axis)
    features['temp_range'] = temp.max() - temp.min()
    
    # Heart rate features
    hr = window_df['heart_rate']
    features['hr_mean'] = hr.mean()
    features['hr_std'] = hr.std() if len(hr) > 1 else 0
    features['hr_min'] = hr.min()
    features['hr_max'] = hr.max()
    features['hr_last'] = hr.iloc[-1]
    features['hr_trend'] = safe_trend(hr.values, time_axis)
    
    # Activity features
    activity = window_df['activity']
    features['activity_mean'] = activity.mean()
    features['activity_std'] = activity.std() if len(activity) > 1 else 0
    features['activity_trend'] = safe_trend(activity.values, time_axis)
    
    # Sound level features (may indicate coughing/distress)
    sound = window_df['sound_level']
    features['sound_mean'] = sound.mean()
    features['sound_std'] = sound.std() if len(sound) > 1 else 0
    
    # Cross-feature correlations
    features['temp_hr_corr'] = safe_corr(temp, hr)
    features['temp_activity_corr'] = safe_corr(temp, activity)
    
    # Derived indicators
    features['hr_elevated'] = 1 if features['hr_mean'] > 90 else 0
    features['temp_elevated'] = 1 if features['temp_last'] > 37.5 else 0
    
    return features


def create_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create features and targets using sliding windows."""
    print("Creating time-series features and targets...")
    
    lookback = timedelta(minutes=LOOKBACK_MINUTES)
    forecast = timedelta(minutes=FORECAST_MINUTES)
    slide = timedelta(seconds=60)  # Slide by 1 minute
    
    feature_list = []
    target_list = []
    
    # Iterate through time
    start_time = df.index.min() + lookback
    end_time = df.index.max() - forecast
    
    current = start_time
    while current <= end_time:
        # Get lookback window
        window_start = current - lookback
        window_df = df.loc[window_start:current]
        
        # Get forecast window
        forecast_df = df.loc[current:current + forecast]
        
        if len(window_df) >= 3 and len(forecast_df) >= 1:
            # Build features
            features = build_features(window_df)
            
            # Create target: will temperature exceed threshold in next 15 min?
            max_future_temp = forecast_df['temperature'].max()
            target = 1 if max_future_temp > FEVER_THRESHOLD else 0
            
            feature_list.append(features)
            target_list.append(target)
        
        current += slide
    
    X = pd.DataFrame(feature_list).fillna(0)
    y = pd.Series(target_list)
    
    print(f"Created {len(X)} samples")
    print(f"Positive class (fever spike): {y.sum()} ({100*y.mean():.1f}%)")
    
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Train and evaluate the fever prediction model."""
    print("\n--- Training Fever Prediction Model ---")
    
    if len(X) < 20 or y.nunique() < 2:
        raise ValueError("Not enough data or only one class present")
    
    # Stratified split
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    
    # Build pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(
            random_state=42,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=5,
            subsample=0.8
        ))
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("\n--- Model Performance ---")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
    
    if y_test.nunique() > 1:
        print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
    
    # Feature importance
    print("\n--- Top Features ---")
    feature_names = X.columns.tolist()
    importances = pipeline.named_steps['classifier'].feature_importances_
    top_features = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:10]
    for name, imp in top_features:
        print(f"  {name}: {imp:.4f}")
    
    return pipeline


def main():
    # Load data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, parse_dates=['created_at'], index_col='created_at')
    
    # Clean data
    df = df[['temperature', 'heart_rate', 'activity', 'sound_level']].dropna()
    
    # Ensure numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    
    print(f"Loaded {len(df)} records")
    print(f"Temperature range: {df['temperature'].min():.1f} - {df['temperature'].max():.1f}°C")
    print(f"Records above fever threshold ({FEVER_THRESHOLD}°C): {(df['temperature'] > FEVER_THRESHOLD).sum()}")
    
    # Create dataset
    X, y = create_dataset(df)
    
    # Train model
    model = train_model(X, y)
    
    # Save model
    os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT)
    print(f"\n✅ Model saved to {MODEL_OUTPUT}")
    
    # Save feature names for inference
    feature_names_path = MODEL_OUTPUT.replace('.joblib', '_features.txt')
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(X.columns.tolist()))
    print(f"✅ Feature names saved to {feature_names_path}")


if __name__ == "__main__":
    main()
