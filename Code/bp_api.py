#!/usr/bin/env python3
"""
FastAPI service for non-invasive BP estimation with buffered ECG/PPG ingestion.

- POST /ingest: Append ECG/PPG samples for a device_id
- GET  /predict: Return SBP/DBP using the latest window (fs * window_sec)
- GET  /status: Buffer status per device
- GET  /health: Healthcheck

Models are loaded from a trained pipeline output directory (default: data_collection/mimic-bp/models_rt).
Feature schema and fill-means are derived from features_hw827.csv in that directory.

Environment variables (optional):
- BP_MODEL_DIR: path to model dir (default data_collection/mimic-bp/models_rt)
- BP_FS: int sampling frequency (default 125)
- BP_WINDOW_SEC: int window seconds (default 30)
- BP_CORS_ORIGINS: comma-separated list (default *)

Run:
  /home/abhijit-suhas/esw/IOMT-eswproj/Zzz/.venv/bin/python -m uvicorn bp_api:app --host 0.0.0.0 --port 8000

"""
import os
import time
import threading
from collections import deque, defaultdict
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import importlib.util
from pathlib import Path

# ----------------------------- Config -----------------------------
MODEL_DIR = os.environ.get('BP_MODEL_DIR', 'data_collection/mimic-bp/models_rt')
FS_DEFAULT = int(os.environ.get('BP_FS', '125'))
WINDOW_SEC_DEFAULT = int(os.environ.get('BP_WINDOW_SEC', '30'))
CORS_ORIGINS = os.environ.get('BP_CORS_ORIGINS', '*')

# ------------------------- Buffer management ----------------------
class DeviceBuffer:
    def __init__(self, fs: int, window_sec: int):
        self.fs = int(fs)
        self.window_sec = int(window_sec)
        self.maxlen = int(self.fs * self.window_sec)
        self.ecg = deque(maxlen=self.maxlen)
        self.ppg = deque(maxlen=self.maxlen)
        self.lock = threading.Lock()

    def extend(self, ecg: Optional[List[float]], ppg: Optional[List[float]]):
        with self.lock:
            if ecg:
                self.ecg.extend(float(x) for x in ecg)
            if ppg:
                self.ppg.extend(float(x) for x in ppg)

    def window_arrays(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self.lock:
            ecg_arr = np.asarray(self.ecg, dtype=np.float32) if len(self.ecg) > 0 else None
            ppg_arr = np.asarray(self.ppg, dtype=np.float32) if len(self.ppg) > 0 else None
        if ecg_arr is not None and ecg_arr.size >= self.maxlen:
            ecg_arr = ecg_arr[-self.maxlen:]
        if ppg_arr is not None and ppg_arr.size >= self.maxlen:
            ppg_arr = ppg_arr[-self.maxlen:]
        return ecg_arr, ppg_arr

    def sizes(self):
        with self.lock:
            return len(self.ecg), len(self.ppg), self.maxlen


class BufferManager:
    def __init__(self, fs: int, window_sec: int):
        self.fs = fs
        self.window_sec = window_sec
        self.buffers: Dict[str, DeviceBuffer] = {}
        self.lock = threading.Lock()

    def get(self, device_id: str) -> DeviceBuffer:
        with self.lock:
            if device_id not in self.buffers:
                self.buffers[device_id] = DeviceBuffer(self.fs, self.window_sec)
            return self.buffers[device_id]

    def status(self):
        with self.lock:
            out = {}
            for did, buf in self.buffers.items():
                e, p, m = buf.sizes()
                out[did] = {
                    'ecg_samples': e,
                    'ppg_samples': p,
                    'maxlen': m,
                    'fs': buf.fs,
                    'window_sec': buf.window_sec,
                }
            return out


# ---------------------------- Models ------------------------------
class IngestPayload(BaseModel):
    device_id: str = Field(..., description='Device identifier (e.g., esp32-01)')
    ecg: Optional[List[float]] = Field(default=None, description='Array of ECG samples (floats)')
    ppg: Optional[List[float]] = Field(default=None, description='Array of PPG samples (floats)')


class PredictResponse(BaseModel):
    device_id: str
    SBP_pred: Optional[float]
    DBP_pred: Optional[float]
    ecg_samples: int
    ppg_samples: int
    maxlen: int
    fs: int
    window_sec: int
    timestamp: int
    detail: Optional[str] = None


# ----------------------------- App --------------------------------
app = FastAPI(title='Non-invasive BP API', version='1.0.0')

origins = [o.strip() for o in CORS_ORIGINS.split(',')] if CORS_ORIGINS else ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins != ['*'] else ['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------- Helper functions ------------------------
def _import_feature_extractor():
    """Dynamically import the feature extractor from data_collection/mimic-bp."""
    feat_path = Path(__file__).parent / 'data_collection' / 'mimic-bp' / 'extract_features_hw827.py'
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature extractor not found at {feat_path}")
    spec = importlib.util.spec_from_file_location('extract_features_hw827', str(feat_path))
    if spec is None or spec.loader is None:
        raise ImportError('Unable to load feature extractor spec')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def load_models_and_schema(model_dir: str):
    """Load tuned SBP/DBP models and derive feature schema and fill means.

    Returns: (model_sbp, model_dbp, feat_names (list), fill_means (Series))
    """
    tuned_dir = Path(model_dir) / 'models_tuned'
    sbp_path = tuned_dir / 'tuned_model_sbp.joblib'
    dbp_path = tuned_dir / 'tuned_model_dbp.joblib'
    feats_csv = Path(model_dir) / 'features_hw827.csv'
    if not sbp_path.exists() or not dbp_path.exists():
        raise FileNotFoundError(f"Missing tuned models under {tuned_dir}")
    if not feats_csv.exists():
        raise FileNotFoundError(f"Missing features CSV at {feats_csv}")

    model_sbp = joblib.load(sbp_path)
    model_dbp = joblib.load(dbp_path)
    df = pd.read_csv(feats_csv)
    drop_cols = [c for c in ['SBP', 'DBP', 'patient', 'segment'] if c in df.columns]
    feat_names = [c for c in df.columns if c not in drop_cols]
    fill_means = df[feat_names].mean(numeric_only=True)
    return model_sbp, model_dbp, feat_names, fill_means


def build_feature_row(ecg: Optional[np.ndarray], ppg: Optional[np.ndarray], fs: int, feat_names: List[str]) -> pd.DataFrame:
    """Compute a single-row DataFrame of features aligned to feat_names.

    Missing features are added with NaN and should be filled by caller using fill_means.
    """
    featmod = _import_feature_extractor()
    # labels (SBP, DBP) are unknown at inference time
    fdict = featmod.segment_features(ecg, ppg, spo2_seg=None, labels_seg=(np.nan, np.nan), fs=fs)
    if fdict is None:
        fdict = {}
    row = pd.DataFrame([fdict])
    # Ensure all required columns exist
    for col in feat_names:
        if col not in row.columns:
            row[col] = np.nan
    # Reorder and drop any extras
    row = row[feat_names]
    return row

# Load models and schema at startup
try:
    MODEL_SBP, MODEL_DBP, FEAT_NAMES, FILL_MEANS = load_models_and_schema(MODEL_DIR)
except Exception as e:
    MODEL_SBP = MODEL_DBP = FEAT_NAMES = FILL_MEANS = None
    print(f"[WARN] Could not load models from {MODEL_DIR}: {e}")

BUFFERS = BufferManager(fs=FS_DEFAULT, window_sec=WINDOW_SEC_DEFAULT)


@app.get('/health')
def health():
    return {'status': 'ok', 'models_loaded': MODEL_SBP is not None and MODEL_DBP is not None}


@app.get('/status')
def status():
    return {
        'buffers': BUFFERS.status(),
        'model_dir': MODEL_DIR,
        'fs_default': FS_DEFAULT,
        'window_sec_default': WINDOW_SEC_DEFAULT,
    }


@app.post('/ingest')
def ingest(payload: IngestPayload):
    if payload.ecg is None and payload.ppg is None:
        raise HTTPException(status_code=400, detail='Provide at least one of ecg or ppg arrays')
    buf = BUFFERS.get(payload.device_id)
    buf.extend(payload.ecg, payload.ppg)
    e, p, m = buf.sizes()
    return {'device_id': payload.device_id, 'ecg_samples': e, 'ppg_samples': p, 'maxlen': m}


@app.get('/predict', response_model=PredictResponse)
def predict(device_id: str = Query(..., description='Device identifier')):
    if MODEL_SBP is None or MODEL_DBP is None:
        raise HTTPException(status_code=500, detail='Models not loaded. Check model_dir.')

    buf = BUFFERS.get(device_id)
    ecg_arr, ppg_arr = buf.window_arrays()

    # Need at least some data in either channel
    if (ecg_arr is None or ecg_arr.size == 0) and (ppg_arr is None or ppg_arr.size == 0):
        e, p, m = buf.sizes()
        return PredictResponse(
            device_id=device_id,
            SBP_pred=None,
            DBP_pred=None,
            ecg_samples=e,
            ppg_samples=p,
            maxlen=m,
            fs=buf.fs,
            window_sec=buf.window_sec,
            timestamp=int(time.time()),
            detail='Waiting for data',
        )

    # Build feature row matching training schema
    try:
        row = build_feature_row(ecg_arr, ppg_arr, buf.fs, FEAT_NAMES)
        row = row.fillna(FILL_MEANS)
        X = row.values
        sbp = float(MODEL_SBP.predict(X)[0])
        dbp = float(MODEL_DBP.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Inference error: {e}')

    e, p, m = buf.sizes()
    return PredictResponse(
        device_id=device_id,
        SBP_pred=round(sbp, 2),
        DBP_pred=round(dbp, 2),
        ecg_samples=e,
        ppg_samples=p,
        maxlen=m,
        fs=buf.fs,
        window_sec=buf.window_sec,
        timestamp=int(time.time()),
    )


if __name__ == '__main__':
    # Optional: run the API directly
    import uvicorn
    uvicorn.run("bp_api:app", host="0.0.0.0", port=8000, reload=False)
