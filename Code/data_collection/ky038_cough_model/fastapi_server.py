#!/usr/bin/env python3
"""FastAPI server for KY-038 cough detection.

Provides buffered audio ingestion so the full waveform (or a sensible window)
is received before feature extraction, avoiding fragmentation that can distort
RMS / envelope statistics when sending tiny chunks individually.

Endpoints:
  GET  /health              -> basic liveness
  POST /infer               -> multipart/form-data file upload (wav/pcm)
                               params: threshold (optional)
  POST /infer/raw           -> raw bytes (audio/wav or application/octet-stream) body
  WS   /ws/stream           -> (optional) streaming binary frames; send text 'END' to
                               trigger inference on current buffer.

Model: expects a scikit-learn pipeline saved with joblib producing
probabilities via predict_proba for class 1 (cough). It consumes the same
feature vector used in training (see preprocess_and_train.py).

Feature extraction mirrors training: 100ms RMS windows, MAD normalization,
summary stats, threshold proportions, peak count, max/median ratio.

Example usage (after starting server):
  curl -F "file=@example.wav" "http://localhost:8000/infer?threshold=0.6"

To run:
  uvicorn fastapi_server:app --host 0.0.0.0 --port 8000

Notes for MCU / KY-038:
  The simplest robust pattern is buffering N seconds of raw ADC samples on
  the device, converting to a PCM WAV (or just 16-bit little-endian bytes
  plus a minimal WAV header), and POSTing that single file every time you
  want a decision. Avoid sending per-sample HTTP requests.
"""

from __future__ import annotations

import io
import os
import json
import asyncio
from typing import List, Tuple, Optional
import time

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from pydantic import BaseModel


# ---------------- Configuration ---------------- #
MODEL_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), 'models', 'ky038_cough_model.joblib')
WIN_S = 0.1  # window size (seconds) for RMS
HOP_S = 0.1  # hop size (seconds) for RMS


def window_rms(x: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    """Vectorized sliding RMS (like in training)."""
    if len(x) < frame_len:
        pad = frame_len - len(x)
        x = np.pad(x, (0, pad), mode='constant')
    if len(x) < frame_len:
        return np.array([0.0], dtype=np.float32)
    n_frames = 1 + (len(x) - frame_len) // hop_len
    if n_frames <= 0:
        return np.array([0.0], dtype=np.float32)
    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, frame_len),
        strides=(x.strides[0] * hop_len, x.strides[0])
    )
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)
    return rms.astype(np.float32)


FEATURE_ORDER = [
    'len','mean','std','max','median','p90','p95','p99','skew','kurt',
    'frac_thr1','frac_thr2','peak_count','max_over_median'
]


def extract_clip_features(audio: np.ndarray, sr: int, win_s: float = WIN_S, hop_s: float = HOP_S) -> Tuple[np.ndarray, dict]:
    """Replicates training feature extraction for consistency."""
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    win = max(1, int(sr * win_s))
    hop = max(1, int(sr * hop_s))
    rms = window_rms(audio, win, hop)
    med = np.median(rms)
    mad = np.median(np.abs(rms - med)) + 1e-9
    rms_norm = (rms - med) / mad

    feats = {
        'len': len(rms_norm),
        'mean': float(np.mean(rms_norm)),
        'std': float(np.std(rms_norm)),
        'max': float(np.max(rms_norm)),
        'median': float(np.median(rms_norm)),
        'p90': float(np.percentile(rms_norm, 90)),
        'p95': float(np.percentile(rms_norm, 95)),
        'p99': float(np.percentile(rms_norm, 99)),
        'skew': float(skew(rms_norm)),
        'kurt': float(kurtosis(rms_norm)),
    }
    thr1 = 2.0
    thr2 = 4.0
    feats['frac_thr1'] = float(np.mean(rms_norm > thr1))
    feats['frac_thr2'] = float(np.mean(rms_norm > thr2))
    peaks, _ = find_peaks(rms_norm, height=thr1, distance=max(1, int(0.3 / hop_s)))
    feats['peak_count'] = int(len(peaks))
    feats['max_over_median'] = float((np.max(rms) + 1e-9) / (np.median(rms) + 1e-9))

    vec = np.array([feats[k] for k in FEATURE_ORDER], dtype=np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6)
    return vec, feats


def extract_rms_features(rms: np.ndarray, hop_s: float = HOP_S) -> Tuple[np.ndarray, dict]:
    """Feature extraction when caller provides an RMS time series (e.g., ThingSpeak RMS).

    Mirrors extract_clip_features statistics but skips windowed RMS step.
    """
    if rms.ndim != 1:
        rms = rms.reshape(-1)
    # MAD normalization as in training
    med = np.median(rms)
    mad = np.median(np.abs(rms - med)) + 1e-9
    rms_norm = (rms - med) / mad

    feats = {
        'len': len(rms_norm),
        'mean': float(np.mean(rms_norm)),
        'std': float(np.std(rms_norm)),
        'max': float(np.max(rms_norm)),
        'median': float(np.median(rms_norm)),
        'p90': float(np.percentile(rms_norm, 90)),
        'p95': float(np.percentile(rms_norm, 95)),
        'p99': float(np.percentile(rms_norm, 99)),
        'skew': float(skew(rms_norm)),
        'kurt': float(kurtosis(rms_norm)),
    }
    thr1 = 2.0
    thr2 = 4.0
    feats['frac_thr1'] = float(np.mean(rms_norm > thr1))
    feats['frac_thr2'] = float(np.mean(rms_norm > thr2))
    peaks, _ = find_peaks(rms_norm, height=thr1, distance=max(1, int(0.3 / max(hop_s, 1e-3))))
    feats['peak_count'] = int(len(peaks))
    feats['max_over_median'] = float((np.max(rms) + 1e-9) / (np.median(rms) + 1e-9))

    vec = np.array([feats[k] for k in FEATURE_ORDER], dtype=np.float32)
    vec = np.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6)
    return vec, feats


class ModelWrapper:
    def __init__(self, path: str):
        self.path = path
        self.model = None

    def load(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Model not found: {self.path}")
        self.model = load(self.path)

    def predict_prob(self, feat_vec: np.ndarray) -> float:
        if self.model is None:
            self.load()
        fv = feat_vec.reshape(1, -1)
        if hasattr(self.model, 'predict_proba'):
            return float(self.model.predict_proba(fv)[0, 1])
        # fallback using decision_function sigmoid
        from math import exp
        d = float(self.model.decision_function(fv))
        return 1.0 / (1.0 + exp(-d))


model_wrapper = ModelWrapper(MODEL_PATH_DEFAULT)
try:
    model_wrapper.load()
except Exception:
    # Delay error until first inference; allows container to start without model
    pass


app = FastAPI(title="KY-038 Cough Detection API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store last inference per device for dashboard polling
LAST_RESULTS: dict[str, dict] = {}


@app.get('/health')
async def health():
    return {"status": "ok"}


def _infer_from_audio(audio: np.ndarray, sr: int, threshold: float) -> dict:
    feat_vec, feat_map = extract_clip_features(audio, sr)
    prob = model_wrapper.predict_prob(feat_vec)
    decision = prob >= threshold
    return {
        'probability': prob,
        'decision': 'COUGH' if decision else 'no_cough',
        'threshold': threshold,
        'features': feat_map,
        'feature_vector_order': FEATURE_ORDER,
    }


async def _read_upload_to_bytes(upload: UploadFile) -> bytes:
    # Read fully; FastAPI / Starlette buffers efficiently in temp files if large.
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail='Empty file upload')
    return data


def _decode_wav_bytes(data: bytes) -> Tuple[np.ndarray, int]:
    try:
        bio = io.BytesIO(data)
        audio, sr = sf.read(bio)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        return audio, sr
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Failed to decode audio: {e}') from e


@app.post('/infer')
async def infer_file(threshold: float = 0.6, device_id: str = "default", file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail='Missing filename')
    data = await _read_upload_to_bytes(file)
    audio, sr = _decode_wav_bytes(data)
    result = _infer_from_audio(audio, sr, threshold)
    result_out = {**result, 'device_id': device_id, 'timestamp': int(time.time())}
    LAST_RESULTS[device_id] = result_out
    return result_out


@app.post('/infer/raw')
async def infer_raw(request: Request, threshold: float = 0.6, device_id: str = "default"):
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail='Empty body')
    audio, sr = _decode_wav_bytes(data)
    result = _infer_from_audio(audio, sr, threshold)
    result_out = {**result, 'device_id': device_id, 'timestamp': int(time.time())}
    LAST_RESULTS[device_id] = result_out
    return result_out


@app.get('/latest')
async def latest(device_id: str = "default"):
    """Return the last inference result for a device, if available."""
    if device_id not in LAST_RESULTS:
        raise HTTPException(status_code=404, detail='No result yet for this device')
    return LAST_RESULTS[device_id]


class RmsRequest(BaseModel):
    values: list[float]
    hop_s: float | None = None


@app.post('/infer/rms')
async def infer_rms(req: RmsRequest, threshold: float = 0.6):
    """Infer from an RMS series (e.g., KY-038 RMS posted to ThingSpeak).

    Body JSON: {"values": [...], "hop_s": 15.0}
    """
    vals = np.asarray(req.values, dtype=np.float32)
    if vals.size < 2:
        raise HTTPException(status_code=400, detail='Need at least 2 RMS samples')
    hop_s = float(req.hop_s) if req.hop_s is not None else HOP_S
    feat_vec, feat_map = extract_rms_features(vals, hop_s=hop_s)
    prob = model_wrapper.predict_prob(feat_vec)
    decision = prob >= threshold
    return {
        'probability': prob,
        'decision': 'COUGH' if decision else 'no_cough',
        'threshold': threshold,
        'features': feat_map,
        'feature_vector_order': FEATURE_ORDER,
        'hop_s': hop_s,
    }


# -------------- WebSocket Streaming (Optional) -------------- #
class StreamSession:
    def __init__(self):
        self.buffer = io.BytesIO()
        self.started = False

    def append(self, data: bytes):
        self.buffer.write(data)
        self.started = True

    def clear(self):
        self.buffer = io.BytesIO()
        self.started = False

    def infer(self, threshold: float) -> dict:
        if not self.started:
            raise ValueError('No audio received')
        data = self.buffer.getvalue()
        audio, sr = _decode_wav_bytes(data)
        return _infer_from_audio(audio, sr, threshold)


@app.websocket('/ws/stream')
async def ws_stream(ws: WebSocket):
    await ws.accept()
    session = StreamSession()
    threshold = 0.6
    try:
        while True:
            msg = await ws.receive()
            if 'bytes' in msg and msg['bytes'] is not None:
                session.append(msg['bytes'])
                await ws.send_json({'event': 'chunk_received', 'bytes': len(msg['bytes'])})
            elif 'text' in msg and msg['text'] is not None:
                text = msg['text'].strip()
                if text.upper().startswith('THRESH='):
                    try:
                        threshold = float(text.split('=',1)[1])
                        await ws.send_json({'event': 'threshold_set', 'threshold': threshold})
                    except ValueError:
                        await ws.send_json({'event': 'error', 'detail': 'Invalid threshold value'})
                elif text.upper() == 'END':
                    try:
                        result = session.infer(threshold)
                        await ws.send_json({'event': 'inference', **result})
                    except Exception as e:
                        await ws.send_json({'event': 'error', 'detail': str(e)})
                    finally:
                        session.clear()
                elif text.upper() == 'RESET':
                    session.clear()
                    await ws.send_json({'event': 'reset'})
                else:
                    await ws.send_json({'event': 'unknown_command', 'detail': text})
            else:
                await asyncio.sleep(0)  # yield control
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({'event': 'error', 'detail': str(e)})
        except Exception:
            pass


if __name__ == '__main__':
    # Allow running directly: python fastapi_server.py
    import uvicorn
    uvicorn.run('fastapi_server:app', host='0.0.0.0', port=8000, reload=False)
