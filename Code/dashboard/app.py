#!/usr/bin/env python3
"""
Unified Real-Time IoMT Dashboard
================================
A single FastAPI application that provides:
- Real-time fall detection alerts
- Real-time cough detection with probability
- Blood pressure monitoring
- Fever spike prediction (next 15 minutes)
- Live sensor graphs from ThingSpeak

Run with:
    uvicorn dashboard.app:app --host 0.0.0.0 --port 8080

Then open http://localhost:8080 in your browser.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any
from collections import deque

import httpx
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

# ==================== Configuration ==================== #
COUGH_API_URL = os.environ.get("COUGH_API_URL", "http://localhost:8000")
FALL_API_URL = os.environ.get("FALL_API_URL", "http://localhost:8200")
BP_API_URL = os.environ.get("BP_API_URL", "http://localhost:8000")
THINGSPEAK_CHANNEL = os.environ.get("THINGSPEAK_CHANNEL", "3110381")
THINGSPEAK_READ_KEY = os.environ.get("THINGSPEAK_READ_KEY", "")
COUGH_DECISION_THRESHOLD = float(os.environ.get("COUGH_DECISION_THRESHOLD", 0.10))
FEVER_MODEL_PATH = os.environ.get("FEVER_MODEL_PATH", "fever_model/fever_spike_model.joblib")
FEVER_THRESHOLD = float(os.environ.get("FEVER_THRESHOLD", 37.2))

# Device IDs
COUGH_DEVICE_ID = os.environ.get("COUGH_DEVICE_ID", "esp32-ky038-01")
FALL_DEVICE_ID = os.environ.get("FALL_DEVICE_ID", "esp32-mpu6050-01")
BP_DEVICE_ID = os.environ.get("BP_DEVICE_ID", "esp32-01")

# ==================== App Setup ==================== #
app = FastAPI(title="IoMT Real-Time Dashboard", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Fever Model Loading ==================== #
_FEVER_MODEL = None
_FEVER_BUFFER: deque = deque(maxlen=50)  # Store last ~5 minutes of data

def load_fever_model():
    """Load the fever prediction model."""
    global _FEVER_MODEL
    try:
        if os.path.exists(FEVER_MODEL_PATH):
            _FEVER_MODEL = joblib.load(FEVER_MODEL_PATH)
            print(f"✅ Fever model loaded from {FEVER_MODEL_PATH}")
        else:
            print(f"⚠️ Fever model not found at {FEVER_MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading fever model: {e}")

# Load on startup
load_fever_model()

# ==================== Fever Feature Engineering ==================== #

def safe_trend(values: np.ndarray, time_axis: np.ndarray) -> float:
    """Compute slope of values over time."""
    if len(values) < 2 or np.std(values) < 1e-6:
        return 0.0
    try:
        return float(np.polyfit(time_axis, values, 1)[0])
    except:
        return 0.0

def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute correlation with NaN handling."""
    if len(a) < 2:
        return 0.0
    try:
        corr = np.corrcoef(a, b)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0
    except:
        return 0.0

def build_fever_features(data: list[dict]) -> dict:
    """Build features from sensor data buffer for fever prediction."""
    if len(data) < 3:
        return None
    
    df = pd.DataFrame(data)
    
    # Time axis for trends (seconds from start)
    if 'timestamp' in df.columns:
        times = pd.to_datetime(df['timestamp'])
        time_axis = (times - times.iloc[0]).dt.total_seconds().values
    else:
        time_axis = np.arange(len(df)) * 20  # Assume 20s intervals
    
    features = {}
    
    # Temperature features
    temp = df['temperature'].values.astype(float)
    features['temp_mean'] = np.mean(temp)
    features['temp_std'] = np.std(temp) if len(temp) > 1 else 0
    features['temp_min'] = np.min(temp)
    features['temp_max'] = np.max(temp)
    features['temp_last'] = temp[-1]
    features['temp_trend'] = safe_trend(temp, time_axis)
    features['temp_range'] = np.max(temp) - np.min(temp)
    
    # Heart rate features
    hr = df['heart_rate'].values.astype(float)
    features['hr_mean'] = np.mean(hr)
    features['hr_std'] = np.std(hr) if len(hr) > 1 else 0
    features['hr_min'] = np.min(hr)
    features['hr_max'] = np.max(hr)
    features['hr_last'] = hr[-1]
    features['hr_trend'] = safe_trend(hr, time_axis)
    
    # Activity features
    activity = df['activity'].values.astype(float)
    features['activity_mean'] = np.mean(activity)
    features['activity_std'] = np.std(activity) if len(activity) > 1 else 0
    features['activity_trend'] = safe_trend(activity, time_axis)
    
    # Sound features
    sound = df['sound_level'].values.astype(float)
    features['sound_mean'] = np.mean(sound)
    features['sound_std'] = np.std(sound) if len(sound) > 1 else 0
    
    # Cross correlations
    features['temp_hr_corr'] = safe_corr(temp, hr)
    features['temp_activity_corr'] = safe_corr(temp, activity)
    
    # Derived indicators
    features['hr_elevated'] = 1 if features['hr_mean'] > 90 else 0
    features['temp_elevated'] = 1 if features['temp_last'] > 37.5 else 0
    
    return features

# ==================== State Storage ==================== #
_LOCK = threading.Lock()
_STATE: Dict[str, Any] = {
    "cough": {"probability": None, "decision": None, "timestamp": None, "error": None},
    "fall": {"detected": False, "timestamp": None, "elapsed_sec": None, "error": None},
    "bp": {"sbp": None, "dbp": None, "timestamp": None, "error": None},
    "fever": {"probability": None, "risk": None, "temp_current": None, "error": None},
    "thingspeak": {"data": [], "error": None},
}


# ==================== API Endpoints ==================== #

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard HTML."""
    return DASHBOARD_HTML


@app.get("/api/status")
async def get_status():
    """Get current status of all sensors."""
    with _LOCK:
        return {
            "cough": _STATE["cough"].copy(),
            "fall": _STATE["fall"].copy(),
            "bp": _STATE["bp"].copy(),
            "timestamp": time.time(),
        }


@app.get("/api/thingspeak")
async def get_thingspeak_data(results: int = 30):
    """Fetch latest ThingSpeak data."""
    try:
        url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL}/feeds.json?results={results}"
        if THINGSPEAK_READ_KEY:
            url += f"&api_key={THINGSPEAK_READ_KEY}"
        
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cough")
async def get_cough_status():
    """Get latest cough detection status from cough API."""
    try:
        async with httpx.AsyncClient() as client:
            start = time.perf_counter()
            resp = await client.get(
                f"{COUGH_API_URL}/latest",
                params={"device_id": COUGH_DEVICE_ID},
                timeout=5
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            if resp.status_code == 404:
                return {"probability": None, "decision": None, "waiting": True, "latency_ms": latency_ms}
            resp.raise_for_status()
            data = resp.json()
            prob = data.get("probability")
            decision = data.get("decision")
            # Enforce dashboard-level decision threshold (user requested 10%)
            try:
                if prob is not None:
                    prob_f = float(prob)
                    if prob_f > COUGH_DECISION_THRESHOLD:
                        decision = "COUGH"
            except Exception:
                pass
            return {
                "probability": prob,
                "decision": decision,
                "timestamp": data.get("timestamp"),
                "waiting": False,
                "latency_ms": latency_ms,
            }
    except httpx.ConnectError:
        return {"error": "Cough API unavailable", "waiting": True}
    except Exception as e:
        return {"error": str(e), "waiting": True}


@app.get("/api/fall")
async def get_fall_status(alert_window_min: float = 0.5):
    """Get latest fall detection status from fall API.

    alert_window_min defaults to 0.5 (30 seconds) so dashboard alarms clear automatically
    after ~30s unless a new event arrives.
    """
    try:
        async with httpx.AsyncClient() as client:
            start = time.perf_counter()
            resp = await client.get(
                f"{FALL_API_URL}/latest",
                params={"device_id": FALL_DEVICE_ID},
                timeout=5
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            if resp.status_code == 404:
                return {"detected": False, "waiting": True, "latency_ms": latency_ms}
            resp.raise_for_status()
            data = resp.json()
            
            # Calculate elapsed time
            event_ts = data.get("ts")
            elapsed_sec = None
            is_recent = False
            if event_ts:
                elapsed_sec = time.time() - float(event_ts)
                is_recent = elapsed_sec <= (alert_window_min * 60)
            
            return {
                "detected": data.get("fall", True) and is_recent,
                "timestamp": event_ts,
                "elapsed_sec": elapsed_sec,
                "svm": data.get("svm"),
                "waiting": False,
                "latency_ms": latency_ms,
            }
    except httpx.ConnectError:
        return {"error": "Fall API unavailable", "waiting": True}
    except Exception as e:
        return {"error": str(e), "waiting": True}


@app.get("/api/bp")
async def get_bp_status():
    """Get latest BP prediction from BP API."""
    try:
        async with httpx.AsyncClient() as client:
            start = time.perf_counter()
            resp = await client.get(
                f"{BP_API_URL}/predict",
                params={"device_id": BP_DEVICE_ID},
                timeout=8
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            resp.raise_for_status()
            data = resp.json()
            return {
                "sbp": data.get("SBP_pred"),
                "dbp": data.get("DBP_pred"),
                "ecg_samples": data.get("ecg_samples"),
                "ppg_samples": data.get("ppg_samples"),
                "maxlen": data.get("maxlen"),
                "latency_ms": latency_ms,
            }
    except httpx.ConnectError:
        return {"error": "BP API unavailable"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/fever")
async def get_fever_status():
    """Predict fever spike risk in next 15 minutes using ThingSpeak data."""
    global _FEVER_BUFFER
    
    if _FEVER_MODEL is None:
        return {"error": "Fever model not loaded", "probability": None, "risk": None}
    
    try:
        # Fetch recent data from ThingSpeak (last 5 minutes = ~15 samples at 20s intervals)
        url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL}/feeds.json?results=20"
        if THINGSPEAK_READ_KEY:
            url += f"&api_key={THINGSPEAK_READ_KEY}"
        
        async with httpx.AsyncClient() as client:
            start = time.perf_counter()
            resp = await client.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            latency_ms = (time.perf_counter() - start) * 1000.0
        
        feeds = data.get("feeds", [])
        if len(feeds) < 3:
            return {"error": "Not enough data", "waiting": True, "latency_ms": latency_ms}
        
        # Parse feeds into buffer format
        buffer_data = []
        for feed in feeds:
            try:
                entry = {
                    'timestamp': feed.get('created_at'),
                    'heart_rate': float(feed.get('field1', 0) or 0),
                    'activity': float(feed.get('field3', 0) or 0),
                    'sound_level': float(feed.get('field4', 0) or 0),
                    'temperature': float(feed.get('field5', 0) or 0),
                }
                if entry['temperature'] > 0:  # Valid temp reading
                    buffer_data.append(entry)
            except (ValueError, TypeError):
                continue
        
        if len(buffer_data) < 3:
            return {"error": "Not enough valid data", "waiting": True, "latency_ms": latency_ms}
        
        # Build features
        features = build_fever_features(buffer_data)
        if features is None:
            return {"error": "Could not build features", "waiting": True, "latency_ms": latency_ms}
        
        # Create feature dataframe in correct order
        feature_names = [
            'temp_mean', 'temp_std', 'temp_min', 'temp_max', 'temp_last', 'temp_trend', 'temp_range',
            'hr_mean', 'hr_std', 'hr_min', 'hr_max', 'hr_last', 'hr_trend',
            'activity_mean', 'activity_std', 'activity_trend',
            'sound_mean', 'sound_std',
            'temp_hr_corr', 'temp_activity_corr',
            'hr_elevated', 'temp_elevated'
        ]
        X = pd.DataFrame([[features.get(f, 0) for f in feature_names]], columns=feature_names)
        
        # Predict
        prob = _FEVER_MODEL.predict_proba(X)[0][1]
        prediction = _FEVER_MODEL.predict(X)[0]
        
        # Determine risk level
        if prob > 0.7:
            risk = "HIGH"
        elif prob > 0.4:
            risk = "MEDIUM"
        else:
            risk = "LOW"
        
        current_temp = buffer_data[-1]['temperature'] if buffer_data else None
        
        return {
            "probability": float(prob),
            "risk": risk,
            "prediction": int(prediction),
            "temp_current": current_temp,
            "temp_trend": features.get('temp_trend', 0),
            "hr_current": buffer_data[-1]['heart_rate'] if buffer_data else None,
            "samples_used": len(buffer_data),
            "waiting": False,
            "latency_ms": latency_ms,
        }
        
    except Exception as e:
        return {"error": str(e), "waiting": True}


@app.get("/api/stream")
async def event_stream(request: Request):
    """Server-Sent Events stream for real-time updates."""
    async def generate():
        while True:
            if await request.is_disconnected():
                break
            
            # Gather all statuses
            cough = await get_cough_status()
            fall = await get_fall_status()
            bp = await get_bp_status()
            fever = await get_fever_status()
            
            data = {
                "cough": cough,
                "fall": fall,
                "bp": bp,
                "fever": fever,
                "timestamp": time.time(),
            }
            
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(2)  # Update every 2 seconds
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


# ==================== Dashboard HTML ==================== #
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IoMT Real-Time Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #00d4ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .header .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9em;
            color: #888;
        }
        
        .header .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #00ff88;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .alerts-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .alert-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s ease;
        }
        
        .alert-card:hover {
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.2);
        }
        
        .alert-card.danger {
            background: linear-gradient(135deg, rgba(255, 71, 87, 0.2), rgba(255, 71, 87, 0.05));
            border-color: rgba(255, 71, 87, 0.5);
            animation: alertPulse 1s infinite;
        }
        
        @keyframes alertPulse {
            0%, 100% { box-shadow: 0 0 20px rgba(255, 71, 87, 0.3); }
            50% { box-shadow: 0 0 40px rgba(255, 71, 87, 0.6); }
        }
        
        .alert-card.warning {
            background: linear-gradient(135deg, rgba(255, 165, 0, 0.2), rgba(255, 165, 0, 0.05));
            border-color: rgba(255, 165, 0, 0.5);
        }
        
        .alert-card.success {
            background: linear-gradient(135deg, rgba(0, 255, 136, 0.1), rgba(0, 255, 136, 0.02));
            border-color: rgba(0, 255, 136, 0.3);
        }
        
        .alert-card .icon {
            font-size: 2.5em;
            margin-bottom: 12px;
        }
        
        .alert-card .title {
            font-size: 1.1em;
            font-weight: 600;
            margin-bottom: 8px;
            color: #fff;
        }
        
        .alert-card .value {
            font-size: 2em;
            font-weight: 700;
            margin-bottom: 8px;
        }
        
        .alert-card .subtitle {
            font-size: 0.85em;
            color: #888;
        }
        
        .alert-card.danger .value {
            color: #ff4757;
        }
        
        .alert-card.success .value {
            color: #00ff88;
        }
        
        .alert-card.warning .value {
            color: #ffa500;
        }
        
        .charts-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }
        
        .chart-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 24px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .chart-card h3 {
            margin-bottom: 16px;
            font-size: 1.1em;
            color: #00d4ff;
        }
        
        .chart-container {
            height: 200px;
            position: relative;
        }
        
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #666;
        }
        
        .bp-display {
            display: flex;
            gap: 30px;
            justify-content: center;
        }
        
        .bp-value {
            text-align: center;
        }
        
        .bp-value .number {
            font-size: 2.5em;
            font-weight: 700;
        }
        
        .bp-value .label {
            font-size: 0.9em;
            color: #888;
        }
        
        .bp-value.sbp .number { color: #ff6b6b; }
        .bp-value.dbp .number { color: #4ecdc4; }
        
        .connection-status {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 10px 16px;
            border-radius: 8px;
            font-size: 0.85em;
            background: rgba(0, 0, 0, 0.7);
        }
        
        .connection-status.connected { border-left: 3px solid #00ff88; }
        .connection-status.disconnected { border-left: 3px solid #ff4757; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 IoMT Real-Time Monitor</h1>
        <div class="status-indicator">
            <span class="status-dot"></span>
            <span id="lastUpdate">Connecting...</span>
        </div>
    </div>
    
    <div class="alerts-container">
        <!-- Fall Detection Card -->
        <div class="alert-card success" id="fallCard">
            <div class="icon">🚨</div>
            <div class="title">Fall Detection</div>
            <div class="value" id="fallStatus">Monitoring...</div>
            <div class="subtitle" id="fallSubtitle">Waiting for sensor data</div>
        </div>
        
        <!-- Cough Detection Card -->
        <div class="alert-card success" id="coughCard">
            <div class="icon">🫁</div>
            <div class="title">Cough Detection</div>
            <div class="value" id="coughStatus">--</div>
            <div class="subtitle" id="coughSubtitle">Probability: --</div>
        </div>
        
        <!-- Blood Pressure Card -->
        <div class="alert-card" id="bpCard">
            <div class="icon">💓</div>
            <div class="title">Blood Pressure</div>
            <div class="bp-display">
                <div class="bp-value sbp">
                    <div class="number" id="sbpValue">--</div>
                    <div class="label">SBP (mmHg)</div>
                </div>
                <div class="bp-value dbp">
                    <div class="number" id="dbpValue">--</div>
                    <div class="label">DBP (mmHg)</div>
                </div>
            </div>
            <div class="subtitle" id="bpSubtitle">Waiting for data...</div>
        </div>
        
        <!-- Fever Spike Prediction Card -->
        <div class="alert-card success" id="feverCard">
            <div class="icon">🌡️</div>
            <div class="title">Fever Spike (15 min)</div>
            <div class="value" id="feverStatus">--</div>
            <div class="subtitle" id="feverSubtitle">Analyzing temperature trends...</div>
        </div>
    </div>
    
    <div class="charts-container">
        <div class="chart-card">
            <h3>❤️ Heart Rate</h3>
            <div class="chart-container">
                <canvas id="hrChart"></canvas>
            </div>
        </div>
        
        <div class="chart-card">
            <h3>🌡️ Temperature</h3>
            <div class="chart-container">
                <canvas id="tempChart"></canvas>
            </div>
        </div>
        
        <div class="chart-card">
            <h3>🤸 Activity Level</h3>
            <div class="chart-container">
                <canvas id="activityChart"></canvas>
            </div>
        </div>
        
        <div class="chart-card">
            <h3>🎤 Sound Level</h3>
            <div class="chart-container">
                <canvas id="soundChart"></canvas>
            </div>
        </div>
    </div>
    
    <div class="connection-status connected" id="connectionStatus">
        🟢 Connected
    </div>
    
    <script>
        // Chart configuration
        const chartConfig = {
            type: 'line',
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 0 },
                scales: {
                    x: {
                        type: 'time',
                        time: { tooltipFormat: 'HH:mm:ss' },
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#888', maxTicksLimit: 6 }
                    },
                    y: {
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#888' }
                    }
                },
                plugins: { legend: { display: false } }
            }
        };
        
        // Initialize charts
        const charts = {
            hr: new Chart(document.getElementById('hrChart'), {
                ...chartConfig,
                data: {
                    datasets: [{
                        data: [],
                        borderColor: '#ff6b6b',
                        backgroundColor: 'rgba(255,107,107,0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                }
            }),
            temp: new Chart(document.getElementById('tempChart'), {
                ...chartConfig,
                data: {
                    datasets: [{
                        data: [],
                        borderColor: '#a855f7',
                        backgroundColor: 'rgba(168,85,247,0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                }
            }),
            activity: new Chart(document.getElementById('activityChart'), {
                ...chartConfig,
                data: {
                    datasets: [{
                        data: [],
                        borderColor: '#f97316',
                        backgroundColor: 'rgba(249,115,22,0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                }
            }),
            sound: new Chart(document.getElementById('soundChart'), {
                ...chartConfig,
                data: {
                    datasets: [{
                        data: [],
                        borderColor: '#22c55e',
                        backgroundColor: 'rgba(34,197,94,0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                }
            })
        };
        
        // Update functions
        function updateFallStatus(data) {
            const card = document.getElementById('fallCard');
            const status = document.getElementById('fallStatus');
            const subtitle = document.getElementById('fallSubtitle');
            
            if (data.error) {
                card.className = 'alert-card warning';
                status.textContent = 'API Error';
                subtitle.textContent = data.error;
            } else if (data.waiting) {
                card.className = 'alert-card success';
                status.textContent = 'Monitoring';
                subtitle.textContent = 'No fall events';
            } else if (data.detected) {
                card.className = 'alert-card danger';
                status.textContent = '⚠️ FALL DETECTED!';
                const mins = data.elapsed_sec ? (data.elapsed_sec / 60).toFixed(1) : '?';
                subtitle.textContent = `${mins} minutes ago`;
                // Play alert sound (optional)
                // new Audio('alert.mp3').play();
            } else {
                card.className = 'alert-card success';
                status.textContent = '✓ No Fall';
                if (data.elapsed_sec) {
                    const mins = (data.elapsed_sec / 60).toFixed(1);
                    subtitle.textContent = `Last event: ${mins} min ago`;
                } else {
                    subtitle.textContent = 'Monitoring active';
                }
            }
        }
        
        function updateCoughStatus(data) {
            const card = document.getElementById('coughCard');
            const status = document.getElementById('coughStatus');
            const subtitle = document.getElementById('coughSubtitle');
            
            if (data.error) {
                card.className = 'alert-card warning';
                status.textContent = 'API Error';
                subtitle.textContent = data.error;
            } else if (data.waiting || data.probability === null) {
                card.className = 'alert-card';
                status.textContent = 'Waiting...';
                subtitle.textContent = 'Awaiting audio data';
            } else {
                    const probVal = parseFloat(data.probability);
                    const prob = isNaN(probVal) ? null : (probVal * 100).toFixed(1);
                    subtitle.textContent = prob === null ? 'Probability: --' : `Probability: ${prob}%`;

                    // Dashboard-level detection: treat prob > 10% as cough (user request)
                    const probDetect = probVal && !isNaN(probVal) && probVal > 0.10;
                    const decision = (data.decision === 'COUGH') || probDetect;

                    if (decision) {
                        card.className = 'alert-card danger';
                        status.textContent = '🔴 COUGH!';
                    } else {
                        card.className = 'alert-card success';
                        status.textContent = '✓ Clear';
                    }
            }
        }
        
        function updateBPStatus(data) {
            const card = document.getElementById('bpCard');
            const sbp = document.getElementById('sbpValue');
            const dbp = document.getElementById('dbpValue');
            const subtitle = document.getElementById('bpSubtitle');
            
            if (data.error) {
                card.className = 'alert-card warning';
                sbp.textContent = '--';
                dbp.textContent = '--';
                subtitle.textContent = data.error;
            } else if (data.sbp === null || data.dbp === null) {
                card.className = 'alert-card';
                sbp.textContent = '--';
                dbp.textContent = '--';
                const ecg = data.ecg_samples || 0;
                const ppg = data.ppg_samples || 0;
                const max = data.maxlen || '?';
                subtitle.textContent = `Buffer: ECG=${ecg}, PPG=${ppg} / ${max}`;
            } else {
                card.className = 'alert-card success';
                sbp.textContent = data.sbp.toFixed(0);
                dbp.textContent = data.dbp.toFixed(0);
                subtitle.textContent = 'Live measurement';
            }
        }
        
        function updateFeverStatus(data) {
            const card = document.getElementById('feverCard');
            const status = document.getElementById('feverStatus');
            const subtitle = document.getElementById('feverSubtitle');
            
            if (data.error) {
                card.className = 'alert-card warning';
                status.textContent = 'Error';
                subtitle.textContent = data.error;
            } else if (data.waiting) {
                card.className = 'alert-card';
                status.textContent = 'Analyzing...';
                subtitle.textContent = 'Collecting sensor data';
            } else {
                const prob = (data.probability * 100).toFixed(0);
                const temp = data.temp_current ? data.temp_current.toFixed(1) : '--';
                const trend = data.temp_trend > 0.001 ? '↑' : (data.temp_trend < -0.001 ? '↓' : '→');
                
                if (data.risk === 'HIGH') {
                    card.className = 'alert-card danger';
                    status.textContent = `⚠️ HIGH RISK`;
                    subtitle.textContent = `${prob}% chance | Temp: ${temp}°C ${trend}`;
                } else if (data.risk === 'MEDIUM') {
                    card.className = 'alert-card warning';
                    status.textContent = `⚡ MEDIUM`;
                    subtitle.textContent = `${prob}% chance | Temp: ${temp}°C ${trend}`;
                } else {
                    card.className = 'alert-card success';
                    status.textContent = `✓ LOW`;
                    subtitle.textContent = `${prob}% chance | Temp: ${temp}°C ${trend}`;
                }
            }
        }
        
        function updateCharts(data) {
            if (!data.feeds) return;
            
            const feeds = data.feeds;
            const hrData = [], tempData = [], activityData = [], soundData = [];
            
            feeds.forEach(feed => {
                const time = new Date(feed.created_at);
                if (feed.field1) hrData.push({ x: time, y: parseFloat(feed.field1) });
                if (feed.field5) tempData.push({ x: time, y: parseFloat(feed.field5) });
                if (feed.field3) activityData.push({ x: time, y: parseFloat(feed.field3) });
                if (feed.field4) soundData.push({ x: time, y: parseFloat(feed.field4) });
            });
            
            charts.hr.data.datasets[0].data = hrData;
            charts.temp.data.datasets[0].data = tempData;
            charts.activity.data.datasets[0].data = activityData;
            charts.sound.data.datasets[0].data = soundData;
            
            charts.hr.update('none');
            charts.temp.update('none');
            charts.activity.update('none');
            charts.sound.update('none');
        }
        
        // Real-time updates using Server-Sent Events
        function connectSSE() {
            const evtSource = new EventSource('/api/stream');
            const connStatus = document.getElementById('connectionStatus');
            const lastUpdate = document.getElementById('lastUpdate');
            
            evtSource.onopen = () => {
                connStatus.className = 'connection-status connected';
                connStatus.innerHTML = '🟢 Connected';
            };
            
            evtSource.onmessage = (event) => {
            const data = JSON.parse(event.data);

            updateFallStatus(data.fall);
            updateCoughStatus(data.cough);
            updateBPStatus(data.bp);
            updateFeverStatus(data.fever);

            // Show timings to help debug API latency
            const coughLat = data.cough && data.cough.latency_ms ? `${Math.round(data.cough.latency_ms)}ms` : '-';
            const fallLat = data.fall && data.fall.latency_ms ? `${Math.round(data.fall.latency_ms)}ms` : '-';
            const bpLat = data.bp && data.bp.latency_ms ? `${Math.round(data.bp.latency_ms)}ms` : '-';
            const feverLat = data.fever && data.fever.latency_ms ? `${Math.round(data.fever.latency_ms)}ms` : '-';
            lastUpdate.textContent = `Last update: ${new Date().toLocaleTimeString()} · latencies: cough=${coughLat} fall=${fallLat} fever=${feverLat}`;
            };
            
            evtSource.onerror = () => {
                connStatus.className = 'connection-status disconnected';
                connStatus.innerHTML = '🔴 Disconnected - Reconnecting...';
                evtSource.close();
                setTimeout(connectSSE, 3000);
            };
        }
        
        // Fetch ThingSpeak data periodically
        async function fetchThingSpeakData() {
            try {
                const resp = await fetch('/api/thingspeak?results=50');
                const data = await resp.json();
                updateCharts(data);
            } catch (e) {
                console.error('ThingSpeak fetch error:', e);
            }
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            connectSSE();
            fetchThingSpeakData();
            setInterval(fetchThingSpeakData, 15000); // Update charts every 15s
        });
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard.app:app", host="0.0.0.0", port=8080, reload=True)
