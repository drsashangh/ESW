# IoMT Medical Monitoring System - Code Documentation

## 📋 Overview

This project implements a comprehensive Internet of Medical Things (IoMT) system for real-time health monitoring, combining embedded devices, machine learning models, and web-based dashboards.

---

## 🏗️ Architecture

The system consists of two main application interfaces:

### 1. **Real-Time Dashboard** (`dashboard/app.py`)
- **Type**: FastAPI-based real-time monitoring interface
- **Purpose**: Live sensor data visualization and predictive AI analytics
- **Features**:
  - Real-time fall detection alerts
  - Live cough detection with probability scoring
  - Continuous blood pressure monitoring
  - Fever spike prediction (15-minute window)
  - Live sensor graphs from ThingSpeak
- **Port**: 8080
- **Technology**: FastAPI, WebSocket streaming, SSE (Server-Sent Events)

### 2. **Static Prediction Interface** (`app.py`)
- **Type**: Streamlit-based static prediction dashboard (https://flashstep-zzz.streamlit.app/)
- **Purpose**: Manual health condition assessment using pre-trained ML models
- **Features**:
  - Diabetes prediction model
  - Hypertension risk assessment
  - Chronic Kidney Disease (CKD) detection
  - Fever monitoring with ThingSpeak integration
- **Technology**: Streamlit, scikit-learn pipelines

---

## 📁 Project Structure

```
Code/
│
├── 📊 Main Applications
│   ├── app.py                          # Streamlit dashboard (static ML predictions)
│   ├── bp_api.py                       # FastAPI blood pressure estimation service
│   ├── fall_api.py                     # FastAPI fall detection event service
│   └── requirements.txt                # Python dependencies
│
├── 📱 Real-Time Dashboard
│   └── dashboard/
│       └── app.py                      # FastAPI real-time monitoring dashboard
│
├── 🤖 Machine Learning Models
│   ├── 01_end_to_end_diabetes_model.ipynb      # Diabetes model training
│   ├── 02_hypertension_model.ipynb             # Hypertension model training
│   └── 03_ckd_model.ipynb                      # CKD model training
│
├── 💾 Trained Models
│   └── models/
│       ├── diabetes_prediction_model.joblib
│       ├── hypertension_model.joblib
│       ├── ckd_model.joblib
│       └── fever_model.joblib
│
├── 🔧 Arduino/ESP32 Firmware
│   └── arduino/
│       ├── cough_fall_detection/
│       │   └── cough_fall_detection.ino
│       ├── ecg_ppg_fastapi_stream/
│       │   └── ecg_ppg_fastapi_stream.ino      # ECG/PPG streaming for BP
│       ├── fall_detection_fastapi/
│       │   └── fall_detection_fastapi.ino      # MPU6050 fall detection
│       ├── ky038_cough_stream_esp32/
│       │   └── ky038_cough_stream_esp32.ino    # KY-038 cough detection
│       └── motors/
│           └── motors.ino                       # Actuator control
│
├── 📡 Data Collection & Model Training
│   └── data_collection/
│       ├── load_data.py                         # Data loading utilities
│       │
│       ├── ky038_cough_model/                   # Cough Detection ML Pipeline
│       │   ├── preprocess_and_train.py
│       │   ├── export_edge_model.py
│       │   ├── fastapi_server.py
│       │   ├── realtime_infer_serial.py
│       │   ├── realtime_infer_thingspeak.py
│       │   └── send_wav.py
│       │
│       └── mimic-bp/                            # Blood Pressure Estimation
│           ├── extract_features_hw827.py        # Feature engineering
│           ├── train_baseline.py
│           ├── train_baseline_no_spo2.py
│           ├── hyperparam_tune.py
│           ├── calibrate_predictions.py
│           ├── ensemble_eval.py
│           ├── ptt_analysis.py
│           └── run_pipeline.py                  # End-to-end BP training
│
├── 🌡️ Fever Prediction Model
│   └── fever_model/
│       ├── train_fever_model.py
│       └── fever_spike_model_features.txt
│
└── 📖 Documentation
    └── INSTRUCTIONS.md                          # Setup and deployment guide

```

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
Arduino IDE / PlatformIO
ESP32 development boards
Sensors: MPU6050, KY-038, AD8232 (ECG), MAX30102 (PPG)
```

### Installation

1. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

2. **Run Static Prediction Dashboard**
```bash
streamlit run app.py
```

3. **Run Real-Time Monitoring Dashboard**
```bash
cd dashboard
uvicorn app:app --host 0.0.0.0 --port 8080
```

4. **Start Backend Services**
```bash
# Blood Pressure API
python -m uvicorn bp_api:app --host 0.0.0.0 --port 8000

# Fall Detection API
python -m uvicorn fall_api:app --host 0.0.0.0 --port 8200
```

---

## 🧪 Model Training

### Train Individual Models
```bash
# Diabetes Model
jupyter notebook 01_end_to_end_diabetes_model.ipynb

# Hypertension Model
jupyter notebook 02_hypertension_model.ipynb

# CKD Model
jupyter notebook 03_ckd_model.ipynb

# Fever Model
python fever_model/train_fever_model.py

# Cough Detection Model
cd data_collection/ky038_cough_model
python preprocess_and_train.py

# Blood Pressure Model
cd data_collection/mimic-bp
python run_pipeline.py
```

---

## 🔌 Hardware Setup

### Supported Sensors
- **MPU6050**: Accelerometer/Gyroscope for fall detection
- **KY-038**: Sound sensor for cough detection
- **AD8232**: ECG sensor for heart activity
- **MAX30102**: PPG sensor for SpO2 and pulse
- **DS18B20/DHT22**: Temperature sensors

### Firmware Upload
1. Open respective `.ino` files in Arduino IDE
2. Select ESP32 board
3. Configure WiFi credentials and API endpoints
4. Upload to device

---

## 📊 Data Flow

```
[Sensors] → [ESP32] → [WiFi] → [ThingSpeak/FastAPI] → [Dashboard]
                                        ↓
                                   [ML Models]
                                        ↓
                                  [Predictions]
```

---

## 🛠️ API Endpoints

### Blood Pressure API (`bp_api.py`)
- `POST /ingest` - Receive ECG/PPG samples
- `GET /predict` - Get BP prediction
- `GET /status` - Buffer status
- `GET /health` - Health check

### Fall Detection API (`fall_api.py`)
- `POST /event` - Receive fall event
- `GET /latest` - Get latest fall event
- `GET /health` - Health check

### Real-Time Dashboard (`dashboard/app.py`)
- `GET /` - Dashboard HTML interface
- `GET /stream` - SSE event stream
- `GET /api/cough/latest` - Latest cough detection
- `GET /api/fall/latest` - Latest fall event
- `GET /api/bp/predict` - Blood pressure prediction
- `GET /api/fever/predict` - Fever spike prediction

---

## 🧠 Machine Learning Models

| Model | Type | Purpose | Input Features |
|-------|------|---------|----------------|
| Diabetes | Classification | Risk assessment | Glucose, BMI, Age, Insulin, etc. |
| Hypertension | Classification | Risk assessment | Age, BMI, Smoking, Family history |
| CKD | Classification | Disease detection | Creatinine, BP, Albumin, Hemoglobin |
| Fever | Time-series | Spike prediction | Temperature history (15-min window) |
| Cough | Audio ML | Event detection | Audio features (MFCC, spectral) |
| Blood Pressure | Regression | Non-invasive BP | ECG/PPG waveform features |

---

## 📝 Configuration

Key environment variables:
```bash
# Dashboard
COUGH_API_URL=http://localhost:8000
FALL_API_URL=http://localhost:8200
BP_API_URL=http://localhost:8000
THINGSPEAK_CHANNEL=3110381
THINGSPEAK_READ_KEY=your_key_here

# Blood Pressure API
BP_MODEL_DIR=data_collection/mimic-bp/models_rt
BP_FS=125
BP_WINDOW_SEC=30

# Fever Model
FEVER_THRESHOLD=37.2
FEVER_MODEL_PATH=fever_model/fever_spike_model.joblib
```

---

## 📚 Documentation

- `INSTRUCTIONS.md` - Detailed setup and deployment guide
- `data_collection/ky038_cough_model/README.md` - Cough detection model details
- `data_collection/mimic-bp/` - Blood pressure model documentation

---


---
