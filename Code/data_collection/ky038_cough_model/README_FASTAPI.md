# FastAPI Cough Detection Service

This adds a buffered audio ingestion server so you avoid fragmenting the cough waveform into tiny HTTP requests (which corrupts envelope statistics). Instead, your device buffers audio for a short window (e.g. 2–5 seconds) and uploads a single WAV.

## Endpoints

- `GET /health` – liveness check.
- `POST /infer` – multipart/form-data file upload (`file` field). Returns JSON with probability, decision, and feature summary.
- `POST /infer/raw` – raw request body containing a WAV file.
- `WS  /ws/stream` – optional WebSocket for sending binary frames; send text `END` to trigger inference; `THRESH=0.7` to adjust threshold.

## Run the Server

```bash
pip install -r requirements.txt
uvicorn fastapi_server:app --host 0.0.0.0 --port 8000
```

## Test with a Sample File

```bash
python send_wav.py --wav ../COUGHVID-dataset/public_dataset/$(ls ../COUGHVID-dataset/public_dataset | grep -m1 '.wav') --threshold 0.6
```

Or with `curl`:

```bash
curl -F "file=@example.wav" 'http://localhost:8000/infer?threshold=0.6'
```

## Integrating with KY-038 Sensor (MCU)

1. Sample the analog pin at a modest rate (e.g., 8 kHz) for a window (2–5 s).
2. Convert collected samples to 16-bit PCM and wrap in a minimal WAV header (or use a library if available).
3. POST that single WAV to `/infer`.
4. (Optional) For lower bandwidth, just compute RMS values on the MCU and send aggregated statistics – but that shifts feature parity; uploading the short WAV preserves flexibility.

### Why Not Send Individual Samples?

Sending each sample (or very small chunks) over HTTP introduces:

- Latency and overhead per request.
- Out-of-order or dropped packets causing irregular spacing.
- Loss of contiguous context needed for envelope / peak statistics.

Buffering locally then sending one file ensures feature extraction matches training.

## WebSocket Streaming (Optional)

If you truly need live streaming, open a binary WebSocket and push PCM WAV bytes (you can start by sending an actual WAV header followed by data). When ready to classify, send the text frame `END`. The server will decode the accumulated buffer and respond with an `inference` JSON event.

## Output JSON Example

```json
{
  "probability": 0.73,
  "decision": "COUGH",
  "threshold": 0.6,
  "features": {"len": 45, "mean": 0.12, "std": 1.03, "max": 4.9, ...},
  "feature_vector_order": ["len", "mean", "std", "max", ...]
}
```

## Next Ideas

- Add authentication (API key) if exposed publicly.
- Rate-limit to prevent abuse.
- Optionally store recent inference results for dashboarding.
- Add a lightweight on-device RMS prefilter to reduce upload size.
