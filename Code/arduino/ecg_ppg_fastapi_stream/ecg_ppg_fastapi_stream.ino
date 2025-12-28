/*
  ecg_ppg_fastapi_stream.ino

  Streams buffered ECG and PPG samples from an ESP32 to the non-invasive BP FastAPI service.
  The sketch samples the configured analog pins at SAMPLE_RATE_HZ, batches BATCH_SIZE samples,
  and POSTs them to http://<SERVER_HOST>:<SERVER_PORT>/ingest as JSON arrays so the backend
  can maintain a rolling buffer for continuous waveform inference.

  Customize the WiFi credentials, server host/port, and analog input pins before flashing.

  FastAPI service (from this repo):
      /home/abhijit-suhas/esw/IOMT-eswproj/Zzz/.venv/bin/python -m uvicorn bp_api:app --host 0.0.0.0 --port 8000
*/

#include <WiFi.h>
#include <HTTPClient.h>

// ----------------------- User configuration -----------------------
const char *WIFI_SSID = "YOUR_WIFI_SSID";
const char *WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// Use the LAN IP address of the machine running bp_api.py for reliability
const char *SERVER_HOST = "192.168.39.4";   // TODO: replace with your server's IP
const uint16_t SERVER_PORT = 8100;
const char *DEVICE_ID = "esp32-ecg-01";

// Analog input configuration
const int ECG_PIN = 34;  // GPIO34 is ADC1_CH6 (input only)
const int PPG_PIN = 35;  // GPIO35 is ADC1_CH7 (input only)

// Sampling / batching settings (match bp_api defaults: 125 Hz, 30 s window)
const uint16_t SAMPLE_RATE_HZ = 125;                 // Hz
const uint16_t BATCH_SIZE = 125;                     // Samples per POST (≈1 second)
const float ADC_REF_VOLTAGE = 3.30f;                 // ESP32 ADC reference (approx.)
const float ADC_FULL_SCALE = 4095.0f;                // 12-bit ADC

// Optional: scale readings to millivolts. Set to false to send raw 0-4095 values.
const bool SCALE_TO_MILLIVOLTS = true;

// ------------------------------------------------------------------

WiFiClient wifiClient;

void connectWiFi()
{
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("[WiFi] Connecting");
  uint8_t attempts = 0;
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print('.');
    attempts++;
    if (attempts >= 60)
    {
      Serial.println("\n[WiFi] Failed to connect, restarting...");
      ESP.restart();
    }
  }
  Serial.printf("\n[WiFi] Connected: %s\n", WiFi.localIP().toString().c_str());
}

float convertAdc(int raw)
{
  if (!SCALE_TO_MILLIVOLTS)
  {
    return static_cast<float>(raw);
  }
  return (raw / ADC_FULL_SCALE) * ADC_REF_VOLTAGE * 1000.0f; // millivolts
}

String arrayToJson(const float *data, size_t count)
{
  String json;
  if (count == 0)
  {
    return String("[]");
  }
  json.reserve(count * 8 + 2); // rough capacity
  json += '[';
  for (size_t i = 0; i < count; ++i)
  {
    if (i > 0)
    {
      json += ',';
    }
    json += String(data[i], 3); // three decimal places
  }
  json += ']';
  return json;
}

bool postBatch(const float *ecg, size_t ecgCount, const float *ppg, size_t ppgCount)
{
  if (WiFi.status() != WL_CONNECTED)
  {
    Serial.println("[HTTP] WiFi disconnected, attempting reconnection...");
    connectWiFi();
  }

  HTTPClient http;
  String url = String("http://") + SERVER_HOST + ":" + SERVER_PORT + "/ingest";
  http.setTimeout(7000);

  if (!http.begin(wifiClient, url))
  {
    Serial.println("[HTTP] Failed to begin connection");
    return false;
  }

  http.addHeader("Content-Type", "application/json");

  String payload = String("{\"device_id\":\"") + DEVICE_ID + "\"";
  if (ecgCount > 0)
  {
    payload += ",\"ecg\":";
    payload += arrayToJson(ecg, ecgCount);
  }
  if (ppgCount > 0)
  {
    payload += ",\"ppg\":";
    payload += arrayToJson(ppg, ppgCount);
  }
  payload += "}";

  int statusCode = http.POST(payload);
  if (statusCode > 0)
  {
    Serial.printf("[HTTP] POST /ingest -> %d\n", statusCode);
    if (statusCode >= 200 && statusCode < 300)
    {
      String resp = http.getString();
      Serial.printf("[HTTP] Response: %s\n", resp.c_str());
      http.end();
      return true;
    }
    else
    {
      Serial.printf("[HTTP] Unexpected status %d\n", statusCode);
    }
  }
  else
  {
    Serial.printf("[HTTP] POST failed: %s\n", http.errorToString(statusCode).c_str());
  }

  http.end();
  return false;
}

void setup()
{
  Serial.begin(115200);
  delay(200);
  Serial.println("\n[Setup] ECG + PPG FastAPI streamer starting...");

  analogReadResolution(12); // 12-bit resolution (0-4095)
  pinMode(ECG_PIN, INPUT);
  pinMode(PPG_PIN, INPUT);

  connectWiFi();
}

void loop()
{
  static float ecgBatch[BATCH_SIZE];
  static float ppgBatch[BATCH_SIZE];

  const uint32_t samplePeriodUs = 1000000UL / SAMPLE_RATE_HZ;
  uint32_t nextSampleUs = micros();

  for (uint16_t i = 0; i < BATCH_SIZE; ++i)
  {
    // wait until the next sample time
    while ((int32_t)(micros() - nextSampleUs) < 0)
    {
      // tight wait; keep loop short for timing accuracy
      delayMicroseconds(10);
    }
    nextSampleUs += samplePeriodUs;

    int ecgRaw = analogRead(ECG_PIN);
    int ppgRaw = analogRead(PPG_PIN);

    ecgBatch[i] = convertAdc(ecgRaw);
    ppgBatch[i] = convertAdc(ppgRaw);
  }

  if (!postBatch(ecgBatch, BATCH_SIZE, ppgBatch, BATCH_SIZE))
  {
    Serial.println("[Loop] POST failed, will retry next batch.");
    delay(500);
  }
}
