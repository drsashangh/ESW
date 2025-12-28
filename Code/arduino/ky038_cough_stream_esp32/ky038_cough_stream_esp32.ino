/*
  ESP32 + KY-038 buffered uploader for real-time(ish) cough detection

  - Samples KY-038 analog output via ESP32 ADC at 8 kHz
  - Buffers a short window (default 2 s => 16k samples => ~32 KB)
  - Builds a PCM WAV header and POSTs the whole clip to FastAPI /infer/raw
  - Prints the JSON response; toggle threshold if needed

  IMPORTANT: KY-038 modules often output up to 5V analog. ESP32 ADC is 3.3V max.
  Ensure you use a voltage divider or a module variant safe for 3.3V inputs.

  FastAPI server (from this repo):
    uvicorn fastapi_server:app --host 0.0.0.0 --port 8000
  Endpoint: POST http://<host>:8000/infer/raw?threshold=0.6

  Board: ESP32
  Libraries: WiFi.h, HTTPClient.h (bundled with ESP32 core)
*/

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>

// ---------- User Config ---------- //
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// FastAPI server (use IP for reliability on local networks)
const char* SERVER_HOST = "192.168.0.100"; // TODO: set to your machine's LAN IP (same subnet as ESP32)
const uint16_t SERVER_PORT = 8000; // FastAPI default in this repo (see README_FASTAPI.md)
const char* SERVER_PATH = "/infer/raw?threshold=0.6&device_id=esp32-ky038-01"; // include device_id for dashboard polling

// KY-038 analog pin (ESP32 ADC1 recommended pins: 32-39). Example uses GPIO34.
const int MIC_PIN = 34;

// Sampling parameters
const uint32_t SAMPLE_RATE = 8000;      // Hz
const float    WINDOW_SEC  = 0.4f;      // seconds of audio per upload (further reduced latency)
const uint8_t  BITS_PER_SAMPLE = 16;    // 16-bit PCM
const uint8_t  NUM_CHANNELS = 1;        // mono

// DC offset calibration duration at start of each window (seconds)
const float CALIB_SEC = 0.2f;

// ADC config (ESP32: 0..4095 at 12-bit). We'll expand to signed 16-bit centered on offset.
// If your KY-038 baseline sits around mid-scale (~2048), conversion maps to +/- range.

// --------------------------------- //

static const size_t NUM_SAMPLES = (size_t)(SAMPLE_RATE * WINDOW_SEC);
static const size_t CALIB_SAMPLES = (size_t)(SAMPLE_RATE * CALIB_SEC);

// Audio sample buffer (int16_t mono)
static int16_t g_samples[NUM_SAMPLES];

// Minimal WAV header helper
void writeWavHeader(uint8_t* header, uint32_t sampleRate, uint16_t bitsPerSample,
                    uint16_t numChannels, uint32_t numSamples)
{
  // WAV PCM header is 44 bytes
  // RIFF chunk
  const uint32_t byteRate   = sampleRate * numChannels * (bitsPerSample / 8);
  const uint16_t blockAlign = numChannels * (bitsPerSample / 8);
  const uint32_t dataSize   = numSamples * numChannels * (bitsPerSample / 8);
  const uint32_t riffSize   = 36 + dataSize;

  // Clear header
  for (int i = 0; i < 44; ++i) header[i] = 0;

  // ChunkID "RIFF"
  header[0] = 'R'; header[1] = 'I'; header[2] = 'F'; header[3] = 'F';
  // ChunkSize
  header[4] = (uint8_t)(riffSize & 0xFF);
  header[5] = (uint8_t)((riffSize >> 8) & 0xFF);
  header[6] = (uint8_t)((riffSize >> 16) & 0xFF);
  header[7] = (uint8_t)((riffSize >> 24) & 0xFF);
  // Format "WAVE"
  header[8]  = 'W'; header[9] = 'A'; header[10] = 'V'; header[11] = 'E';

  // Subchunk1ID "fmt "
  header[12] = 'f'; header[13] = 'm'; header[14] = 't'; header[15] = ' ';
  // Subchunk1Size (16 for PCM)
  header[16] = 16; header[17] = 0; header[18] = 0; header[19] = 0;
  // AudioFormat (1 = PCM)
  header[20] = 1; header[21] = 0;
  // NumChannels
  header[22] = (uint8_t)(numChannels & 0xFF);
  header[23] = (uint8_t)((numChannels >> 8) & 0xFF);
  // SampleRate
  header[24] = (uint8_t)(sampleRate & 0xFF);
  header[25] = (uint8_t)((sampleRate >> 8) & 0xFF);
  header[26] = (uint8_t)((sampleRate >> 16) & 0xFF);
  header[27] = (uint8_t)((sampleRate >> 24) & 0xFF);
  // ByteRate
  header[28] = (uint8_t)(byteRate & 0xFF);
  header[29] = (uint8_t)((byteRate >> 8) & 0xFF);
  header[30] = (uint8_t)((byteRate >> 16) & 0xFF);
  header[31] = (uint8_t)((byteRate >> 24) & 0xFF);
  // BlockAlign
  header[32] = (uint8_t)(blockAlign & 0xFF);
  header[33] = (uint8_t)((blockAlign >> 8) & 0xFF);
  // BitsPerSample
  header[34] = (uint8_t)(bitsPerSample & 0xFF);
  header[35] = (uint8_t)((bitsPerSample >> 8) & 0xFF);

  // Subchunk2ID "data"
  header[36] = 'd'; header[37] = 'a'; header[38] = 't'; header[39] = 'a';
  // Subchunk2Size (data size)
  header[40] = (uint8_t)(dataSize & 0xFF);
  header[41] = (uint8_t)((dataSize >> 8) & 0xFF);
  header[42] = (uint8_t)((dataSize >> 16) & 0xFF);
  header[43] = (uint8_t)((dataSize >> 24) & 0xFF);
}

void connectWiFi()
{
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  WiFi.setSleep(false); // improve stability for frequent HTTP
  Serial.print("Connecting to WiFi");
  int retries = 0;
  while (WiFi.status() != WL_CONNECTED && retries < 60) {
    delay(500);
    Serial.print(".");
    retries++;
  }
  Serial.println();
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Failed to connect to WiFi");
  } else {
    Serial.print("WiFi connected. IP: ");
    Serial.println(WiFi.localIP());
    Serial.print("Gateway: ");
    Serial.println(WiFi.gatewayIP());
    Serial.print("RSSI: ");
    Serial.println(WiFi.RSSI());
  }
}

// Map 12-bit ADC to signed 16-bit around a runtime-measured DC offset
inline int16_t adcToPCM(int adc, int dcOffset)
{
  int32_t centered = (int32_t)adc - (int32_t)dcOffset; // approx -2048..+2047
  // Expand to 16-bit range: shift left by 4 to map 12-bit to 16-bit
  int32_t val = centered << 4; // ~ -32768..+32752
  if (val > 32767) val = 32767;
  if (val < -32768) val = -32768;
  return (int16_t)val;
}

void sampleWindow()
{
  // Calibrate DC offset for a short period (no strict timing needed)
  int64_t acc = 0;
  for (size_t i = 0; i < CALIB_SAMPLES; ++i) {
    acc += analogRead(MIC_PIN);
    delayMicroseconds((int)(1e6 / SAMPLE_RATE)); // approximate spacing
  }
  int dcOffset = (int)(acc / (int64_t)CALIB_SAMPLES);

  // Collect the window at target sample rate
  uint64_t startMicros = micros();
  for (size_t i = 0; i < NUM_SAMPLES; ++i) {
    int adc = analogRead(MIC_PIN);
    g_samples[i] = adcToPCM(adc, dcOffset);
    uint64_t target = startMicros + (uint64_t)((i + 1) * (1000000.0 / SAMPLE_RATE));
    while ((int64_t)(micros() - target) < 0) {
      // busy wait to maintain sampling rate
    }
  }
}

bool postWav()
{
  // Build header + payload in memory
  const size_t headerSize = 44;
  const size_t dataBytes  = NUM_SAMPLES * sizeof(int16_t);
  const size_t totalSize  = headerSize + dataBytes;

  uint8_t* payload = (uint8_t*)malloc(totalSize);
  if (!payload) {
    Serial.println("malloc failed");
    return false;
  }

  writeWavHeader(payload, SAMPLE_RATE, BITS_PER_SAMPLE, NUM_CHANNELS, NUM_SAMPLES);
  memcpy(payload + headerSize, (uint8_t*)g_samples, dataBytes);

  HTTPClient http;
  String url = String("http://") + SERVER_HOST + ":" + String(SERVER_PORT) + SERVER_PATH;
  http.begin(url);
  http.setTimeout(15000); // 15s timeout to accommodate WiFi hiccups
  http.addHeader("Content-Type", "audio/wav");
  http.addHeader("Connection", "close");

  int code = http.POST(payload, totalSize);
  Serial.printf("HTTP POST %s -> code %d\n", url.c_str(), code);
  if (code > 0) {
    String resp = http.getString();
    Serial.println("Response:");
    Serial.println(resp);
  } else {
    Serial.printf("HTTP error: %s\n", http.errorToString(code).c_str());
  }
  http.end();
  free(payload);
  return code > 0 && code < 400;
}

bool testHealth() {
  if (WiFi.status() != WL_CONNECTED) return false;
  HTTPClient http;
  String url = String("http://") + SERVER_HOST + ":" + String(SERVER_PORT) + "/health";
  http.begin(url);
  http.setTimeout(5000);
  int code = http.GET();
  if (code > 0) {
    String body = http.getString();
    Serial.printf("Health GET %s -> %d body=%s\n", url.c_str(), code, body.c_str());
  } else {
    // print human-friendly error string to help debug connection issues
    String err = http.errorToString(code);
    Serial.printf("Health GET failed code=%d err=%s url=%s\n", code, err.c_str(), url.c_str());
  }
  http.end();
  return code == 200;
}

void setup()
{
  Serial.begin(115200);
  delay(500);
  Serial.println("KY-038 ESP32 uploader starting...");

  // ADC configuration
  analogReadResolution(12); // 0..4095
  // Adjust attenuation according to your input scaling; 11dB ~ up to ~3.6V
  analogSetPinAttenuation(MIC_PIN, ADC_11db); // use pin-based API

  connectWiFi();
}

void loop()
{
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  // Verify server reachability before sampling & uploading
  if (!testHealth()) {
    Serial.println("Server unreachable, skipping upload this cycle.");
    delay(1000);
    return;
  }

  Serial.println("Sampling window...");
  sampleWindow();
  Serial.println("Uploading...");
  bool ok = postWav();
  if (!ok) {
    Serial.println("Upload failed");
  }

  // Small pause between uploads; adjust to your needs
  delay(200);
}
