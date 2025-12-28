/*
 * ESP32 + KY-038 buffered uploader for real-time(ish) cough detection
 * AND MPU-6050 for fall detection.
 *
 * Publishes to:
 *   - Cough API: http://<SERVER_HOST>:8000/infer/raw
 *   - Fall API:  http://<SERVER_HOST>:8200/alert/fall
 *
 * IMPORTANT: KY-038 modules often output up to 5V analog. ESP32 ADC is 3.3V max.
 * Ensure you use a voltage divider or a module variant safe for 3.3V inputs.
 *
 * MPU-6050: Uses I2C (SDA=GPIO21, SCL=GPIO22).
 */

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>

// ==================== USER CONFIG ==================== //
// WiFi credentials
const char* WIFI_SSID     = "realme Gt6t";
const char* WIFI_PASSWORD = "anish123";

// FastAPI server IP (your laptop's IP on the network)
const char* SERVER_HOST = "192.168.39.4";

// Cough API (fastapi_server.py on port 8000)
const uint16_t COUGH_API_PORT = 8000;
const char* COUGH_ENDPOINT = "/infer/raw";
const float COUGH_THRESHOLD = 0.6f;
const char* COUGH_DEVICE_ID = "esp32-ky038-01";

// Fall API (fall_api.py on port 8200)
const uint16_t FALL_API_PORT = 8200;
const char* FALL_ENDPOINT = "/alert/fall";
const char* FALL_DEVICE_ID = "esp32-mpu6050-01";

// KY-038 analog pin
const int MIC_PIN = 36;

// MPU-6050 I2C Address (0x68 when AD0 is low, 0x69 when high)
const int MPU_ADDR = 0x68;

// Fall Detection Config
const float FALL_THRESHOLD_G = 2.5f;  // Threshold for sudden acceleration change (G-forces)

// Timing
const unsigned long AUDIO_CYCLE_MS = 500;      // How often to sample/upload audio
const unsigned long FALL_CHECK_MS = 50;        // How often to check for falls
const unsigned long HEALTH_CHECK_MS = 10000;   // How often to check server health

// Sampling parameters (KY-038)
const uint32_t SAMPLE_RATE     = 8000;
const float    WINDOW_SEC      = 0.4f;
const uint8_t  BITS_PER_SAMPLE = 16;
const uint8_t  NUM_CHANNELS    = 1;
const float    CALIB_SEC       = 0.2f;

// ===================================================== //

// --- KY-038 Buffer ---
static const size_t NUM_SAMPLES   = (size_t)(SAMPLE_RATE * WINDOW_SEC);
static const size_t CALIB_SAMPLES = (size_t)(SAMPLE_RATE * CALIB_SEC);
static int16_t g_samples[NUM_SAMPLES];

// --- MPU-6050 State ---
float g_last_accel_mag_g = 0.0f;

// --- Timing State ---
unsigned long g_last_audio_time = 0;
unsigned long g_last_fall_check_time = 0;
unsigned long g_last_health_check_time = 0;
bool g_cough_api_healthy = false;
bool g_fall_api_healthy = false;

// ==================== WAV HEADER ==================== //
void writeWavHeader(uint8_t* header, uint32_t sampleRate, uint16_t bitsPerSample,
                    uint16_t numChannels, uint32_t numSamples)
{
  const uint32_t byteRate   = sampleRate * numChannels * (bitsPerSample / 8);
  const uint16_t blockAlign = numChannels * (bitsPerSample / 8);
  const uint32_t dataSize   = numSamples * numChannels * (bitsPerSample / 8);
  const uint32_t riffSize   = 36 + dataSize;

  for (int i = 0; i < 44; ++i) header[i] = 0;

  // RIFF
  header[0] = 'R'; header[1] = 'I'; header[2] = 'F'; header[3] = 'F';
  header[4] = (uint8_t)(riffSize & 0xFF); header[5] = (uint8_t)((riffSize >> 8) & 0xFF);
  header[6] = (uint8_t)((riffSize >> 16) & 0xFF); header[7] = (uint8_t)((riffSize >> 24) & 0xFF);
  // WAVE
  header[8]  = 'W'; header[9] = 'A'; header[10] = 'V'; header[11] = 'E';
  // fmt
  header[12] = 'f'; header[13] = 'm'; header[14] = 't'; header[15] = ' ';
  header[16] = 16; header[17] = 0; header[18] = 0; header[19] = 0;
  header[20] = 1; header[21] = 0; // PCM
  header[22] = (uint8_t)(numChannels & 0xFF); header[23] = (uint8_t)((numChannels >> 8) & 0xFF);
  header[24] = (uint8_t)(sampleRate & 0xFF); header[25] = (uint8_t)((sampleRate >> 8) & 0xFF);
  header[26] = (uint8_t)((sampleRate >> 16) & 0xFF); header[27] = (uint8_t)((sampleRate >> 24) & 0xFF);
  header[28] = (uint8_t)(byteRate & 0xFF); header[29] = (uint8_t)((byteRate >> 8) & 0xFF);
  header[30] = (uint8_t)((byteRate >> 16) & 0xFF); header[31] = (uint8_t)((byteRate >> 24) & 0xFF);
  header[32] = (uint8_t)(blockAlign & 0xFF); header[33] = (uint8_t)((blockAlign >> 8) & 0xFF);
  header[34] = (uint8_t)(bitsPerSample & 0xFF); header[35] = (uint8_t)((bitsPerSample >> 8) & 0xFF);
  // data
  header[36] = 'd'; header[37] = 'a'; header[38] = 't'; header[39] = 'a';
  header[40] = (uint8_t)(dataSize & 0xFF); header[41] = (uint8_t)((dataSize >> 8) & 0xFF);
  header[42] = (uint8_t)((dataSize >> 16) & 0xFF); header[43] = (uint8_t)((dataSize >> 24) & 0xFF);
}

// ==================== WIFI ==================== //
void connectWiFi()
{
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  WiFi.setSleep(false);
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

// ==================== HEALTH CHECKS ==================== //
bool checkCoughApiHealth() {
  if (WiFi.status() != WL_CONNECTED) return false;
  
  HTTPClient http;
  String url = String("http://") + SERVER_HOST + ":" + String(COUGH_API_PORT) + "/health";
  http.begin(url);
  http.setTimeout(5000);
  int code = http.GET();
  
  if (code == 200) {
    Serial.println("[Cough API] Health OK");
    http.end();
    return true;
  } else {
    Serial.printf("[Cough API] Health FAILED (code=%d)\n", code);
    http.end();
    return false;
  }
}

bool checkFallApiHealth() {
  if (WiFi.status() != WL_CONNECTED) return false;
  
  HTTPClient http;
  String url = String("http://") + SERVER_HOST + ":" + String(FALL_API_PORT) + "/health";
  http.begin(url);
  http.setTimeout(5000);
  int code = http.GET();
  
  if (code == 200) {
    Serial.println("[Fall API] Health OK");
    http.end();
    return true;
  } else {
    Serial.printf("[Fall API] Health FAILED (code=%d)\n", code);
    http.end();
    return false;
  }
}

void updateHealthStatus() {
  g_cough_api_healthy = checkCoughApiHealth();
  g_fall_api_healthy = checkFallApiHealth();
}

// ==================== KY-038 AUDIO ==================== //
inline int16_t adcToPCM(int adc, int dcOffset)
{
  int32_t centered = (int32_t)adc - (int32_t)dcOffset;
  int32_t val = centered << 4;
  if (val > 32767) val = 32767;
  if (val < -32768) val = -32768;
  return (int16_t)val;
}

void sampleWindow()
{
  // Calibration: measure DC offset
  int64_t acc = 0;
  for (size_t i = 0; i < CALIB_SAMPLES; ++i) {
    acc += analogRead(MIC_PIN);
    delayMicroseconds((int)(1e6 / SAMPLE_RATE));
  }
  int dcOffset = (int)(acc / (int64_t)CALIB_SAMPLES);

  // Sample audio
  uint64_t startMicros = micros();
  for (size_t i = 0; i < NUM_SAMPLES; ++i) {
    int adc = analogRead(MIC_PIN);
    g_samples[i] = adcToPCM(adc, dcOffset);
    uint64_t target = startMicros + (uint64_t)((i + 1) * (1000000.0 / SAMPLE_RATE));
    while ((int64_t)(micros() - target) < 0) {
      // busy wait for precise timing
    }
  }
}

bool postCoughAudio()
{
  const size_t headerSize = 44;
  const size_t dataBytes  = NUM_SAMPLES * sizeof(int16_t);
  const size_t totalSize  = headerSize + dataBytes;

  uint8_t* payload = (uint8_t*)malloc(totalSize);
  if (!payload) {
    Serial.println("[Cough] malloc failed");
    return false;
  }

  writeWavHeader(payload, SAMPLE_RATE, BITS_PER_SAMPLE, NUM_CHANNELS, NUM_SAMPLES);
  memcpy(payload + headerSize, (uint8_t*)g_samples, dataBytes);

  HTTPClient http;
  // Build URL: /infer/raw?threshold=0.6&device_id=esp32-ky038-01
  String url = String("http://") + SERVER_HOST + ":" + String(COUGH_API_PORT) + COUGH_ENDPOINT;
  url += "?threshold=" + String(COUGH_THRESHOLD, 2);
  url += "&device_id=" + String(COUGH_DEVICE_ID);
  
  http.begin(url);
  http.setTimeout(15000);
  http.addHeader("Content-Type", "audio/wav");
  http.addHeader("Connection", "close");

  int code = http.POST(payload, totalSize);
  
  if (code > 0) {
    String resp = http.getString();
    Serial.printf("[Cough] POST %s -> %d\n", url.c_str(), code);
    
    // Parse response to check if cough detected
    if (resp.indexOf("\"decision\":\"COUGH\"") >= 0 || resp.indexOf("\"decision\": \"COUGH\"") >= 0) {
      Serial.println("*** COUGH DETECTED! ***");
    }
  } else {
    Serial.printf("[Cough] HTTP error: %s\n", http.errorToString(code).c_str());
  }
  
  http.end();
  free(payload);
  return code > 0 && code < 400;
}

// ==================== MPU-6050 FALL DETECTION ==================== //
bool readAccel(float& accel_x, float& accel_y, float& accel_z) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);  // ACCEL_XOUT_H register
  if (Wire.endTransmission(false) != 0) {
    return false;
  }
  
  if (Wire.requestFrom(MPU_ADDR, 6, true) != 6) {
    return false;
  }

  int16_t Ax = Wire.read() << 8 | Wire.read();
  int16_t Ay = Wire.read() << 8 | Wire.read();
  int16_t Az = Wire.read() << 8 | Wire.read();

  // Convert to G-forces (assuming +/- 2g scale: 16384 LSB/g)
  const float ACCEL_SCALE = 16384.0f;
  accel_x = (float)Ax / ACCEL_SCALE;
  accel_y = (float)Ay / ACCEL_SCALE;
  accel_z = (float)Az / ACCEL_SCALE;

  return true;
}

bool checkForFall() {
  float ax, ay, az;
  if (!readAccel(ax, ay, az)) {
    return false;
  }

  // Calculate magnitude of acceleration vector
  float accel_mag_g = sqrt(ax * ax + ay * ay + az * az);
  
  // Check for sudden change in acceleration (impact detection)
  if (g_last_accel_mag_g > 0.1f) {
    float delta_g = fabsf(accel_mag_g - g_last_accel_mag_g);
    if (delta_g > FALL_THRESHOLD_G) {
      Serial.printf("!!! FALL DETECTED (Delta %.2f G) !!!\n", delta_g);
      g_last_accel_mag_g = accel_mag_g;
      return true;
    }
  }
  
  g_last_accel_mag_g = accel_mag_g;
  return false;
}

bool postFallAlert() {
  HTTPClient http;
  
  // Build URL: /alert/fall?device_id=esp32-mpu6050-01
  String url = String("http://") + SERVER_HOST + ":" + String(FALL_API_PORT) + FALL_ENDPOINT;
  url += "?device_id=" + String(FALL_DEVICE_ID);
  
  http.begin(url);
  http.setTimeout(5000);
  http.addHeader("Content-Type", "application/json");
  http.addHeader("Connection", "close");

  // JSON payload matching the SimpleFallAlert model in fall_api.py
  const char* jsonPayload = "{\"event\": \"fall\", \"sensor\": \"MPU-6050\"}";

  int code = http.POST(jsonPayload);
  
  if (code > 0) {
    String resp = http.getString();
    Serial.printf("[Fall] Alert POST -> %d\n", code);
    Serial.println(resp);
  } else {
    Serial.printf("[Fall] HTTP error: %s\n", http.errorToString(code).c_str());
  }
  
  http.end();
  return code > 0 && code < 400;
}

// ==================== SETUP ==================== //
void setup()
{
  Serial.begin(115200);
  delay(500);
  Serial.println("\n========================================");
  Serial.println("ESP32 Cough & Fall Detection Starting...");
  Serial.println("========================================");
  Serial.printf("Cough API: http://%s:%d%s\n", SERVER_HOST, COUGH_API_PORT, COUGH_ENDPOINT);
  Serial.printf("Fall API:  http://%s:%d%s\n", SERVER_HOST, FALL_API_PORT, FALL_ENDPOINT);
  Serial.println("========================================\n");

  // I2C setup for MPU-6050
  Wire.begin();
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);  // PWR_MGMT_1 register
  Wire.write(0);     // Wake up MPU-6050
  if (Wire.endTransmission(true) == 0) {
    Serial.println("[MPU-6050] Initialized successfully");
  } else {
    Serial.println("[MPU-6050] Initialization FAILED - check wiring!");
  }

  // ADC configuration for KY-038
  analogReadResolution(12);
  analogSetPinAttenuation(MIC_PIN, ADC_11db);
  Serial.println("[KY-038] ADC configured");

  // Connect to WiFi
  connectWiFi();
  
  // Initial health check
  if (WiFi.status() == WL_CONNECTED) {
    updateHealthStatus();
  }
  
  Serial.println("\n[READY] Monitoring started...\n");
}

// ==================== LOOP ==================== //
void loop()
{
  unsigned long now = millis();
  
  // Reconnect WiFi if needed
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] Disconnected, reconnecting...");
    connectWiFi();
    return;
  }

  // Periodic health check
  if (now - g_last_health_check_time >= HEALTH_CHECK_MS) {
    g_last_health_check_time = now;
    updateHealthStatus();
  }

  // Fall detection (high priority, checked frequently)
  if (now - g_last_fall_check_time >= FALL_CHECK_MS) {
    g_last_fall_check_time = now;
    
    if (checkForFall()) {
      if (g_fall_api_healthy) {
        Serial.println("[Fall] Uploading alert...");
        postFallAlert();
      } else {
        Serial.println("[Fall] API unavailable, alert not sent");
      }
    }
  }

  // Audio sampling and cough detection
  if (now - g_last_audio_time >= AUDIO_CYCLE_MS) {
    g_last_audio_time = now;
    
    if (g_cough_api_healthy) {
      sampleWindow();
      postCoughAudio();
    }
  }
}
