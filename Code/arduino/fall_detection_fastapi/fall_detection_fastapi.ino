/*
  fall_detection_fastapi.ino

  ESP32 + MPU6050 fall detector that performs on-board classification and reports
  events to the Fall Detection FastAPI service (Zzz/fall_api.py).

  The sketch reuses the tuning values supplied in the reference code:
    - FREEFALL_THRESHOLD (Sum Vector Magnitude) indicates potential free-fall
    - IMPACT_THRESHOLD signals the impact following a free-fall
    - FALL_TIME_WINDOW bounds the time allowed between free-fall and impact

  When a fall is detected we POST a JSON payload to
      http://<SERVER_HOST>:<SERVER_PORT>/event
  containing acceleration, gyro, temperature, and SVM readings.

  We also send a periodic "heartbeat" (fall=false) so the dashboard can
  display "No fall detected" before the first event occurs.

  Requirements:
    - ESP32 development board
    - MPU6050 connected via I2C (default pins: SDA=21, SCL=22)
    - Adafruit_MPU6050 library and Adafruit Unified Sensor library

  FastAPI service:
    /home/abhijit-suhas/esw/IOMT-eswproj/Zzz/.venv/bin/python -m uvicorn Zzz.fall_api:app --host 0.0.0.0 --port 8200
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// ----------------------- User configuration -----------------------
const char *WIFI_SSID = "YOUR_WIFI_SSID";
const char *WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// Fall API configuration (set to the LAN IP running fall_api.py)
const char *SERVER_HOST = "192.168.39.4";   // TODO: replace with your server's IP
const uint16_t SERVER_PORT = 8200;
const char *DEVICE_ID = "esp32-mpu6050-01";

// Thresholds (in m/s^2)
const float FREEFALL_THRESHOLD = 5.89f; // ~0.6 g
const float IMPACT_THRESHOLD   = 11.0f; // ~1.1 g
const unsigned long FALL_TIME_WINDOW = 2000UL; // ms

// Heartbeat interval when no fall occurs (ms)
const unsigned long HEARTBEAT_INTERVAL = 30000UL;

// ------------------------------------------------------------------

Adafruit_MPU6050 mpu;
WiFiClient wifiClient;
unsigned long lastHeartbeat = 0;
bool fallInProgress = false;
unsigned long fallStartTime = 0;

// ------------------------------------------------------------------

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

String buildEventPayload(bool fallDetected, float svm,
                         const sensors_event_t &accel,
                         const sensors_event_t &gyro,
                         const sensors_event_t &temp)
{
  String payload = "{";
  payload.reserve(256);
  payload += "\"device_id\":\"" + String(DEVICE_ID) + "\"";
  payload += ",\"fall\":";
  payload += fallDetected ? "true" : "false";
  payload += ",\"svm\":" + String(svm, 3);
  payload += ",\"ax\":" + String(accel.acceleration.x, 3);
  payload += ",\"ay\":" + String(accel.acceleration.y, 3);
  payload += ",\"az\":" + String(accel.acceleration.z, 3);
  payload += ",\"gx\":" + String(gyro.gyro.x, 3);
  payload += ",\"gy\":" + String(gyro.gyro.y, 3);
  payload += ",\"gz\":" + String(gyro.gyro.z, 3);
  payload += ",\"temperature_c\":" + String(temp.temperature, 2);
  payload += "}";
  return payload;
}

bool postEvent(bool fallDetected, float svm,
               const sensors_event_t &accel,
               const sensors_event_t &gyro,
               const sensors_event_t &temp)
{
  if (WiFi.status() != WL_CONNECTED)
  {
    Serial.println("[HTTP] WiFi disconnected, attempting reconnection...");
    connectWiFi();
  }

  HTTPClient http;
  String url = String("http://") + SERVER_HOST + ":" + SERVER_PORT + "/event";
  http.setTimeout(7000);

  if (!http.begin(wifiClient, url))
  {
    Serial.println("[HTTP] Failed to begin connection");
    return false;
  }

  http.addHeader("Content-Type", "application/json");
  String payload = buildEventPayload(fallDetected, svm, accel, gyro, temp);
  int statusCode = http.POST(payload);
  if (statusCode > 0)
  {
    Serial.printf("[HTTP] POST /event -> %d\n", statusCode);
    if (statusCode >= 200 && statusCode < 300)
    {
      String resp = http.getString();
      Serial.printf("[HTTP] Response: %s\n", resp.c_str());
      http.end();
      return true;
    }
    Serial.printf("[HTTP] Unexpected status %d\n", statusCode);
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

  Serial.println("\n[Setup] Fall detection streamer starting...");
  connectWiFi();

  Serial.println("Finding MPU6050 chip...");
  if (!mpu.begin())
  {
    Serial.println("Failed to find MPU6050 chip. Check wiring!");
    while (true)
    {
      delay(10);
    }
  }
  Serial.println("MPU6050 Found!");

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  Serial.println("------------------------------------");
  lastHeartbeat = millis();
}

void loop()
{
  sensors_event_t accel, gyro, temp;
  mpu.getEvent(&accel, &gyro, &temp);

  // Compute Sum Vector Magnitude (SVM)
  float ax = accel.acceleration.x;
  float ay = accel.acceleration.y;
  float az = accel.acceleration.z;
  float svm = sqrtf(ax * ax + ay * ay + az * az);

  bool fallDetectedNow = false;

  if (svm < FREEFALL_THRESHOLD && !fallInProgress)
  {
    fallInProgress = true;
    fallStartTime = millis();
    Serial.println("--> Potential free fall detected");
  }

  if (fallInProgress)
  {
    if (svm > IMPACT_THRESHOLD)
    {
      Serial.println("!!!!!!!!!!!!!!!!!!!!!!");
      Serial.println("!!! FALL DETECTED! !!!");
      Serial.println("!!!!!!!!!!!!!!!!!!!!!!");
      fallDetectedNow = true;
      fallInProgress = false;
    }
    else if (millis() - fallStartTime > FALL_TIME_WINDOW)
    {
      Serial.println("--> Fall timeout. Resetting state.");
      fallInProgress = false;
    }
  }

  // Send events ----------------------------------------------------
  unsigned long now = millis();
  if (fallDetectedNow)
  {
    if (postEvent(true, svm, accel, gyro, temp))
    {
      lastHeartbeat = now;
    }
  }
  else if (now - lastHeartbeat >= HEARTBEAT_INTERVAL)
  {
    if (postEvent(false, svm, accel, gyro, temp))
    {
      lastHeartbeat = now;
    }
  }

  // Optional: serial diagnostics
  Serial.print("SVM: ");
  Serial.print(svm, 3);
  Serial.print(" m/s^2\t");
  Serial.print("AX: ");
  Serial.print(ax, 3);
  Serial.print("\tAY: ");
  Serial.print(ay, 3);
  Serial.print("\tAZ: ");
  Serial.print(az, 3);
  Serial.println(" m/s^2");

  Serial.print("Gyro X: ");
  Serial.print(gyro.gyro.x, 3);
  Serial.print(" rad/s\tY: ");
  Serial.print(gyro.gyro.y, 3);
  Serial.print("\tZ: ");
  Serial.print(gyro.gyro.z, 3);
  Serial.println(" rad/s");

  Serial.print("Temperature: ");
  Serial.print(temp.temperature, 2);
  Serial.println(" °C");

  Serial.println("------------------------------------");
  delay(100); // faster updates than original 1 second for improved responsiveness
}
