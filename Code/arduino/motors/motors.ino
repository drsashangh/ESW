#define USE_ARDUINO_INTERRUPTS false // ESP32 often struggles with the library's interrupts
#include <PulseSensorPlayground.h>

const int PulseWire = 36; // Connected to GPIO 36 (VP) - strictly an analog input pin
int Threshold = 550;      // Adjust this if your readings are unreliable (range 0-4095 for ESP32 usually, but library might scale it)

PulseSensorPlayground pulseSensor;

void setup() {
  Serial.begin(115200);

  // Configure the PulseSensor object
  pulseSensor.analogInput(PulseWire);
  pulseSensor.setThreshold(Threshold);

  if (pulseSensor.begin()) {
    Serial.println("PulseSensor Object created!");
  }
}

void loop() {
  int myBPM = pulseSensor.getBeatsPerMinute();

  if (pulseSensor.sawStartOfBeat()) {
    Serial.println("♥  A HeartBeat Happened ! ");
    Serial.print("BPM: ");
    Serial.println(myBPM);
  }

  // A short delay to keep the loop running smoothly without missing beats.
  // Since interrupts are off, we must poll frequently.
  delay(200);
}