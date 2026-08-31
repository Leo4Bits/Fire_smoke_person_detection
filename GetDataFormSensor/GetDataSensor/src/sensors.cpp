#include "sensors.h"

DHT dht(DHTPIN, DHTTYPE);

void initSensors() {
  dht.begin();
  pinMode(MQ135_AO_PIN, INPUT);
  pinMode(MQ135_DO_PIN, INPUT);
}

void readDHT11(float &temp, float &hum) {
  hum = dht.readHumidity();
  temp = dht.readTemperature(); 
}

int readMQ135_AO() {
  return analogRead(MQ135_AO_PIN);
}

int readMQ135_DO() {
  return digitalRead(MQ135_DO_PIN);
}