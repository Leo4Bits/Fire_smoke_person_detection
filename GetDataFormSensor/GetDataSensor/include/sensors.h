#ifndef SENSORS_H
#define SENSORS_H

#include <Arduino.h>
#include <DHT.h>

#define DHTPIN 4
#define DHTTYPE DHT11

#define MQ135_AO_PIN 34
#define MQ135_DO_PIN 35

void initSensors();
void readDHT11(float &temp, float &hum);
int readMQ135_AO();
int readMQ135_DO();

#endif