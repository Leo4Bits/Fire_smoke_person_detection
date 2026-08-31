#include <Arduino.h>
#include "sensors.h"
#include <ArduinoJson.h> 

unsigned long thoiGianDocDHT = 0;
unsigned long thoiGianDocMQ = 0;
int giaTriAOMoiNhat = 0; 

void setup() {
  Serial.begin(115200); 
  initSensors();
}

void loop() {
  unsigned long thoiGianHienTai = millis();

  if (thoiGianHienTai - thoiGianDocMQ >= 200) {
    thoiGianDocMQ = thoiGianHienTai; 
    giaTriAOMoiNhat = readMQ135_AO();
  }
  
  if (thoiGianHienTai - thoiGianDocDHT >= 2000) {
    thoiGianDocDHT = thoiGianHienTai; 
    
    float nhietDo = 0.0;
    float doAm = 0.0;
    readDHT11(nhietDo, doAm);
    
    // Đóng gói thành JSON
    JsonDocument doc; 
    doc["nhiet_do"] = nhietDo;
    doc["do_am"] = doAm;
    doc["khi_gas"] = giaTriAOMoiNhat;
    
    //truyền chuỗi JSON qua cổng Serial (Uart)
    serializeJson(doc, Serial);
    Serial.println(); // Bắt buộc phải có dấu xuống dòng để Python dễ tách lớp
  }
}