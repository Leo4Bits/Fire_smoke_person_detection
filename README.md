## 1. Phần cứng sử dụng
### 1.1 MCU:
**ESP32**
- Phiên bản: dev kit v1 CP2102

Cảm biến:
- DHT11
- MQ135
## 2. Môi trường quản lý
### 2.1 Môi trường cần chuẩn bị cho Computer vision
- Máy ảo: Miniconda
- Ngôn ngữ: Python
	- Phiên bản: 3.10.20
```
	conda create -n test_model python=3.10.20
```
- Thư viện sử dụng (tự pip install các model dưới): 
	- Ultralytics: Sử dụng xử lý các thao tác với model YOLO11L
	- openCV (cv2)
	- numpy
	- torch
	- sahi
	- logging: tắt log ra màn hình nếu chạy sahi
	- math
	- statistic: tính toán với các iterable
	- collections: thao tác với deque
	- joblib: load Model định dạng .pkl (ở đây là model random forest)
```
	conda install ....
```

## 2.2 Môi trường cần chuẩn bị cho esp32 
 **Lưu ý:** làm các bước ở phần "3 nạp chương trình cho esp32 - cài đặt" rồi quay lại 2.2 này
**Cài đặt extension platformIO**
- Môi trường: Platform Io
![](README\createIMG.png)

- Ngôn ngữ: CPP
- Thư viện:
	- Arduino.h
	- DHT.h
	- ArduinoJson.h: Đóng gói tin gửi qua uart để xử lý trên Model
#### 2.2.1 Cấu hình file platformIo.init
```
; PlatformIO Project Configuration File

;

;   Build options: build flags, source filter

;   Upload options: custom upload port, speed and extra flags

;   Library options: dependencies, extra library storages

;   Advanced options: extra scripting

;

; Please visit documentation for the other options and examples

; https://docs.platformio.org/page/projectconf.html

  

[env:esp32doit-devkit-v1]

platform = espressif32

board = esp32doit-devkit-v1

framework = arduino

monitor_speed = 115200

board_build.filesystem = littlefs

upload_port = COM7

lib_deps =

    adafruit/DHT sensor library @ ^1.4.4

    adafruit/Adafruit Unified Sensor @ ^1.1.9

    bblanchon/ArduinoJson @ ^7.0.4
```
- upload_port: Đổi cổng cho phù hợp với port kết nối từ máy đến esp32

## 3. Nạp chương trình cho esp32
**Cài đặt extension platformIO**
- Tạo file mới " Create New Project " rồi cấu hình như phần **2.2 Môi trường cần chuẩn bị cho esp32**
![](README\projectWizard.png)
- Góc dưới bên trái cho chỗ **BUILD** nhấn **BUILD** và đợi "SUCCESS"
**Sau khi build xong**
- Kết nối uart với esp32
- Nạp chương trình ở  `\GetDataFromSensor\GetDataSensor\src\main.cpp cho esp32
## 4. Chạy chương trình
**Kích hoạt môi trường**
- Chạy máy ảo, sử dụng các thư viện đã cài trong máy ảo

```
conda activate
conda activate test_model
```
**Sau khi kích hoạt môi trường**
- Chạy file mainTest.py `C:\Users\trann\Documents\NHAT_NAM_TRAN\WORK_SPACE\NGHIEN_CUU_KHOA_HOC\NCKH_MODEL_AI\TEST_MODEL\mainTest.py`
