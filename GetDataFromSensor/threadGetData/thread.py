import serial
import json
import time
import threading
from PreProcessing import imgPreProcessing, numPreProcessing
from PreProcessing.numPreProcessing import pre_gas, buf_gas

# Cấu hình cổng COM
SERIAL_PORT = 'COM7'  
BAUD_RATE = 115200
JSON_FILE = 'sensor_data.json'
# Reuse lại các biến ở numPreProcessing
pre_gas
buf_gas

#data sau predicted
sensor_info_predicted = {'status': "",# str
        'label': 0, # int
        'confidence': 0.0,
        'smoothed_data': {
            'temp': 0.0,
            'hum': 0.0,
            'gas': 0.0,
            'diff_gas': 0.0}
}
# Mở kết nối với ESP32
def serial_reader_thread():
    '''Hàm sử dụng để đọc realtime data từ cảm biến - UART
        - Kết hợp:
        
    Khởi chạy Thread Serial
        + thread = threading.Thread(target=serial_reader_thread, daemon=True)
        + thread.start()'''
    global pre_gas,buf_gas, sensor_info_predicted
    

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected {SERIAL_PORT}. Saving data from sensor")
        print("Ctrl+C for STOPPING and SAVE FILE")
    except Exception as e:
        print(f"Không thể mở {SERIAL_PORT}. Đảm bảo tắt Serial Monitor trong PlatformIO")
        exit()

    try:
        while True:
            # Nếu có dữ liệu gửi lên từ ESP32
            if ser.in_waiting > 0:
                try:
                    # Cố gắng đọc và giải mã. 
                    # Nếu có byte rác, nó sẽ nhảy thẳng xuống dòng except UnicodeDecodeError
                    line = ser.readline().decode('utf-8').strip()
                except UnicodeDecodeError:
                    # Gặp nhiễu thì bỏ qua luôn, quay lại từ đầu vòng lặp để đọc dòng tiếp theo
                    continue 
                
                # Kiểm tra xem có đúng là chuỗi JSON không
                if line.startswith("{") and line.endswith("}"):
                    try:
                        # Chuyển chuỗi văn bản thành Dictionary
                        raw_data = json.loads(line)
                        raw_temp = raw_data["nhiet_do"]
                        raw_hum = raw_data["do_am"]
                        raw_gas = raw_data["khi_gas"]
                        sensor_info_predicted.update(numPreProcessing.process_realtime_sensor(raw_temp, raw_hum, raw_gas, pre_gas))

                        # Bổ sung thêm nhãn thời gian (timestamp)
                        # raw_data["thoi_gian"] = time.strftime("%H:%M:%S")
                        
                        
                        print(f"Đã ghi nhận: {raw_data}")
                        
                    except json.JSONDecodeError:
                        # Nếu file JSON bị đứt đoạn giữa chừng, cũng skip luôn
                        pass 


    except KeyboardInterrupt as e:
        print(f"Be stopped because of Interupting: {e}")

