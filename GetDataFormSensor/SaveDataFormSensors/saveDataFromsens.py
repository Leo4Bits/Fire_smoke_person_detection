import serial
import json
import time

# Cấu hình cổng COM
SERIAL_PORT = 'COM7'  
BAUD_RATE = 115200
JSON_FILE = 'sensor_data.json'

data_list = []

# Mở kết nối với ESP32
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected {SERIAL_PORT}. Saving data form sensor...")
    print("Ctrl+C for STOPPING and SAVE FILE\n")
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
                    data = json.loads(line)
                    
                    # Bổ sung thêm nhãn thời gian (timestamp)
                    data["thoi_gian"] = time.strftime("%H:%M:%S")
                    
                    data_list.append(data)
                    print(f"Đã ghi nhận: {data}")
                    
                except json.JSONDecodeError:
                    # Nếu file JSON bị đứt đoạn giữa chừng, cũng skip luôn
                    pass 

except KeyboardInterrupt:#nhấn Ctrl+C,sau exit chương trình sẽ tự động gom toàn bộ dữ liệu lưu thành file .json
    with open(JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)
    
    print(f"\nĐã lưu {len(data_list)} mẫu dữ liệu vào file '{JSON_FILE}'.")
    ser.close()