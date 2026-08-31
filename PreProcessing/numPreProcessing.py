import math
import statistics
from collections import deque
import joblib
import numpy as np

model = joblib.load(r"C:\Users\trann\Documents\NHAT_NAM_TRAN\WORK_SPACE\NGHIEN_CUU_KHOA_HOC\NCKH_MODEL_AI\BAO_CAO\TONG HOP KET QUA\random_forest\random_forest_air_quality.pkl")

WINDOW_SIZE = 10
buf_gas = deque(maxlen=WINDOW_SIZE)
pre_gas = 0

def rolling_n_mean(buffer:deque, cur_num:float) -> float:
    '''Trung bình n mẫu'''
    buffer.append(cur_num)
    return statistics.mean(buffer)

def gas_diff(pre_num:float, cur_num:float) -> float:
    '''Trả về độ lệch giữa hai mẫu liên tiếp: 
    - Mẫu hiện tại: cur_num
    - Mẫu ngay trước nó: pre_num'''
    return cur_num - pre_num

LABEL_NAMES = {0: 'Bình thường (Normal)', 1: 'Khói (Smoke)', 2: 'Lửa (Fire)'}

# Hàm xử lý khi nhận 1 gói dữ liệu realtime (hiện tại là Serial - sẽ đổi qua socket hoặc MQTT)
def process_realtime_sensor(raw_temp, raw_hum, raw_gas, pre_raw_gas) -> dict:
    '''Hàm nhận đầu vô là 3 thông số, lần lượt 2 thông số đầu từ DHT11 và 1 thông số sau từ MQ135:
    - Biến rolling_10_mean_gas và biến diff_gas được tính trong đây
    - model Random forrest predict và trả về target, probability(xác xuất)

    -> Đầu ra là một dictionary chứa thông tin sau predict'''
    # Khử nhiễu qua Rolling Mean
    rolling_10_mean_gas = rolling_n_mean(buf_gas,raw_gas)
    diff_gas = gas_diff(pre_raw_gas,raw_gas)

    # Đưa về dạng ma trận 2D: [[nhiet_do, do_am, khi_gas,gas_diff,rolling_10_mean_gas]]
    features = np.array([[raw_temp, raw_hum, raw_gas,diff_gas,rolling_10_mean_gas]])

    # Dự đoán nhãn và xác suất
    pred_label = model.predict(features)[0]
    pred_proba = model.predict_proba(features)[0]

    return {
        'status': LABEL_NAMES[pred_label],
        'label': int(pred_label),
        'confidence': float(pred_proba[pred_label]),
        'smoothed_data': {
            'temp': round(raw_temp, 1),
            'hum': round(raw_hum, 1),
            'gas': round(raw_gas, 1),
            'diff_gas': round(diff_gas,1)
        },
    }