import cv2
import numpy as np
from ultralytics import YOLO
from PreProcessing import imgPreProcessing,numPreProcessing,outputGUI
from MODEL_USING import *
from GetDataFromSensor.threadGetData import thread
import threading


# ____________________MODEL YOLO
path_model_yolo = r"C:\Users\trann\Documents\NHAT_NAM_TRAN\WORK_SPACE\NGHIEN_CUU_KHOA_HOC\NCKH_MODEL_AI\BAO_CAO\TONG HOP KET QUA\yolo11l_40kimg\best.pt"
path_model_onnx = r"C:\Users\trann\Documents\NHAT_NAM_TRAN\WORK_SPACE\NGHIEN_CUU_KHOA_HOC\NCKH_MODEL_AI\BAO_CAO\TONG HOP KET QUA\yolo11l_40kimg\best.onnx"
cap = cv2.VideoCapture(0)
detection_model =  MODEL(path_model_yolo,verbose=False)
class_names = {0:"Person", 1: 'Fire', 2: 'Smoke'}
# ____________________THREAD + RANDOMFOREST
thread_running = threading.Thread(target=thread.serial_reader_thread, daemon=True)
thread_running.start()
# ____________________
# CẤU HÌNH BỘ LỌC THỜI GIAN (TEMPORAL FILTER)
MAX_PATIENCE = 100  # Số lượng frame tối đa sẽ giữ lại khung hình cũ nếu model bị trượt
patience_counters = {}  # Lưu số frame còn lại cho từng class
saved_boxes = {}        # Lưu tọa độ cũ của từng class

while cap.isOpened():
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not success: break

    # tien xu ly anh voi CLAHE
    clahe_img = imgPreProcessing.clahe_img_ret(frame,1,(8,8))

    detected_classes_this_frame = set()
    boxes = detection_model.get_boxes(clahe_img)  # Lấy danh sách boxes của frame hiện tại

    for box in boxes:
        cls, conf_img, (x1,y1,x2,y2) = MODEL.get_box_info(detection_model, box)

        # nếu detect được, cập nhật tọa độ mới nhất và reset bộ đếm
        detected_classes_this_frame.add(cls)
        saved_boxes[cls] = (x1, y1, x2, y2, conf_img)
        patience_counters[cls] = MAX_PATIENCE
    #__ Lấy thông tin từ UART
    rf_label = thread.sensor_info_predicted["status"]
    rf_conf = thread.sensor_info_predicted["confidence"]
    # VẼ KHUNG HÌNH (Kết hợp cả hàng mới detect và hàng cũ đang được "giữ")
    for cls in list(patience_counters.keys()):
        # Nếu frame này model bị trượt class này, trừ đi 1 patience
        if cls not in detected_classes_this_frame:
            patience_counters[cls] -= 1
        
        # Nếu vẫn còn trong thời gian chờ (counter > 0), tiến hành vẽ
        if patience_counters[cls] > 0:
            x1, y1, x2, y2, conf_img = saved_boxes[cls]
            label_name = class_names.get(cls, f"Unknown {cls}")
            color = (0, 0, 255) if cls == 0 else (255, 0, 0)
            
            # Vẽ Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if (cls == 1 or cls == 2):
                try:
                    x_center = int((x1+x2)/2)
                    y_center = int((y1+y2)/2)
                    cv2.circle(frame, (x_center,y_center),4, (0,0,255), -1)
                except Exception as e:
                    print(f"Error drawing circle: {e}")
                    print(f"x_center: {x_center}")
                    print(f"y_center: {y_center}")
            
            # Thêm chữ (ghi chú thêm chữ [Hold] nếu đang dùng khung hình cũ để dễ theo dõi)
            status = "" if cls in detected_classes_this_frame else " [Hold]"
            text_yolo = outputGUI.output_text_yolo(label_name,conf_img)
            cv2.putText(frame, text_yolo, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if label_name != 0:
                text_yolo = outputGUI.output_text_rf(rf_label,rf_conf)
                cv2.putText(frame, text_yolo, (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            # Hết thời gian kiên nhẫn mà vẫn không thấy lại -> Xóa bỏ hoàn toàn
            patience_counters.pop(cls, None)
            saved_boxes.pop(cls, None)
    
    cv2.imshow("Smoothed YOLOv8 Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()