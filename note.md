Trong SAHI:
    Danh sách các vật thể phát hiện được nằm trong thuộc tính: results.object_prediction_list

    Tọa độ box: obj.bbox.to_xyxy()

    Class ID: obj.category.id

    Độ tin cậy (Confidence): obj.score.value

Convert sang TensorRT
yolo export model="C:/Users/trann/Documents/NHAT_NAM_TRAN/WORK_SPACE/NGHIEN_CUU_KHOA_HOC/NCKH_MODEL_AI/BAO_CAO/TONG HOP KET QUA/yolo11l_40kimg/best.pt" format=engine half=True device=0


temp (Nhiệt độ)

hum (Độ ẩm)

mq (Khí gas thô)

gas_diff (Độ lệch khí gas so với mẫu trước)

rolling_mean_10 (Trung bình trượt 10 mẫu để chống nhiễu)


Đọc data chạy trên thread ngầm:
- Đọc Serial trên một Thread ngầm (Background Thread) với cơ chế bộ đệm Rolling Mean deque. Khi kết hợp xác suất (confidence) của cả YOLO và Random Forest (từ Serial), nếu cả hai đều vượt ngưỡng 0.5, hệ thống sẽ kích hoạt cảnh báo.