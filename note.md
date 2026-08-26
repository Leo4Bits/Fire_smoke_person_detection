Trong SAHI:
    Danh sách các vật thể phát hiện được nằm trong thuộc tính: results.object_prediction_list

    Tọa độ box: obj.bbox.to_xyxy()

    Class ID: obj.category.id

    Độ tin cậy (Confidence): obj.score.value

Convert sang TensorRT
yolo export model="C:/Users/trann/Documents/NHAT_NAM_TRAN/WORK_SPACE/NGHIEN_CUU_KHOA_HOC/NCKH_MODEL_AI/BAO_CAO/TONG HOP KET QUA/yolo11l_40kimg/best.pt" format=engine half=True device=0