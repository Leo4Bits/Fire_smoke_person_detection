import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import torch


# CLAHE
def clahe_img_ret(frame, clip_limit=3.0, tileGrid_size=(8,8)):
    #Chuyển BGR sang LAB (Tách riêng độ sáng L và màu sắc A, B)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Khởi tạo CLAHE (clipLimit nên để 2.0 - 3.0, để 5.0 rất dễ bị nhiễu hạt)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tileGrid_size)

    #Chỉ áp dụng CLAHE lên kênh độ sáng L
    cl = clahe.apply(l)

    #Gộp kênh L đã làm nét với kênh màu sắc (a, b) gốc
    limg = cv2.merge((cl, a, b))

    #Chuyển ngược lại về BGR để đưa vào YOLO
    clahe_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return clahe_img


# SAHI
def declare_sahi(model_path,conf = 0.25, run_device = ""):
    if run_device == "":
        run_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    return AutoDetectionModel.from_pretrained(
        model_type="yolo11",
        model_path=model_path,
        confidence_threshold=conf,  
        device="cuda:0"
    )
    
def sahi_img_ret(frame,detection_model, overlap_height_ratio=0.2,overlap_width_ratio=0.2):
    width_frame ,height_frame = frame.shape[:2]

    return get_sliced_prediction(
        image=frame,
        detection_model=detection_model,
        slice_height=height_frame,
        slice_width=width_frame,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio
    )


