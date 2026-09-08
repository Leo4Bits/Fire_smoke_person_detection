import cv2
import numpy as np
from ultralytics import YOLO
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

