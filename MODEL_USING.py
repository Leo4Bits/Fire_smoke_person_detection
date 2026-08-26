import cv2
import numpy as np
import torch
import logging
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

class MODEL_YOLO:
    """Class xử lý dự đoán bounding box cơ bản qua YOLO"""
    def __init__(self, path: str):
        self.path = path
        self.model = YOLO(self.path)

    def yolo_img_res(self, frame): 
        """Trả về kết quả dự đoán của YOLO trên frame"""
        return self.model(frame)

    def boxes(self, frame):
        """Trả về danh sách boxes của frame hiện tại"""
        results = self.yolo_img_res(frame)
        return results[0].boxes

    @staticmethod
    def box_get_cls(box) -> int:
        "Trả về class(label) của đối tượng nhận diện được trong bounding box"
        return int(box.cls[0])

    @staticmethod
    def box_get_conf(box) -> float:
        "Trả về confidence của đối tượng nhận diện được trong bounding box"
        return float(box.conf[0])

    @staticmethod
    def box_get_xyxy(box) :
        "Trả về tọa độ của bounding box"
        return map(int, box.xyxy[0])

class MODEL_SAHI_METHOD:
    """Class hỗ trợ cắt slice và dự đoán qua SAHI"""
    def __init__(self, path: str, conf: float = 0.25, run_device: str = ""):
        self.path = path
        self.conf = conf
        self.run_device = run_device if run_device != "" else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.sahi_model = self.declare_sahi_method()

    def declare_sahi_method(self):
        return AutoDetectionModel.from_pretrained(
            model_type="yolo11", # model type này khác model_type mà các class mình khai báo
            model_path=self.path,
            confidence_threshold=self.conf,  
            device=self.run_device
        )

    def sahi_img_res(self, frame, overlap_height_ratio: float = 0.2, overlap_width_ratio: float = 0.2):
        height_frame, width_frame = frame.shape[:2]

        return get_sliced_prediction(
            image=frame,
            detection_model=self.sahi_model,
            slice_height=height_frame // 2,  # SAHI cần slice nhỏ hơn kích thước gốc để phát huy hiệu quả
            slice_width=width_frame // 2,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio
        )

    @staticmethod
    def box_get_cls(box):
        "Trả về class(label) của đối tượng nhận diện được trong bounding box"
        return int(box.category.id)
    @staticmethod
    def box_get_conf(box) -> float:
        "Trả về confidence của đối tượng nhận diện được trong bounding box"
        return float(box.score.value)

    @staticmethod
    def box_get_xyxy(box) :
        "Trả về tọa độ của bounding box"
        return map(int, box.bbox.to_xyxy())


class MODEL(MODEL_YOLO, MODEL_SAHI_METHOD):
    """Class tổng hợp kế thừa cả YOLO gốc và SAHI"""
    def __init__(self, path: str, conf: float = 0.25, run_device: str = "", model_type: str = "YOLO"):

        self.model_type = model_type.upper()
        if self.model_type == "SAHI":
            MODEL_SAHI_METHOD.__init__(self, path=path, conf=conf, run_device=run_device)
        elif self.model_type == "YOLO":
            MODEL_YOLO.__init__(self, path=path)
        else:
            raise ValueError(f"Không hỗ trợ model_type '{model_type}'. 'YOLO' hoặc 'SAHI'.")

    def get_boxes(self, frame):
        """Trả về danh sách boxes tùy theo model đang chạy"""
        if self.model_type == "SAHI":
            return self.sahi_img_res(frame).object_prediction_list
        return self.boxes(frame)

    def get_box_info(self, box):
        """Tự động bóc tách (cls, conf, (x1, y1, x2, y2)) cho cả YOLO và SAHI"""
        if self.model_type == "SAHI":
            cls = MODEL_SAHI_METHOD.box_get_cls(box)
            conf = MODEL_SAHI_METHOD.box_get_conf(box)
            x1, y1, x2, y2 = MODEL_SAHI_METHOD.box_get_xyxy(box)
        else:
            cls = MODEL_YOLO.box_get_cls(box)
            conf = MODEL_YOLO.box_get_conf(box)
            x1, y1, x2, y2 = MODEL_YOLO.box_get_xyxy(box)
        return cls, conf, (x1, y1, x2, y2)
