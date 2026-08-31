import cv2
import numpy as np
import torch
import logging
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

class MODEL_YOLO:
    """Class xử lý dự đoán bounding box cơ bản qua YOLO"""
    def __init__(self, path: str, verbose: bool = True):
        self.path = path
        self.verbose = verbose
        self.model = YOLO(self.path)

    def yolo_img_res(self, frame, verbose: bool = None): 
        """Trả về kết quả dự đoán của YOLO trên frame"""
        is_verbose = self.verbose if verbose is None else verbose
        return self.model(frame, verbose=is_verbose)

    def get_yolo_boxes(self, frame, verbose: bool = None):
        """Trả về danh sách boxes của frame hiện tại"""
        results = self.yolo_img_res(frame, verbose=verbose)
        return results[0].boxes

    @staticmethod
    def box_get_cls(box) -> int:
        return int(box.cls[0])

    @staticmethod
    def box_get_conf(box) -> float:
        return float(box.conf[0])

    @staticmethod
    def box_get_xyxy(box):
        return map(int, box.xyxy[0])

class MODEL_SAHI_METHOD:
    """Class hỗ trợ cắt slice và dự đoán qua SAHI"""
    def __init__(self, path: str, conf: float = 0.25, run_device: str = "", verbose: bool = True):
        self.path = path
        self.conf = conf
        self.verbose = verbose
        self.run_device = run_device if run_device != "" else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.sahi_model = self.declare_sahi_method()

    def declare_sahi_method(self):
        return AutoDetectionModel.from_pretrained(
            model_type="yolo11", # model type này khác model_type mà các class mình khai báo
            model_path=self.path,
            confidence_threshold=self.conf,  
            device=self.run_device
        )

    def sahi_img_res(self, frame, overlap_height_ratio: float = 0.2, overlap_width_ratio: float = 0.2, verbose: bool = None):
        height_frame, width_frame = frame.shape[:2]
        is_verbose = self.verbose if verbose is None else verbose

        return get_sliced_prediction(
            image=frame,
            detection_model=self.sahi_model,
            slice_height=height_frame // 2,  # SAHI cần slice nhỏ hơn kích thước gốc để phát huy hiệu quả
            slice_width=width_frame // 2,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            verbose=is_verbose
        )

    @staticmethod
    def box_get_cls(box):
        return int(box.category.id)

    @staticmethod
    def box_get_conf(box) -> float:
        return float(box.score.value)

    @staticmethod
    def box_get_xyxy(box):
        return map(int, box.bbox.to_xyxy())

class MODEL(MODEL_YOLO, MODEL_SAHI_METHOD):
    """Class tổng hợp kế thừa cả YOLO gốc và SAHI"""
    def __init__(self, path: str, conf: float = 0.25, run_device: str = "", model_type: str = "YOLO", verbose: bool = True):
        self.model_type = model_type.upper()
        self.verbose = verbose
        if self.model_type == "SAHI":
            MODEL_SAHI_METHOD.__init__(self, path=path, conf=conf, run_device=run_device, verbose=verbose)
        elif self.model_type == "YOLO":
            MODEL_YOLO.__init__(self, path=path, verbose=verbose)
        else:
            raise ValueError(f"Không hỗ trợ model_type '{model_type}'. Chỉ nhận 'YOLO' hoặc 'SAHI'.")

    def get_boxes(self, frame, verbose: bool = None):
        """Trả về danh sách boxes tùy theo model đang chạy"""
        if self.model_type == "SAHI":
            return self.sahi_img_res(frame, verbose=verbose).object_prediction_list
        return self.get_yolo_boxes(frame, verbose=verbose)

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