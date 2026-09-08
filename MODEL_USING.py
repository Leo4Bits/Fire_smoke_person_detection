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


class MODEL(MODEL_YOLO):
    """Class tổng hợp kế thừa cả YOLO gốc và SAHI"""
    def __init__(self, path: str, conf: float = 0.4, run_device: str = "", verbose: bool = True):
        self.verbose = verbose
        MODEL_YOLO.__init__(self, path=path, verbose=verbose)
        

    def get_boxes(self, frame, verbose: bool = None):
        """Trả về danh sách boxes tùy theo model đang chạy"""
        return self.get_yolo_boxes(frame, verbose=verbose)

    def get_box_info(self, box):
        """Tự động bóc tách (cls, conf, (x1, y1, x2, y2)) cho cả YOLO và SAHI"""
    
        cls = MODEL_YOLO.box_get_cls(box)
        conf = MODEL_YOLO.box_get_conf(box)
        x1, y1, x2, y2 = MODEL_YOLO.box_get_xyxy(box)
        return cls, conf, (x1, y1, x2, y2)