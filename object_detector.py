"""Object detection wrapper using Ultralytics YOLOv8.

This module replaces the older YOLOv4/tensorflow wrapper and provides a
drop-in ObjectDetectorAPI used by demo.py. It returns:
- result: BGR image with boxes drawn
- pred_bboxes: Nx6 numpy array: [cx, cy, w, h, score, class_id] with cx/cy/w/h normalized

If `ultralytics` is not installed, the class will print an instruction to
install it and return no detections.
"""

import time
import os
import cv2
import numpy as np
from ultralytics import YOLO

import torch


class ObjectDetectorAPI:
    def __init__(self, model_name: str = 'yolov8n.pt', conf: float = 0.4, device: str = None):
        self.conf = conf
        self.model = None
        self.device = device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if YOLO is None:
            print("Ultralytics YOLO (ultralytics) not found. Install with: pip install ultralytics")
            return

        try:
            # Initialize the YOLOv8 model. If the .pt file is not present it will be downloaded.
            self.model = YOLO(model_name)
            # set device in predict call since ultralytics lets you pass device there
            print(f"Loaded YOLOv8 model '{model_name}' on device '{self.device}'")
        except Exception as e:
            print(f"Failed to load YOLOv8 model '{model_name}': {e}")
            self.model = None

    def predict(self, image):
        """Run detection on an OpenCV BGR image.

        Returns (result_image_bgr, pred_bboxes)
        pred_bboxes is an Nx6 numpy array: [cx, cy, w, h, score, class]
        with cx,cy,w,h normalized to [0,1] relative to image width/height.
        """
        start_time = time.time()
        if self.model is None:
            return image, np.array([])

        # ultralytics accepts numpy arrays (RGB or BGR) — pass RGB for clarity
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        try:
            results = self.model(img_rgb, conf=self.conf, device=self.device, verbose=False)
        except Exception as e:
            print(f"YOLOv8 prediction failed: {e}")
            return image, np.array([])

        if len(results) == 0:
            return image, np.array([])

        res = results[0]
        boxes = getattr(res, 'boxes', None)
        pred_list = []
        out_img = image.copy()
        h, w = image.shape[:2]

        if boxes is not None and len(boxes) > 0:
            # boxes.xyxy, boxes.conf, boxes.cls
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else np.array(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else np.array(boxes.conf)
            cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else np.array(boxes.cls)

            for i in range(xyxy.shape[0]):
                x1, y1, x2, y2 = xyxy[i]
                score = float(confs[i])
                class_id = int(cls[i]) if len(cls) > 0 else -1

                cx = ((x1 + x2) / 2.0) / w
                cy = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / float(w)
                bh = (y2 - y1) / float(h)
                pred_list.append([cx, cy, bw, bh, score, class_id])

                # draw box on out_img
                p1 = (int(x1), int(y1))
                p2 = (int(x2), int(y2))
                cv2.rectangle(out_img, p1, p2, (0, 255, 0), 2)
                label = f"{class_id}:{score:.2f}"
                cv2.putText(out_img, label, (p1[0], max(p1[1] - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        exec_time = time.time() - start_time
        return out_img, np.array(pred_list)

