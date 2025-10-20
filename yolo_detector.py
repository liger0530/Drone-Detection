from ultralytics import YOLO
import cv2
import numpy as np
import torch
import logging
from pathlib import Path

class YOLODetector:
    def __init__(self, model_path='models/weights/best.pt', confidence_threshold=0.5, device='auto'):
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.class_names = None
        self.model_path = model_path

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize model
        self.load_model(model_path)

    def load_model(self, model_path):
        try:
            # Check if model file exists
            model_file = Path(model_path)
            if not model_file.exists():
                # Try common locations for trained models
                alternative_paths = [
                    Path('models/weights/best.pt'),
                    Path('models/weights/last.pt'),
                    Path(model_path)
                ]

                for alt_path in alternative_paths:
                    if alt_path.exists():
                        model_path = str(alt_path)
                        self.logger.info(f"Using model from: {model_path}")
                        break

            # Load YOLO model
            self.model = YOLO(model_path)
            self.model_path = model_path

            # Set device
            if self.device == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

            self.model.to(self.device)

            # Get class names
            self.class_names = self.model.names

            self.logger.info(f"YOLO model loaded successfully on {self.device}")
            self.logger.info(f"Model path: {model_path}")
            self.logger.info(f"Model classes: {list(self.class_names.values())}")
            self.logger.info(f"Confidence threshold: {self.confidence_threshold}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            return False

    def detect(self, image):
        if self.model is None:
            return []

        try:
            # Run inference
            results = self.model(image, conf=self.confidence_threshold, verbose=False)

            detections = []

            # Process results
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                        # Get confidence and class
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.class_names[class_id]

                        # Calculate center point for distance measurement
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)

                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': class_name,
                            'center': (center_x, center_y)
                        }

                        detections.append(detection)

            return detections

        except Exception as e:
            self.logger.error(f"Error during detection: {e}")
            return []

    def set_confidence_threshold(self, threshold):
        if not 0.0 <= threshold <= 1.0:
            self.logger.warning(f"Invalid threshold {threshold}. Must be between 0.0 and 1.0")
            return

        self.confidence_threshold = threshold
        self.logger.info(f"Confidence threshold set to {threshold}")

    def get_confidence_threshold(self):
        return self.confidence_threshold

    def get_fps_info(self):
        if self.model:
            return self.model.predictor.speed if hasattr(self.model, 'predictor') else None
        return None