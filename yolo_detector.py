from ultralytics import YOLO
import cv2
import numpy as np
import torch
import logging

class YOLODetector:
    def __init__(self, model_path='yolo11n.pt', confidence_threshold=0.5, device='auto'):
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.class_names = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize model
        self.load_model(model_path)

    def load_model(self, model_path):
        try:
            # Load YOLO model
            self.model = YOLO(model_path)

            # Set device
            if self.device == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

            self.model.to(self.device)

            # Get class names
            self.class_names = self.model.names

            self.logger.info(f"YOLO model loaded successfully on {self.device}")
            self.logger.info(f"Model classes: {list(self.class_names.values())}")

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

    def filter_drone_classes(self, detections):
        # Filter for drone-related classes
        # You can customize this list based on your specific needs
        drone_classes = ['drone']  # Add more classes as needed

        filtered_detections = []
        for detection in detections:
            if detection['class_name'].lower() in [cls.lower() for cls in drone_classes]:
                filtered_detections.append(detection)

        return filtered_detections

    def set_confidence_threshold(self, threshold):
        self.confidence_threshold = threshold
        self.logger.info(f"Confidence threshold set to {threshold}")

    def detect_drone_simple(self, image):
        if self.model is None:
            return None

        try:
            # Run inference
            results = self.model(image, conf=self.confidence_threshold, verbose=False)

            # Look for drone detections
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class info
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.class_names[class_id].lower()

                        # Check if it's a drone-related object
                        if self._is_drone_class(class_name):
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                            # Calculate center point
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)

                            return center_x, center_y

            # No drone found
            return None

        except Exception as e:
            self.logger.error(f"Error during simple drone detection: {e}")
            return None

    def _is_drone_class(self, class_name):
        drone_keywords = [
            'drone', 'uav', 'quadcopter', 'aircraft', 'helicopter',
            'multirotor', 'copter', 'flying', 'aerial'
        ]

        class_name = class_name.lower()
        return any(keyword in class_name for keyword in drone_keywords)

    def get_fps_info(self):
        if self.model:
            return self.model.predictor.speed if hasattr(self.model, 'predictor') else None
        return None