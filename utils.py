import cv2
import numpy as np
import time
from typing import List, Dict, Tuple

class FPSCounter:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.frame_times = []
        self.last_time = time.time()

    def update(self):
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time

        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)

    def get_fps(self):
        if len(self.frame_times) == 0:
            return 0
        return len(self.frame_times) / sum(self.frame_times)

def draw_detections(image: np.ndarray, detections: List[Dict], depth_image: np.ndarray = None,
                   camera_manager=None) -> np.ndarray:
    annotated_image = image.copy()

    for detection in detections:
        bbox = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class_name']
        center = detection['center']

        # Draw bounding box
        color = get_class_color(detection['class_id'])
        cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # Prepare label text
        label = f"{class_name}: {confidence:.2f}"

        # Add distance if available
        if depth_image is not None and camera_manager is not None:
            distance = camera_manager.get_distance(center[0], center[1], depth_image)
            if distance is not None:
                label += f" | {distance:.2f}m"

        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(annotated_image,
                     (bbox[0], bbox[1] - label_size[1] - 10),
                     (bbox[0] + label_size[0], bbox[1]),
                     color, -1)

        # Draw label text
        cv2.putText(annotated_image, label,
                   (bbox[0], bbox[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw center point
        cv2.circle(annotated_image, center, 5, color, -1)

    return annotated_image

def get_class_color(class_id: int) -> Tuple[int, int, int]:
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
    ]
    return colors[class_id % len(colors)]

def draw_fps(image: np.ndarray, fps: float) -> np.ndarray:
    fps_text = f"FPS: {fps:.1f}"

    # Draw FPS background
    text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.rectangle(image, (10, 10), (10 + text_size[0] + 10, 10 + text_size[1] + 10), (0, 0, 0), -1)

    # Draw FPS text
    cv2.putText(image, fps_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return image

def draw_detection_stats(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    stats_text = f"Detections: {len(detections)}"

    # Position stats below FPS
    text_size = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(image, (10, 50), (10 + text_size[0] + 10, 50 + text_size[1] + 10), (0, 0, 0), -1)
    cv2.putText(image, stats_text, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return image

def resize_frame(frame: np.ndarray, target_width: int = 1280, target_height: int = 720) -> np.ndarray:
    if frame.shape[1] != target_width or frame.shape[0] != target_height:
        return cv2.resize(frame, (target_width, target_height))
    return frame

def save_detection_frame(image: np.ndarray, detections: List[Dict], timestamp: str,
                        output_dir: str = "detections") -> str:
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"detection_{timestamp}_{len(detections)}_objects.jpg"
    filepath = os.path.join(output_dir, filename)

    cv2.imwrite(filepath, image)
    return filepath

def create_detection_log_entry(detections: List[Dict], timestamp: str) -> Dict:
    return {
        'timestamp': timestamp,
        'detection_count': len(detections),
        'detections': [
            {
                'class_name': det['class_name'],
                'confidence': det['confidence'],
                'bbox': det['bbox'],
                'center': det['center']
            }
            for det in detections
        ]
    }