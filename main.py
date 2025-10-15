import cv2
import numpy as np
import time
import logging
from datetime import datetime
import json
import os

from camera_manager import RealSenseCamera
from yolo_detector import YOLODetector
from utils import FPSCounter, draw_detections, draw_fps, draw_detection_stats, create_detection_log_entry

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Configuration
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 30
    CONFIDENCE_THRESHOLD = 0.5
    SHOW_DISPLAY = True
    SAVE_DETECTIONS = False
    LOG_DETECTIONS = False

    logger.info("Starting Drone Detection System")
    logger.info(f"Camera config: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps")

    # Initialize components
    camera = RealSenseCamera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS)
    detector = YOLODetector(confidence_threshold=CONFIDENCE_THRESHOLD)
    fps_counter = FPSCounter()

    # Initialize camera
    if not camera.initialize():
        logger.error("Failed to initialize camera")
        return

    logger.info("System initialized successfully")

    # Detection log
    detection_log = []

    try:
        while True:
            # Get frame from camera
            color_frame, depth_frame = camera.get_frame()

            if color_frame is None:
                logger.warning("No frame received from camera")
                continue

            # Update FPS counter
            fps_counter.update()
            current_fps = fps_counter.get_fps()

            # Run YOLO detection
            start_time = time.time()
            detections = detector.detect(color_frame)

            # Get simplified drone detection output
            drone_coords = detector.detect_drone_simple(color_frame)

            # Filter for drone-related objects (optional)
            # detections = detector.filter_drone_classes(detections)

            inference_time = time.time() - start_time

            # Print drone coordinates to terminal if detected
            if drone_coords is not None:
                center_x, center_y = drone_coords
                print(f"Drone coordinates - X: {center_x}, Y: {center_y}")

            # Draw visualizations
            annotated_frame = draw_detections(color_frame, detections, depth_frame, camera)
            annotated_frame = draw_fps(annotated_frame, current_fps)
            annotated_frame = draw_detection_stats(annotated_frame, detections)

            # Log detections if enabled
            if LOG_DETECTIONS and detections:
                timestamp = datetime.now().isoformat()
                log_entry = create_detection_log_entry(detections, timestamp)
                detection_log.append(log_entry)

                # Print detection info
                logger.info(f"Detected {len(detections)} objects - Inference: {inference_time*1000:.1f}ms")

            # Save detection frames if enabled
            if SAVE_DETECTIONS and detections:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                from utils import save_detection_frame
                save_detection_frame(annotated_frame, detections, timestamp)

            # Display frame
            if SHOW_DISPLAY:
                cv2.imshow('Drone Detection - RealSense + YOLO', annotated_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"frame_{timestamp}.jpg", annotated_frame)
                    logger.info(f"Frame saved as frame_{timestamp}.jpg")
                elif key == ord('c'):
                    # Toggle confidence threshold
                    new_threshold = 0.3 if detector.confidence_threshold == 0.5 else 0.5
                    detector.set_confidence_threshold(new_threshold)
                    logger.info(f"Confidence threshold changed to {new_threshold}")

            # Performance monitoring
            if int(time.time()) % 10 == 0:  # Every 10 seconds
                logger.info(f"Performance - FPS: {current_fps:.1f}, Inference: {inference_time*1000:.1f}ms")

    except KeyboardInterrupt:
        logger.info("Stopping detection system...")

    except Exception as e:
        logger.error(f"Error in main loop: {e}")

    finally:
        # Cleanup
        camera.stop()
        cv2.destroyAllWindows()

        # Save detection log if we have entries
        if detection_log:
            log_filename = f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_filename, 'w') as f:
                json.dump(detection_log, f, indent=2)
            logger.info(f"Detection log saved to {log_filename}")

        logger.info("System shutdown complete")

if __name__ == "__main__":
    main()