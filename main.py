import cv2
import numpy as np
import time
import logging
from datetime import datetime
import json
import os
import argparse
from pathlib import Path

from camera_manager import RealSenseCamera
from yolo_detector import YOLODetector
from utils import FPSCounter, draw_detections, draw_fps, draw_detection_stats, create_detection_log_entry

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Drone Detection System using RealSense Camera and YOLO')

    # Model configuration
    parser.add_argument('--model', type=str, default='models/weights/best.pt',
                        help='Path to YOLO model file')

    # Camera configuration
    parser.add_argument('--width', type=int, default=1280,
                        help='Camera frame width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                        help='Camera frame height (default: 720)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Camera FPS (default: 30)')

    # Detection configuration
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold for detections (0.0-1.0, default: 0.5)')

    # Display and logging
    parser.add_argument('--no-display', action='store_true',
                        help='Disable video display window')
    parser.add_argument('--save-detections', action='store_true',
                        help='Save frames with detections')
    parser.add_argument('--log-detections', action='store_true',
                        help='Log detections to JSON file')

    # Device
    parser.add_argument('--device', type=str, default='cuda', choices=['auto', 'cpu', 'cuda'],
                        help='Device to run inference on (default: auto)')

    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Configuration from arguments
    CAMERA_WIDTH = args.width
    CAMERA_HEIGHT = args.height
    CAMERA_FPS = args.fps
    CONFIDENCE_THRESHOLD = args.confidence
    SHOW_DISPLAY = not args.no_display
    SAVE_DETECTIONS = args.save_detections
    LOG_DETECTIONS = args.log_detections
    MODEL_PATH = args.model
    DEVICE = args.device

    # Validate confidence threshold
    if not 0.0 <= CONFIDENCE_THRESHOLD <= 1.0:
        logger.error(f"Invalid confidence threshold: {CONFIDENCE_THRESHOLD}. Must be between 0.0 and 1.0")
        return

    logger.info(f"Camera config: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps")
    logger.info(f"Model: {MODEL_PATH}")
    logger.info(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    logger.info(f"Device: {DEVICE}")

    # Initialize components
    camera = RealSenseCamera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS)
    detector = YOLODetector(model_path=MODEL_PATH, confidence_threshold=CONFIDENCE_THRESHOLD, device=DEVICE)
    fps_counter = FPSCounter()

    # Initialize camera
    if not camera.initialize():
        logger.error("Failed to initialize camera")
        return

    logger.info("System initialized successfully")
    logger.info("\nControls:")
    logger.info("  'q' - Quit")
    logger.info("  's' - Save current frame")
    logger.info("  'c' - Toggle confidence threshold (0.3/0.5)")
    logger.info("  '+' - Increase confidence threshold by 0.05")
    logger.info("  '-' - Decrease confidence threshold by 0.05")
    logger.info("")

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

            inference_time = time.time() - start_time

            # Print all detection coordinates to terminal
            if detections:
                for i, det in enumerate(detections):
                    center = det['center']
                    confidence = det['confidence']
                    class_name = det['class_name']

                    # Print to terminal
                    print(f"{class_name.upper()} #{i+1} - Center: ({center[0]}, {center[1]}) | Confidence: {confidence:.2f}")

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
                # Add threshold info to display
                threshold_text = f"Confidence: {detector.get_confidence_threshold():.2f}"
                cv2.putText(annotated_frame, threshold_text, (15, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                cv2.imshow('Drone Detection', annotated_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quitting...")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"frame_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    logger.info(f"Frame saved as {filename}")
                elif key == ord('c'):
                    # Toggle confidence threshold
                    new_threshold = 0.3 if detector.get_confidence_threshold() == 0.5 else 0.5
                    detector.set_confidence_threshold(new_threshold)
                elif key == ord('+') or key == ord('='):
                    # Increase confidence threshold
                    current = detector.get_confidence_threshold()
                    new_threshold = min(1.0, current + 0.05)
                    detector.set_confidence_threshold(new_threshold)
                elif key == ord('-') or key == ord('_'):
                    # Decrease confidence threshold
                    current = detector.get_confidence_threshold()
                    new_threshold = max(0.0, current - 0.05)
                    detector.set_confidence_threshold(new_threshold)

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