"""
Drone Tracking Data Extractor for ESP32 Gimbal Control
Reads from camera, runs YOLO detection, outputs tracking coordinates
Format: found,dx,dy,confidence

Author: Vision Team
For: Gimbal Motor Control Team
"""

import time
import sys
import cv2
import numpy as np
from ultralytics import YOLO
import torch

# ==================== CONFIGURATION ====================
# Model Configuration
MODEL_PATH = "models/weights/best.pt"              # Path to trained model from teammate
                                     # Change to "yolo11n.pt" for testing

# Camera Configuration
VIDEO_SOURCE = 0                     # 0 = default webcam, 1 = external camera
CAMERA_WIDTH = 1280                  # Camera resolution width
CAMERA_HEIGHT = 720                  # Camera resolution height
CAMERA_FPS = 30                      # Target FPS

# Detection Configuration
MIN_CONFIDENCE = 0.50                # Minimum detection confidence (0.0 - 1.0)

# Output Configuration
SERIAL_PORT = "COM3"                 # Serial port for ESP32
BAUD_RATE = 115200                   # Serial baud rate
SEND_HZ = 30                         # Data output rate (Hz)
ENABLE_SMOOTHING = True              # Enable EMA smoothing filter
SMOOTH_ALPHA = 0.25                  # Smoothing factor (lower = smoother)

# Display Configuration
SHOW_DISPLAY = True                  # Show visual feedback window
VERBOSE_OUTPUT = False               # Print extra debug info
# =======================================================

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[WARN] pyserial not installed - console output only")


class SmoothingFilter:
    """Exponential Moving Average (EMA) filter for smooth tracking"""
    def __init__(self, alpha=0.25):
        self.alpha = alpha if ENABLE_SMOOTHING else 0.0
        self.x = 0.0
        self.y = 0.0
        self.has_data = False
    
    def update(self, x, y):
        if self.alpha <= 0.0:
            return x, y
            
        if not self.has_data:
            self.x = x
            self.y = y
            self.has_data = True
        else:
            self.x = self.alpha * x + (1.0 - self.alpha) * self.x
            self.y = self.alpha * y + (1.0 - self.alpha) * self.y
        
        return self.x, self.y
    
    def reset(self):
        self.has_data = False


class SerialConnection:
    """Handles serial communication with ESP32"""
    def __init__(self, port, baud):
        self.ser = None
        if not SERIAL_AVAILABLE:
            return
        
        try:
            self.ser = serial.Serial(port, baudrate=baud, timeout=0)
            time.sleep(0.5)  # Allow ESP32 to reset
            print(f"[INFO] Serial connected: {port} @ {baud} baud")
        except Exception as e:
            print(f"[WARN] Could not open serial port: {e}")
            print("[INFO] Will output to console only")
    
    def send(self, data_string):
        """Send data via serial"""
        if self.ser and self.ser.writable():
            try:
                self.ser.write(data_string.encode('ascii'))
            except Exception as e:
                if VERBOSE_OUTPUT:
                    print(f"[WARN] Serial write error: {e}")
    
    def close(self):
        if self.ser:
            self.ser.close()


class DroneDetector:
    """YOLO-based drone detection"""
    def __init__(self, model_path):
        print(f"[INFO] Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        
        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"[INFO] Model loaded on: {self.device}")
        
        if VERBOSE_OUTPUT:
            print(f"[INFO] Model classes: {list(self.model.names.values())}")
    
    def find_best_drone(self, frame):
        """
        Detect drones in frame and return best match
        Returns: (x, y, confidence, class_name) or None
        """
        results = self.model(frame, conf=MIN_CONFIDENCE, verbose=False)
        
        if not results or len(results) == 0:
            return None
        
        boxes = getattr(results[0], 'boxes', None)
        if boxes is None or len(boxes) == 0:
            return None
        
        best_detection = None
        best_score = -1.0
        
        for box in boxes:
            conf = float(box.conf[0].cpu().numpy())
            
            if conf < MIN_CONFIDENCE:
                continue
            
            # Get class information
            cls_id = int(box.cls[0].cpu().numpy())
            class_name = self.model.names.get(cls_id, f"class_{cls_id}")
            
            # Check if this is a drone-related detection
            is_drone = self._is_drone_class(class_name)
            
            # Score: prioritize drone classes, then confidence
            score = (2.0 if is_drone else 1.0) * conf
            
            if score > best_score:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Calculate center point
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                
                best_detection = (center_x, center_y, conf, class_name)
                best_score = score
        
        return best_detection


def format_output(found, dx, dy, confidence):
    """
    Format tracking data for ESP32
    Output: found,dx,dy,confidence
    """
    return f"{1 if found else 0},{dx:.4f},{dy:.4f},{confidence:.4f}\n"


def draw_visualization(frame, detection, frame_center, smoothed_coords):
    """Draw tracking visualization on frame"""
    h, w = frame.shape[:2]
    cx, cy = frame_center
    
    # Draw center crosshair
    cross_size = 30
    cv2.line(frame, (int(cx - cross_size), int(cy)), 
             (int(cx + cross_size), int(cy)), (255, 255, 255), 2)
    cv2.line(frame, (int(cx), int(cy - cross_size)), 
             (int(cx), int(cy + cross_size)), (255, 255, 255), 2)
    cv2.circle(frame, (int(cx), int(cy)), 5, (255, 255, 255), -1)
    
    if detection:
        det_x, det_y, conf, class_name = detection
        
        # Draw detection point
        cv2.circle(frame, (int(det_x), int(det_y)), 12, (0, 255, 0), -1)
        cv2.circle(frame, (int(det_x), int(det_y)), 15, (0, 255, 0), 2)
        
        # Draw line from center to target
        cv2.line(frame, (int(cx), int(cy)), 
                 (int(det_x), int(det_y)), (0, 255, 0), 2)
        
        # Status text
        status = "LOCKED"
        status_color = (0, 255, 0)
        
        # Detection info
        info_text = f"{class_name} | Conf: {conf:.2f}"
        cv2.putText(frame, info_text, (int(det_x) + 20, int(det_y) - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Smoothed coordinates
        if smoothed_coords:
            dx_smooth, dy_smooth = smoothed_coords
            coord_text = f"dx:{dx_smooth:+.3f} dy:{dy_smooth:+.3f}"
            cv2.putText(frame, coord_text, (int(det_x) + 20, int(det_y) + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        status = "SEARCHING"
        status_color = (0, 0, 255)
    
    # Main status
    cv2.putText(frame, status, (20, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 1.8, status_color, 4)
    
    # Instructions
    cv2.putText(frame, "Press 'q' to quit | 's' to screenshot", 
               (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return frame


def main():
    print("=" * 70)
    print("  DRONE TRACKING DATA EXTRACTOR")
    print("  For ESP32 Gimbal Motor Control")
    print("=" * 70)
    print()
    
    # Initialize detector
    try:
        detector = DroneDetector(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        print("[TIP] Make sure model file exists and path is correct")
        return
    
    # Initialize camera
    print(f"[INFO] Opening camera: {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera: {VIDEO_SOURCE}")
        return
    
    # Set camera properties
    if isinstance(VIDEO_SOURCE, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    
    # Verify camera settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera resolution: {actual_width}x{actual_height}")
    
    # Initialize serial connection
    serial_conn = SerialConnection(SERIAL_PORT, BAUD_RATE)
    
    # Initialize smoothing filter
    smoother = SmoothingFilter(alpha=SMOOTH_ALPHA)
    
    # Timing control
    send_period = 1.0 / SEND_HZ
    next_send_time = time.perf_counter()
    
    print()
    print("[INFO] System ready!")
    print(f"[INFO] Output format: found,dx,dy,confidence")
    print(f"[INFO] Update rate: {SEND_HZ} Hz")
    print("-" * 70)
    print()
    
    frame_count = 0
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to capture frame")
                break
            
            h, w = frame.shape[:2]
            center_x = w / 2.0
            center_y = h / 2.0
            
            # Detect drone
            detection = detector.find_best_drone(frame)
            
            # Calculate tracking data
            found = False
            dx_out = 0.0
            dy_out = 0.0
            conf_out = 0.0
            smoothed_coords = None
            
            if detection:
                det_x, det_y, conf, class_name = detection
                
                # Calculate normalized offsets
                # Right = positive, Left = negative
                dx_raw = (det_x - center_x) / center_x
                # Up = positive, Down = negative  
                dy_raw = (center_y - det_y) / center_y
                
                # Clamp to valid range
                dx_raw = max(-1.0, min(1.0, dx_raw))
                dy_raw = max(-1.0, min(1.0, dy_raw))
                
                # Apply smoothing
                dx_smooth, dy_smooth = smoother.update(dx_raw, dy_raw)
                
                # Output values
                dx_out = dx_smooth
                dy_out = dy_smooth
                conf_out = conf
                found = True
                smoothed_coords = (dx_smooth, dy_smooth)
                
                if VERBOSE_OUTPUT:
                    print(f"[DETECT] {class_name} @ ({det_x:.0f},{det_y:.0f}) "
                          f"conf={conf:.2f} dx={dx_out:+.3f} dy={dy_out:+.3f}")
            else:
                smoother.reset()
            
            # Send data at fixed rate
            current_time = time.perf_counter()
            if current_time >= next_send_time:
                output_string = format_output(found, dx_out, dy_out, conf_out)
                
                # Send via serial
                serial_conn.send(output_string)
                
                # Print to console
                sys.stdout.write(output_string)
                sys.stdout.flush()
                
                next_send_time = current_time + send_period
            
            # Visualization
            if SHOW_DISPLAY:
                frame = draw_visualization(frame, detection, 
                                          (center_x, center_y), 
                                          smoothed_coords)
                
                cv2.imshow("Drone Tracker - Data Output", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[INFO] Quit command received")
                    break
                elif key == ord('s'):
                    filename = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"[INFO] Screenshot saved: {filename}")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C)")
    
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n[INFO] Shutting down...")
        
        # Send final "no target" signal
        final_output = format_output(False, 0, 0, 0)
        serial_conn.send(final_output)
        sys.stdout.write(final_output)
        
        cap.release()
        cv2.destroyAllWindows()
        serial_conn.close()
        
        print(f"[INFO] Processed {frame_count} frames")
        print("[INFO] Shutdown complete")


if __name__ == "__main__":
    main()