import pyrealsense2 as rs
import numpy as np
import cv2
import logging

class RealSenseCamera:
    def __init__(self, width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.config = None
        self.is_running = False

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def initialize(self):
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            # Configure color stream
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

            # Optional: Enable depth stream for distance estimation
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

            # Start streaming
            profile = self.pipeline.start(self.config)

            # Get device and sensors info
            device = profile.get_device()
            self.logger.info(f"Connected to RealSense device: {device.get_info(rs.camera_info.name)}")

            # Set auto-exposure for better performance
            color_sensor = device.first_color_sensor()
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, True)

            self.is_running = True
            self.logger.info(f"Camera initialized: {self.width}x{self.height} @ {self.fps}fps")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            return False

    def get_frame(self):
        if not self.is_running:
            return None, None

        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)

            # Get color frame
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame:
                return None, None

            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None

            return color_image, depth_image

        except Exception as e:
            self.logger.error(f"Error getting frame: {e}")
            return None, None

    def get_distance(self, x, y, depth_image):
        if depth_image is None:
            return None

        try:
            # Get distance in meters at pixel (x, y)
            depth_value = depth_image[y, x]
            distance = depth_value * 0.001  # Convert mm to meters
            return distance if distance > 0 else None
        except:
            return None

    def stop(self):
        if self.pipeline and self.is_running:
            self.pipeline.stop()
            self.is_running = False
            self.logger.info("Camera stopped")

    def __del__(self):
        self.stop()