#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

# Real-time UAV detection using Intel RealSense camera

import os
import torch
import argparse
import torch.backends.cudnn as cudnn
import cv2
import time
import numpy as np
import pyrealsense2 as rs

from config import cfg
from backbone.tibnet import build_tibnet
from data.augmentations import to_chw_bgr


parser = argparse.ArgumentParser(description='TIB-Net Real-time Demo')
parser.add_argument('--weight', type=str, help='weight file', required=True)
parser.add_argument('--thresh', default=0.3, type=float,
                    help='Final confidence threshold')
parser.add_argument('--save_video', action='store_true',
                    help='Save output video')
parser.add_argument('--output_path', type=str, default='./result/realtime_output.avi',
                    help='Output video path')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()


def detect_frame(net, frame, thresh):
    """
    Detect drones in a single frame
    Returns: frame with detections drawn, list of detection info (bbox, center, score)
    """
    height, width, _ = frame.shape

    # Resize for inference
    max_im_shrink = np.sqrt(1700 * 1200 / (frame.shape[0] * frame.shape[1]))
    image = cv2.resize(frame, None, None, fx=max_im_shrink,
                       fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)

    # Prepare input tensor
    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]
    x = torch.from_numpy(x).unsqueeze(0)

    if use_cuda:
        x = x.cuda()

    # Run inference
    with torch.no_grad():
        t1 = time.time()
        detections = net(x).data
        t2 = time.time()

        scale = torch.Tensor([width, height, width, height]).to(detections.device)

        detection_info = []

        # Draw detections
        for i in range(detections.size(1)):
            j = 0
            while j < detections.size(2) and detections[0, i, j, 0] >= thresh:
                score = detections[0, i, j, 0]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                left_up = (int(pt[0]), int(pt[1]))
                right_bottom = (int(pt[2]), int(pt[3]))

                # Calculate center point
                center_x = int((pt[0] + pt[2]) / 2)
                center_y = int((pt[1] + pt[3]) / 2)
                center = (center_x, center_y)

                # Draw bounding box
                cv2.rectangle(frame, left_up, right_bottom, (0, 0, 255), 2)

                # Draw center point
                cv2.circle(frame, center, 5, (0, 255, 0), -1)
                cv2.circle(frame, center, 7, (255, 255, 255), 1)

                # Draw confidence score
                conf = "{:.3f}".format(score)
                point = (int(left_up[0]), int(left_up[1] - 5))
                cv2.putText(frame, conf, point,
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)

                detection_info.append({
                    'bbox': (left_up, right_bottom),
                    'center': center,
                    'score': float(score)
                })

                j += 1

        inference_time = t2 - t1

    return frame, detection_info, inference_time


def main():
    # Load model
    net = build_tibnet('test', cfg.NUM_CLASSES)
    weight_file = os.path.join(args.weight)
    print("Loading weight file from {}...".format(weight_file))
    net.load_weights(weight_file)
    net.eval()

    if use_cuda:
        net.cuda()
        cudnn.benchmark = True
        print("Using CUDA for inference")
    else:
        print("Using CPU for inference")

    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable color stream at 1280x720 @ 30fps
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    print("Starting RealSense camera...")
    try:
        pipeline.start(config)
    except RuntimeError as e:
        print(f"Error starting RealSense camera: {e}")
        print("Please make sure the RealSense camera is connected.")
        return

    # Video writer setup
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        video_writer = cv2.VideoWriter(args.output_path, fourcc, 30.0, (1280, 720))
        print(f"Saving video to {args.output_path}")

    print("Starting real-time detection... Press 'q' to quit.")

    frame_count = 0
    total_inference_time = 0

    try:
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # Convert to numpy array
            frame = np.asanyarray(color_frame.get_data())

            # Run detection
            frame_with_detections, detections, inference_time = detect_frame(net, frame, args.thresh)

            # Update stats
            frame_count += 1
            total_inference_time += inference_time
            avg_inference_time = total_inference_time / frame_count
            fps = 1.0 / inference_time if inference_time > 0 else 0

            # Display info on frame
            info_text = f"FPS: {fps:.1f} | Inference: {inference_time*1000:.1f}ms | Detections: {len(detections)}"
            cv2.putText(frame_with_detections, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_with_detections, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

            # Display detection centers info
            for idx, det in enumerate(detections):
                center_text = f"Drone {idx+1}: ({det['center'][0]}, {det['center'][1]})"
                cv2.putText(frame_with_detections, center_text, (10, 60 + idx*25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Show frame
            cv2.imshow('TIB-Net Real-time Drone Detection', frame_with_detections)

            # Save video if enabled
            if video_writer is not None:
                video_writer.write(frame_with_detections)

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Cleanup
        pipeline.stop()
        cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()

        print(f"\nStats:")
        print(f"Total frames processed: {frame_count}")
        print(f"Average inference time: {avg_inference_time*1000:.1f}ms")
        print(f"Average FPS: {1.0/avg_inference_time:.1f}")


if __name__ == '__main__':
    main()
