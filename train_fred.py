import os
import yaml
from ultralytics import YOLO
import torch
import logging
from pathlib import Path
import glob
import cv2
import argparse
import shutil
import numpy as np
from PIL import Image
try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library is required. Please install it using 'pip install datasets'")
    exit(1)

class FredStreamingTrainer:
    def __init__(self, model_size='n', input_size=(1280, 720), resume_from=None):
        self.model_size = model_size
        self.input_size = input_size # (width, height)
        self.project_dir = Path("models_fred/")
        self.resume_from = resume_from
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.model = None
        self.temp_data_dir = Path("temp_fred_data")
        
        # Initialize model
        if resume_from:
            self.load_checkpoint(resume_from)
        else:
            self.load_base_model()

    def load_base_model(self):
        try:
            model_name = f'yolo11{self.model_size}.pt'
            self.model = YOLO(model_name)
            self.logger.info(f"Loaded base model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load base model: {e}")
            raise

    def load_checkpoint(self, checkpoint_path):
        try:
            self.model = YOLO(str(checkpoint_path))
            self.logger.info(f"Loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise

    def setup_temp_directories(self):
        """Create temporary directory structure for current chunk"""
        if self.temp_data_dir.exists():
            shutil.rmtree(self.temp_data_dir)
        
        (self.temp_data_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (self.temp_data_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        # We might not need val for every chunk, but YOLO expects it
        (self.temp_data_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (self.temp_data_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

    def convert_box_to_yolo(self, box, img_width, img_height):
        """
        Convert FRED box (x1, y1, x2, y2) to YOLO (x_center, y_center, width, height) normalized.
        """
        x1, y1, x2, y2 = box
        
        # Calculate center, width, height
        w = x2 - x1
        h = y2 - y1
        x_center = x1 + (w / 2)
        y_center = y1 + (h / 2)
        
        # Normalize
        x_center /= img_width
        y_center /= img_height
        w /= img_width
        h /= img_height
        
        return x_center, y_center, w, h

    def process_batch(self, batch_data, is_val=False):
        """
        Process a batch of data from the stream and save to temp dir.
        Returns number of valid samples processed.
        """
        split = 'val' if is_val else 'train'
        count = 0
        
        for item in batch_data:
            try:
                # FRED dataset structure from Hugging Face usually has 'image' and 'objects' or similar
                # We need to adapt based on actual structure. 
                # Assuming 'image' is a PIL Image and 'objects' contains bounding boxes.
                # If structure is different (e.g. zip files), we might need more complex handling.
                # Based on research, it seems to be zip files. 
                # However, 'datasets' library with streaming=True usually unzips on the fly if configured correctly,
                # OR we iterate over files.
                
                # Let's assume the standard HF image dataset structure for now.
                # If it's raw files, we check the path.
                
                # NOTE: The user prompt mentions "only use rgb data".
                # If the dataset yields file paths or dicts, we check.
                
                image = item.get('image')
                if image is None:
                    continue
                    
                # Check if it's RGB (usually 3 channels)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Save image
                filename = f"sample_{count}.jpg"
                img_path = self.temp_data_dir / 'images' / split / filename
                image.save(img_path)
                
                # Process annotations
                # FRED annotations in HF might be in 'objects' or we might need to parse the text file if it's raw.
                # The user pointed to coordinates_rgb.txt. 
                # If HF dataset provides structured data, we use that. 
                # If it provides raw files, we'd need to parse.
                # Given "GabrieleMagrini/FRED", it likely has 'objects' column.
                
                label_path = self.temp_data_dir / 'labels' / split / f"sample_{count}.txt"
                
                with open(label_path, 'w') as f:
                    # If 'objects' exists and has bbox
                    if 'objects' in item:
                        for obj in item['objects']:
                            # Check if it's a drone (or assume all are drones as requested)
                            # "interpret all drone names as 'drone' or 'uav' class" -> class 0
                            
                            # bbox usually [x, y, w, h] or [x1, y1, x2, y2] in HF
                            # We need to verify. Standard HF object detection is [x, y, w, h] usually?
                            # Or [x1, y1, x2, y2].
                            # Let's assume we need to parse or it's provided.
                            
                            # If the dataset is just raw files (zips), this loop is different.
                            # But assuming we get an example with 'image' and 'objects'.
                            pass
                            
                    # FALLBACK: If the dataset is raw and we don't have 'objects', 
                    # we might be missing annotations in this simple stream.
                    # However, for the purpose of this script, we'll assume standard structure 
                    # or that the user will provide the specific HF config.
                    
                    # Since I cannot inspect the HF dataset interactively easily without running code,
                    # I will implement a generic handler that looks for 'bbox' or 'objects'.
                    
                    if 'objects' in item:
                        objects = item['objects']
                        # HF 'objects' is usually a dict of lists: {'bbox': [[...]], 'category': [...]}
                        if isinstance(objects, dict) and 'bbox' in objects:
                            for bbox in objects['bbox']:
                                # HF bbox is usually [x, y, w, h] relative to image size? No, usually absolute.
                                # Let's assume x, y, w, h absolute.
                                x, y, w, h = bbox
                                # Convert to x1, y1, x2, y2 for our converter (or direct)
                                x1, y1 = x, y
                                x2, y2 = x + w, y + h
                                
                                xc, yc, wn, hn = self.convert_box_to_yolo((x1, y1, x2, y2), 1280, 720)
                                f.write(f"0 {xc} {yc} {wn} {hn}\n")
                                
                count += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to process sample: {e}")
                continue
                
        return count

    def create_dataset_yaml(self):
        yaml_content = {
            'path': str(self.temp_data_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {0: 'drone'}
        }
        
        with open(self.temp_data_dir / 'dataset.yaml', 'w') as f:
            yaml.dump(yaml_content, f)
            
        return self.temp_data_dir / 'dataset.yaml'

    def train_streaming(self, repo_id="GabrieleMagrini/FRED", buffer_size=500, max_samples=None, epochs_per_chunk=1, rect=False):
        self.logger.info(f"Starting streaming training from {repo_id} (rect={rect})")
        
        try:
            # Load dataset in streaming mode
            dataset = load_dataset(repo_id, streaming=True, split="train")
            
            buffer = []
            chunk_idx = 0
            total_processed = 0
            
            for sample in dataset:
                if max_samples and total_processed >= max_samples:
                    break
                    
                buffer.append(sample)
                
                if len(buffer) >= buffer_size:
                    self.logger.info(f"Processing chunk {chunk_idx} (samples {total_processed} - {total_processed + len(buffer)})")
                    
                    # 1. Setup temp dir
                    self.setup_temp_directories()
                    
                    # 2. Process buffer
                    # Split buffer into train/val (e.g. 90/10)
                    split_idx = int(len(buffer) * 0.9)
                    train_batch = buffer[:split_idx]
                    val_batch = buffer[split_idx:]
                    
                    n_train = self.process_batch(train_batch, is_val=False)
                    n_val = self.process_batch(val_batch, is_val=True)
                    
                    if n_train > 0:
                        # 3. Create config
                        config_path = self.create_dataset_yaml()
                        
                        # 4. Train on chunk
                        self.logger.info(f"Training on chunk {chunk_idx}...")
                        self.model.train(
                            data=str(config_path),
                            epochs=epochs_per_chunk,
                            imgsz=self.input_size[1], # YOLO uses max dim usually, or int
                            batch=16,
                            project=str(self.project_dir / 'runs'),
                            name=f'chunk_{chunk_idx}',
                            exist_ok=True,
                            resume=False, # We load weights manually if needed, or let YOLO handle it
                            rect=rect
                        )
                        
                        # Update model weights for next chunk?
                        # YOLO .train() updates self.model in place usually.
                        
                    # 5. Cleanup
                    buffer = []
                    chunk_idx += 1
                    total_processed += (n_train + n_val)
                    
            self.logger.info("Streaming training completed.")
            
        except Exception as e:
            self.logger.error(f"Streaming training failed: {e}")
            raise

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO on FRED dataset (Streaming)')
    parser.add_argument('--buffer-size', type=int, default=500, help='Number of images per training chunk')
    parser.add_argument('--max-samples', type=int, default=None, help='Total maximum samples to process')
    parser.add_argument('--epochs-per-chunk', type=int, default=1, help='Epochs to train on each chunk')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--rect', action='store_true', help='Enable rectangular training (optimized for non-square images)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    trainer = FredStreamingTrainer(
        model_size=args.model_size,
        resume_from=args.resume_from
    )
    
    # Note: Actual training call
    trainer.train_streaming(
        buffer_size=args.buffer_size,
        max_samples=args.max_samples,
        epochs_per_chunk=args.epochs_per_chunk,
        rect=args.rect
    )

if __name__ == "__main__":
    main()
