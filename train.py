import os
import yaml
from ultralytics import YOLO
import torch
import logging
from pathlib import Path
import glob
import cv2

class DroneYOLOTrainer:
    def __init__(self, model_size='n', input_size=(1280, 720)):
        self.model_size = model_size
        self.input_size = input_size
        self.project_dir = "models/"

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self.model = None
        self.load_base_model()

    def load_base_model(self):
        try:
            model_name = f'yolo11{self.model_size}.pt'
            self.model = YOLO(model_name)
            self.logger.info(f"Loaded base model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load base model: {e}")
            raise

    def convert_bbox_labels(self, dataset_path):
        dataset_path = Path(dataset_path)
        labels_train = dataset_path / 'labels' / 'train'
        labels_val = dataset_path / 'labels' / 'val'
        images_train = dataset_path / 'images' / 'train'
        images_val = dataset_path / 'images' / 'val'

        if labels_train.exists():
            self._convert_labels_in_directory(labels_train, images_train)

        if labels_val.exists():
            self._convert_labels_in_directory(labels_val, images_val)

        self.logger.info("Label conversion completed")

    def _convert_labels_in_directory(self, labels_dir, images_dir):
        label_files = glob.glob(str(labels_dir / "*.txt"))

        for label_file in label_files:
            try:
                base_name = Path(label_file).stem
                image_file = None

                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    potential_image = images_dir / f"{base_name}{ext}"
                    if potential_image.exists():
                        image_file = potential_image
                        break

                if image_file is None:
                    self.logger.warning(f"No corresponding image found for {label_file}")
                    continue

                img = cv2.imread(str(image_file))
                if img is None:
                    self.logger.warning(f"Could not read image: {image_file}")
                    continue

                img_height, img_width = img.shape[:2]

                with open(label_file, 'r') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]

                converted_labels = []
                for i in range(0, len(lines), 4):
                    if i + 3 < len(lines):
                        try:
                            x1 = float(lines[i])
                            y1 = float(lines[i + 1])
                            x2 = float(lines[i + 2])
                            y2 = float(lines[i + 3])

                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            width = abs(x2 - x1)
                            height = abs(y2 - y1)

                            center_x_norm = center_x / img_width
                            center_y_norm = center_y / img_height
                            width_norm = width / img_width
                            height_norm = height / img_height

                            center_x_norm = max(0, min(1, center_x_norm))
                            center_y_norm = max(0, min(1, center_y_norm))
                            width_norm = max(0, min(1, width_norm))
                            height_norm = max(0, min(1, height_norm))

                            yolo_line = f"0 {center_x_norm:.6f} {center_y_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
                            converted_labels.append(yolo_line)

                        except ValueError as e:
                            self.logger.warning(f"Error converting coordinates in {label_file}: {e}")

                with open(label_file, 'w') as f:
                    f.write('\n'.join(converted_labels))
                    if converted_labels:
                        f.write('\n')

                if converted_labels:
                    self.logger.debug(f"Converted {len(converted_labels)} bounding boxes in {label_file}")

            except Exception as e:
                self.logger.error(f"Error processing {label_file}: {e}")

    def create_dataset_config(self, dataset_path):
        config = {
            'path': str(Path(dataset_path).absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/val', 
            'nc': 1,
            'names': ['drone']
        }

        config_path = self.project_dir / 'drone_dataset.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        self.logger.info(f"Dataset config created: {config_path}")
        return config_path

    def setup_training_directories(self):
        dirs = ['runs', 'datasets', 'models']
        for dir_name in dirs:
            dir_path = self.project_dir / dir_name
            dir_path.mkdir(exist_ok=True)

        self.logger.info("Training directories setup complete")

    def train(self, dataset_path, epochs=100, batch_size=16, patience=50, save_period=10):
        try:
            self.setup_training_directories()

            self.logger.info("Converting bounding box labels to YOLO format...")
            self.convert_bbox_labels(dataset_path)

            config_path = self.create_dataset_config(dataset_path)

            if not Path(dataset_path).exists():
                raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

            train_args = {
                'data': str(config_path),
                'epochs': epochs,
                'imgsz': self.input_size, 
                'batch': batch_size,
                'patience': patience,
                'save_period': save_period,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'workers': 4,
                'project': str(self.project_dir / 'runs'),
                'name': 'drone_detection',
                'exist_ok': True,
                'pretrained': True,
                'optimizer': 'AdamW',
                'lr0': 0.01,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'pose': 12.0,
                'kobj': 1.0,
                'label_smoothing': 0.0,
                'nbs': 64,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.0,
                'copy_paste': 0.0
            }

            results = self.model.train(**train_args)

            self.logger.info("Training completed successfully!")
            return results

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

    def validate(self, model_path=None, dataset_path=None):
        try:
            if model_path:
                model = YOLO(model_path)
            else:
                model = self.model

            if dataset_path:
                config_path = self.create_dataset_config(dataset_path)
                results = model.val(data=str(config_path))
            else:
                results = model.val()

            self.logger.info("Validation completed")
            return results

        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            raise

    def export_model(self, model_path, export_format='onnx'):
        try:
            model = YOLO(model_path)
            exported_path = model.export(format=export_format)
            self.logger.info(f"Model exported to: {exported_path}")
            return exported_path
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            raise

def main():
    trainer = DroneYOLOTrainer(model_size='n', input_size=(1280, 720))

    dataset_path = "data/" 
    epochs = 100
    batch_size = 16

    try:
        print("Starting drone detection model training...")
        results = trainer.train(
            dataset_path=dataset_path,
            epochs=epochs,
            batch_size=batch_size,
            patience=50,
            save_period=10
        )

        best_model_path = Path(trainer.project_dir) / 'best.pt'

        print("Running validation...")
        val_results = trainer.validate(model_path=str(best_model_path), dataset_path=dataset_path)

        print("Exporting model to ONNX format")
        exported_path = trainer.export_model(str(best_model_path), 'onnx')
        print(f"Model exported to: {exported_path}")

    except Exception as e:
        print(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()