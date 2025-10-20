import os
import yaml
from ultralytics import YOLO
import torch
import logging
from pathlib import Path
import glob
import cv2
import argparse
import gc

class DroneYOLOTrainer:
    def __init__(self, model_size='n', input_size=(1280, 720), resume_from=None):
        self.model_size = model_size
        self.input_size = input_size
        self.project_dir = Path("models/")
        self.resume_from = resume_from

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        self.model = None

        # Load model (either resume from checkpoint or load base model)
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
            checkpoint_path = Path(checkpoint_path)

            if not checkpoint_path.exists():
                # Try common locations
                alternative_paths = [
                    self.project_dir / 'weights' / 'last.pt',
                    self.project_dir / 'weights' / 'best.pt',
                    Path(checkpoint_path)
                ]

                for alt_path in alternative_paths:
                    if alt_path.exists():
                        checkpoint_path = alt_path
                        break
                else:
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            self.model = YOLO(str(checkpoint_path))
            self.logger.info(f"Loaded checkpoint for resume training: {checkpoint_path}")
            self.logger.info("Training will resume from this checkpoint")

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise

    def find_latest_checkpoint(self):
        """Find the latest training checkpoint"""
        # Look for last.pt (most recent checkpoint)
        last_pt = self.project_dir / 'weights' / 'last.pt'
        if last_pt.exists():
            return last_pt

        return None

    def verify_dataset(self, dataset_path):
        """Verify that dataset has been properly converted"""
        dataset_path = Path(dataset_path)

        # Check for required directories
        required_dirs = [
            dataset_path / 'images' / 'train',
            dataset_path / 'images' / 'val',
            dataset_path / 'labels' / 'train',
            dataset_path / 'labels' / 'val'
        ]

        for dir_path in required_dirs:
            if not dir_path.exists():
                raise FileNotFoundError(f"Required directory not found: {dir_path}")

        # Count files
        train_images = len(list((dataset_path / 'images' / 'train').glob('*.jpg')))
        train_labels = len(list((dataset_path / 'labels' / 'train').glob('*.txt')))
        val_images = len(list((dataset_path / 'images' / 'val').glob('*.jpg')))
        val_labels = len(list((dataset_path / 'labels' / 'val').glob('*.txt')))

        self.logger.info(f"Dataset verification:")
        self.logger.info(f"  Train: {train_images} images, {train_labels} labels")
        self.logger.info(f"  Val: {val_images} images, {val_labels} labels")

        if train_images == 0 or val_images == 0:
            raise ValueError("Dataset appears to be empty. Please run dataset_converter.py first.")

        return True

    def setup_training_directories(self):
        """Create necessary training directories"""
        self.project_dir.mkdir(exist_ok=True)

        runs_dir = self.project_dir / 'runs'
        runs_dir.mkdir(exist_ok=True)

        self.logger.info("Training directories setup complete")

    def train(self, dataset_path, epochs=100, batch_size=16, patience=50, save_period=10,
              resume=False, workers=4, cache=None, amp=True, single_cls=False):
        """
        Train the YOLO model on drone detection dataset.

        Args:
            dataset_path: Path to dataset directory
            epochs: Number of training epochs
            batch_size: Batch size for training
            patience: Early stopping patience
            save_period: Save checkpoint every N epochs
            resume: If True, automatically resume from last checkpoint
            workers: Number of dataloader workers (reduce to save memory)
            cache: Cache images ('ram', 'disk', or None)
            amp: Use Automatic Mixed Precision (reduces memory usage)
            single_cls: Train as single-class detector
        """
        try:
            self.setup_training_directories()

            # Verify dataset structure
            dataset_path = Path(dataset_path)
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

            self.verify_dataset(dataset_path)

            config_path = dataset_path / 'dataset.yaml'
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Dataset config not found: {config_path}\n"
                )

            # Check if resuming from checkpoint
            if resume and not self.resume_from:
                latest_checkpoint = self.find_latest_checkpoint()
                if latest_checkpoint:
                    self.logger.info(f"Found checkpoint to resume from: {latest_checkpoint}")
                    self.load_checkpoint(latest_checkpoint)
                else:
                    self.logger.warning("Resume requested but no checkpoint found. Starting fresh training.")

            # Log memory optimization settings
            self.logger.info(f"Memory settings: workers={workers}, cache={cache}, amp={amp}, single_cls={single_cls}")

            train_args = {
                'data': str(config_path),
                'epochs': epochs,
                'imgsz': self.input_size,
                'batch': batch_size,
                'patience': patience,
                'save_period': save_period,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'workers': workers, 
                'project': str(self.project_dir / 'runs'),
                'name': 'drone_detection',
                'exist_ok': True,
                'pretrained': True if not self.resume_from else False,
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
                'copy_paste': 0.0,
                'amp': amp,  # Configurable AMP
                'single_cls': single_cls  # Single-class optimization
            }

            # Add cache if specified
            if cache:
                train_args['cache'] = cache
                self.logger.info(f"Caching images to {cache}")
            else:
                train_args['cache'] = False

            # Add resume flag if resuming from checkpoint
            if self.resume_from:
                train_args['resume'] = True
                self.logger.info("Resuming training from checkpoint...")

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
                dataset_path = Path(dataset_path)
                config_path = dataset_path / 'dataset.yaml'
                if not config_path.exists():
                    raise FileNotFoundError(f"Dataset config not found: {config_path}")
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

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO11 for Drone Detection')

    # Model configuration
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLO model size (n=nano, s=small, m=medium, l=large, x=xlarge). Default: n')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience in epochs (default: 50)')
    parser.add_argument('--save-period', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')

    # Dataset
    parser.add_argument('--dataset', type=str, default='data/',
                        help='Path to dataset directory (default: data/)')

    # Resume training
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume training from specific checkpoint path')

    # Image size
    parser.add_argument('--width', type=int, default=1280,
                        help='Input image width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                        help='Input image height (default: 720)')

    # Memory optimization
    parser.add_argument('--cache', type=str, default=None, choices=['ram', 'disk', None],
                        help='Cache images in RAM or disk for faster training (uses more memory)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of dataloader workers (default: 4, reduce if memory issues)')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use Automatic Mixed Precision (AMP) to reduce memory (default: True)')
    parser.add_argument('--no-amp', dest='amp', action='store_false',
                        help='Disable AMP (uses more memory but may be more stable)')
    parser.add_argument('--single-cls', action='store_true',
                        help='Train as single-class (slight memory reduction)')
    
    parser.add_argument('--skip-export', action='store_true',
                        help='Skip ONNX export after training')

    return parser.parse_args()

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def main():
    args = parse_args()

    # Clear GPU memory before starting
    clear_gpu_memory()

    # Display configuration
    print(f"\nConfiguration:")
    print(f"  Model size: YOLO11{args.model_size}")
    print(f"  Input size: {args.width}x{args.height}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Patience: {args.patience}")

    # Memory optimization info
    print(f"\nMemory Optimization:")
    print(f"  Workers: {args.workers}")
    print(f"  Cache: {args.cache if args.cache else 'Disabled'}")
    print(f"  AMP (Mixed Precision): {'Enabled' if args.amp else 'Disabled'}")
    print(f"  Single-class mode: {'Yes' if args.single_cls else 'No'}")

    if args.resume:
        print(f"\n  Mode: Resume training from last checkpoint")
    elif args.resume_from:
        print(f"\n  Mode: Resume from {args.resume_from}")
    else:
        print(f"\n  Mode: Fresh training")

    print()

    # Initialize trainer
    trainer = DroneYOLOTrainer(
        model_size=args.model_size,
        input_size=(args.width, args.height),
        resume_from=args.resume_from
    )

    try:
        print("\nStarting training...\n")

        results = trainer.train(
            dataset_path=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            save_period=args.save_period,
            resume=args.resume,
            workers=args.workers,
            cache=args.cache,
            amp=args.amp,
            single_cls=args.single_cls
        )

        # Find the best model in the runs directory
        runs_dir = Path(trainer.project_dir) / 'runs' / 'drone_detection'
        best_model_path = runs_dir / 'weights' / 'best.pt'

        if not best_model_path.exists():
            print("\nWarning: best.pt not found, looking for last.pt")
            best_model_path = runs_dir / 'weights' / 'last.pt'

        if best_model_path.exists():
            print(f"Model saved at: {best_model_path}")

            # Export to ONNX unless skipped
            if not args.skip_export:
                print("\nExporting model to ONNX format...")
                try:
                    exported_path = trainer.export_model(str(best_model_path), 'onnx')
                    print(f"Model exported to: {exported_path}")
                except Exception as e:
                    print(f"Warning: Export failed: {e}")
        else:
            print("\nWarning: Trained model not found")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    except Exception as e:
        print(f"\nTraining failed: {e}")
        raise

if __name__ == "__main__":
    main()