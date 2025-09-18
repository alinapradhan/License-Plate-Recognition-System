"""
Training script for license plate detection using YOLOv8.
"""
import os
from pathlib import Path
from ultralytics import YOLO
import yaml
import argparse


class LicensePlateTrainer:
    """Trainer for YOLOv8 license plate detection model."""
    
    def __init__(self, 
                 model_size: str = 'n',
                 pretrained: bool = True,
                 device: str = 'auto'):
        """
        Initialize trainer.
        
        Args:
            model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
            pretrained: Use pretrained weights
            device: Device to use ('auto', 'cpu', 'cuda:0', etc.)
        """
        self.model_size = model_size
        self.device = device
        
        # Initialize model
        if pretrained:
            model_path = f'yolov8{model_size}.pt'
        else:
            model_path = f'yolov8{model_size}.yaml'
            
        self.model = YOLO(model_path)
    
    def train(self, 
              dataset_yaml: str,
              epochs: int = 100,
              batch_size: int = 16,
              img_size: int = 640,
              lr0: float = 0.01,
              save_dir: str = 'runs/detect/train',
              **kwargs) -> str:
        """
        Train the model on license plate dataset.
        
        Args:
            dataset_yaml: Path to dataset YAML file
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Input image size
            lr0: Initial learning rate
            save_dir: Directory to save results
            **kwargs: Additional training arguments
            
        Returns:
            Path to trained model
        """
        # Verify dataset configuration
        if not Path(dataset_yaml).exists():
            raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")
        
        # Training arguments
        train_args = {
            'data': dataset_yaml,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'lr0': lr0,
            'device': self.device,
            'project': save_dir,
            'name': f'license_plate_yolov8{self.model_size}',
            'save': True,
            'save_period': 10,
            'val': True,
            'plots': True,
            'verbose': True,
            **kwargs
        }
        
        print(f"Starting training with YOLOv8{self.model_size}")
        print(f"Dataset: {dataset_yaml}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}, Image size: {img_size}")
        
        # Train model
        results = self.model.train(**train_args)
        
        # Get path to best model
        best_model_path = results.save_dir / 'weights' / 'best.pt'
        
        print(f"Training completed! Best model saved to: {best_model_path}")
        return str(best_model_path)
    
    def validate(self, 
                 model_path: str,
                 dataset_yaml: str,
                 img_size: int = 640,
                 batch_size: int = 32) -> dict:
        """
        Validate trained model on test set.
        
        Args:
            model_path: Path to trained model
            dataset_yaml: Path to dataset YAML file
            img_size: Input image size
            batch_size: Batch size for validation
            
        Returns:
            Validation metrics
        """
        # Load trained model
        model = YOLO(model_path)
        
        # Run validation
        results = model.val(
            data=dataset_yaml,
            imgsz=img_size,
            batch=batch_size,
            device=self.device
        )
        
        return results.results_dict
    
    def export_model(self,
                     model_path: str,
                     formats: list = ['onnx', 'tflite'],
                     img_size: int = 640) -> dict:
        """
        Export model to different formats for deployment.
        
        Args:
            model_path: Path to trained model
            formats: List of export formats
            img_size: Input image size
            
        Returns:
            Dictionary of exported model paths
        """
        model = YOLO(model_path)
        exported_paths = {}
        
        for format_type in formats:
            try:
                print(f"Exporting model to {format_type.upper()}...")
                export_path = model.export(
                    format=format_type,
                    imgsz=img_size,
                    optimize=True
                )
                exported_paths[format_type] = export_path
                print(f"✓ {format_type.upper()} model saved: {export_path}")
                
            except Exception as e:
                print(f"✗ Failed to export to {format_type}: {e}")
        
        return exported_paths


def create_training_config(output_path: str = "configs/training_config.yaml"):
    """Create a training configuration file with default hyperparameters."""
    config = {
        'model': {
            'size': 'n',  # n, s, m, l, x
            'pretrained': True
        },
        'training': {
            'epochs': 100,
            'batch_size': 16,
            'img_size': 640,
            'lr0': 0.01,
            'lrf': 0.1,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5
        },
        'augmentation': {
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
            'mixup': 0.0
        },
        'validation': {
            'conf': 0.25,
            'iou': 0.7,
            'max_det': 300
        }
    }
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Training configuration saved to: {output_path}")
    return output_path


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train YOLOv8 license plate detection model")
    
    # Dataset arguments
    parser.add_argument("--data", required=True, help="Path to dataset YAML file")
    parser.add_argument("--model-size", default="n", choices=['n', 's', 'm', 'l', 'x'],
                       help="YOLOv8 model size")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    
    # System arguments
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda:0, etc.)")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers")
    
    # Output arguments
    parser.add_argument("--project", default="runs/detect", help="Project directory")
    parser.add_argument("--name", default="license_plate_train", help="Experiment name")
    
    # Additional options
    parser.add_argument("--pretrained", action="store_true", default=True,
                       help="Use pretrained weights")
    parser.add_argument("--resume", type=str, help="Resume training from checkpoint")
    parser.add_argument("--config", type=str, help="Path to training configuration YAML")
    
    args = parser.parse_args()
    
    # Load additional config if provided
    config_overrides = {}
    if args.config:
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
            config_overrides.update(config_data.get('training', {}))
            config_overrides.update(config_data.get('augmentation', {}))
    
    # Initialize trainer
    trainer = LicensePlateTrainer(
        model_size=args.model_size,
        pretrained=args.pretrained,
        device=args.device
    )
    
    # Train model
    best_model_path = trainer.train(
        dataset_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr0=args.lr0,
        save_dir=args.project,
        workers=args.workers,
        resume=args.resume,
        **config_overrides
    )
    
    # Validate model
    print("\nRunning validation...")
    metrics = trainer.validate(
        model_path=best_model_path,
        dataset_yaml=args.data,
        img_size=args.img_size
    )
    
    print("\nValidation Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Export model for deployment
    print("\nExporting model for deployment...")
    exported_paths = trainer.export_model(
        model_path=best_model_path,
        formats=['onnx', 'tflite'],
        img_size=args.img_size
    )
    
    print("\n🎉 Training completed successfully!")
    print(f"Best model: {best_model_path}")
    
    for format_type, path in exported_paths.items():
        print(f"{format_type.upper()} model: {path}")


if __name__ == "__main__":
    main()