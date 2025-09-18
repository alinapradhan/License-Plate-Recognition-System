"""
Training Script for License Plate Detection Model
=================================================

Trains YOLOv8 model on the prepared car license plate dataset.
"""

import argparse
import os
import yaml
from pathlib import Path
import logging
from datetime import datetime
import json
import sys

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.plate_detector import LicensePlateDetector, create_data_config
from scripts.prepare_dataset import DatasetPreparer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Handles training of license plate detection models.
    """
    
    def __init__(self, 
                 config_file: str,
                 output_dir: str = "runs/train",
                 model_size: str = "n"):
        """
        Initialize model trainer.
        
        Args:
            config_file: Path to dataset YAML configuration
            output_dir: Directory to save training outputs
            model_size: YOLOv8 model size (n, s, m, l, x)
        """
        self.config_file = config_file
        self.output_dir = Path(output_dir)
        self.model_size = model_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training configuration
        self.training_config = {
            "epochs": 100,
            "batch_size": 16,
            "img_size": 640,
            "learning_rate": 0.01,
            "patience": 20,
            "save_period": 10,
            "workers": 8,
            "device": "auto"
        }
        
        # Initialize detector
        base_model = f"yolov8{model_size}.pt"
        self.detector = LicensePlateDetector(model_path=None)
        logger.info(f"Initialized with {base_model}")
    
    def update_training_config(self, **kwargs) -> None:
        """Update training configuration parameters."""
        self.training_config.update(kwargs)
        logger.info(f"Updated training config: {kwargs}")
    
    def validate_dataset_config(self) -> bool:
        """Validate dataset configuration file."""
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            required_keys = ['path', 'train', 'val', 'nc', 'names']
            for key in required_keys:
                if key not in config:
                    logger.error(f"Missing key in dataset config: {key}")
                    return False
            
            # Check if paths exist
            base_path = Path(config['path'])
            train_path = base_path / config['train']
            val_path = base_path / config['val']
            
            if not train_path.exists():
                logger.error(f"Training path not found: {train_path}")
                return False
            
            if not val_path.exists():
                logger.error(f"Validation path not found: {val_path}")
                return False
            
            logger.info("Dataset configuration validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate dataset config: {e}")
            return False
    
    def train_model(self) -> str:
        """
        Train the license plate detection model.
        
        Returns:
            Path to the best trained model
        """
        try:
            logger.info("Starting model training...")
            
            # Create run directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = self.output_dir / f"license_plate_{timestamp}"
            
            # Train model
            best_model_path = self.detector.train(
                data_config=self.config_file,
                epochs=self.training_config["epochs"],
                img_size=self.training_config["img_size"],
                batch_size=self.training_config["batch_size"],
                save_dir=str(run_dir)
            )
            
            # Save training configuration
            config_path = Path(best_model_path).parent.parent / "training_config.json"
            with open(config_path, 'w') as f:
                json.dump(self.training_config, f, indent=2)
            
            logger.info(f"Training completed. Best model: {best_model_path}")
            return best_model_path
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def validate_model(self, model_path: str) -> dict:
        """
        Validate trained model on test set.
        
        Args:
            model_path: Path to trained model
            
        Returns:
            Validation metrics
        """
        try:
            logger.info("Validating trained model...")
            
            # Load trained model
            trained_detector = LicensePlateDetector(model_path)
            
            # Run validation
            metrics = trained_detector.validate(self.config_file)
            
            logger.info(f"Validation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return {}
    
    def export_model(self, model_path: str, formats: list = None) -> dict:
        """
        Export model to different formats for deployment.
        
        Args:
            model_path: Path to trained model
            formats: List of export formats
            
        Returns:
            Dictionary of exported model paths
        """
        if formats is None:
            formats = ['onnx', 'tflite']
        
        exported_models = {}
        
        try:
            # Load trained model
            trained_detector = LicensePlateDetector(model_path)
            
            model_dir = Path(model_path).parent
            
            for format_name in formats:
                try:
                    logger.info(f"Exporting to {format_name}...")
                    
                    export_path = model_dir / f"license_plate_model.{format_name}"
                    exported_path = trained_detector.export_model(
                        format=format_name,
                        export_path=str(export_path)
                    )
                    
                    exported_models[format_name] = exported_path
                    logger.info(f"Exported {format_name} model: {exported_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to export to {format_name}: {e}")
            
            return exported_models
            
        except Exception as e:
            logger.error(f"Model export failed: {e}")
            return {}
    
    def create_training_summary(self, 
                               model_path: str, 
                               metrics: dict, 
                               exported_models: dict) -> str:
        """
        Create a training summary report.
        
        Args:
            model_path: Path to trained model
            metrics: Validation metrics
            exported_models: Exported model paths
            
        Returns:
            Path to summary report
        """
        try:
            model_dir = Path(model_path).parent
            summary_path = model_dir / "training_summary.json"
            
            summary = {
                "training_completed": datetime.now().isoformat(),
                "model_path": str(model_path),
                "model_size": self.model_size,
                "dataset_config": self.config_file,
                "training_config": self.training_config,
                "validation_metrics": metrics,
                "exported_models": exported_models,
                "model_info": {
                    "framework": "YOLOv8",
                    "task": "license_plate_detection",
                    "classes": ["license_plate"],
                    "input_size": self.training_config["img_size"]
                }
            }
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Training summary saved: {summary_path}")
            return str(summary_path)
            
        except Exception as e:
            logger.error(f"Failed to create training summary: {e}")
            return ""
    
    def run_complete_training(self, export_formats: list = None) -> str:
        """
        Run complete training pipeline.
        
        Args:
            export_formats: List of formats to export model
            
        Returns:
            Path to training summary
        """
        if export_formats is None:
            export_formats = ['onnx', 'tflite']
        
        try:
            # Validate dataset
            if not self.validate_dataset_config():
                raise ValueError("Dataset configuration validation failed")
            
            # Train model
            model_path = self.train_model()
            
            # Validate model
            metrics = self.validate_model(model_path)
            
            # Export model
            exported_models = self.export_model(model_path, export_formats)
            
            # Create summary
            summary_path = self.create_training_summary(model_path, metrics, exported_models)
            
            logger.info("Complete training pipeline finished successfully!")
            logger.info(f"Best model: {model_path}")
            logger.info(f"Summary: {summary_path}")
            
            return summary_path
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise


def main():
    """Main function for command-line training."""
    parser = argparse.ArgumentParser(description="Train license plate detection model")
    
    # Dataset arguments
    parser.add_argument("--config", required=True, help="Path to dataset YAML config")
    parser.add_argument("--output-dir", default="runs/train", help="Output directory")
    
    # Model arguments
    parser.add_argument("--model-size", default="n", choices=['n', 's', 'm', 'l', 'x'],
                       help="YOLOv8 model size")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    
    # Export arguments
    parser.add_argument("--export", nargs='+', default=['onnx', 'tflite'],
                       help="Export formats (onnx, tflite, coreml, etc.)")
    
    # Other arguments
    parser.add_argument("--device", default="auto", help="Training device (auto, cpu, cuda)")
    parser.add_argument("--workers", type=int, default=8, help="Number of data loader workers")
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        
        # Try to prepare dataset if config not found
        logger.info("Attempting to prepare dataset...")
        preparer = DatasetPreparer()
        
        if preparer.process_dataset():
            args.config = preparer.get_dataset_config_path()
            logger.info(f"Using prepared dataset config: {args.config}")
        else:
            logger.error("Dataset preparation failed. Please prepare dataset first.")
            return
    
    # Initialize trainer
    trainer = ModelTrainer(
        config_file=args.config,
        output_dir=args.output_dir,
        model_size=args.model_size
    )
    
    # Update training configuration
    trainer.update_training_config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        learning_rate=args.lr,
        patience=args.patience,
        device=args.device,
        workers=args.workers
    )
    
    try:
        # Run complete training pipeline
        summary_path = trainer.run_complete_training(args.export)
        
        # Print final summary
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Training summary: {summary_path}")
        print(f"Dataset config: {args.config}")
        print(f"Model size: {args.model_size}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())