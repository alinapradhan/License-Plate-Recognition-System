"""
License Plate Detection Model using YOLOv8
==========================================

This module provides functionality for training and using YOLOv8 models
for license plate detection.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging
from ultralytics import YOLO
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LicensePlateDetector:
    """
    YOLOv8-based license plate detector with training and inference capabilities.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
        """
        Initialize the license plate detector.
        
        Args:
            model_path: Path to pretrained model. If None, uses YOLOv8n
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.device = self._get_device(device)
        self.model = None
        self.model_path = model_path
        self.load_model()
    
    def _get_device(self, device: str) -> str:
        """Get the appropriate device for inference."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def load_model(self) -> None:
        """Load or initialize the YOLO model."""
        try:
            if self.model_path and os.path.exists(self.model_path):
                logger.info(f"Loading model from {self.model_path}")
                self.model = YOLO(self.model_path)
            else:
                logger.info("Loading YOLOv8n base model")
                self.model = YOLO('yolov8n.pt')  # Start with nano version for speed
            
            # Move to appropriate device
            self.model.to(self.device)
            logger.info(f"Model loaded on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def train(self, 
              data_config: str, 
              epochs: int = 100, 
              img_size: int = 640,
              batch_size: int = 16,
              save_dir: str = 'runs/detect/train') -> str:
        """
        Train the YOLOv8 model on license plate data.
        
        Args:
            data_config: Path to YAML data configuration file
            epochs: Number of training epochs
            img_size: Image size for training
            batch_size: Batch size for training
            save_dir: Directory to save training results
            
        Returns:
            Path to the best trained model
        """
        try:
            logger.info("Starting model training...")
            results = self.model.train(
                data=data_config,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                project=save_dir,
                name='license_plate_detection',
                save=True,
                verbose=True
            )
            
            # Get the path to the best model
            best_model_path = os.path.join(save_dir, 'license_plate_detection', 'weights', 'best.pt')
            logger.info(f"Training completed. Best model saved at: {best_model_path}")
            
            # Load the best model
            self.model_path = best_model_path
            self.load_model()
            
            return best_model_path
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def detect(self, 
               image: np.ndarray, 
               confidence: float = 0.5,
               return_crops: bool = False) -> List[Dict[str, Any]]:
        """
        Detect license plates in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            confidence: Confidence threshold for detections
            return_crops: Whether to return cropped plate images
            
        Returns:
            List of detection results with bounding boxes and optional crops
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Run inference
            results = self.model(image, conf=confidence)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'class': 'license_plate'
                        }
                        
                        # Extract crop if requested
                        if return_crops:
                            x1, y1, x2, y2 = detection['bbox']
                            crop = image[y1:y2, x1:x2]
                            if crop.size > 0:
                                detection['crop'] = crop
                        
                        detections.append(detection)
            
            logger.info(f"Detected {len(detections)} license plates")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise
    
    def detect_batch(self, 
                    images: List[np.ndarray], 
                    confidence: float = 0.5) -> List[List[Dict[str, Any]]]:
        """
        Detect license plates in a batch of images.
        
        Args:
            images: List of input images
            confidence: Confidence threshold for detections
            
        Returns:
            List of detection results for each image
        """
        batch_results = []
        for image in images:
            detections = self.detect(image, confidence)
            batch_results.append(detections)
        
        return batch_results
    
    def export_model(self, 
                    format: str = 'onnx', 
                    export_path: Optional[str] = None) -> str:
        """
        Export the model to different formats for deployment.
        
        Args:
            format: Export format ('onnx', 'tflite', 'coreml', 'torchscript')
            export_path: Path to save exported model
            
        Returns:
            Path to exported model
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            logger.info(f"Exporting model to {format} format...")
            exported_path = self.model.export(format=format)
            
            if export_path:
                # Move to specified path
                os.rename(exported_path, export_path)
                exported_path = export_path
            
            logger.info(f"Model exported to: {exported_path}")
            return exported_path
            
        except Exception as e:
            logger.error(f"Model export failed: {e}")
            raise
    
    def validate(self, data_config: str) -> Dict[str, float]:
        """
        Validate the model on a dataset.
        
        Args:
            data_config: Path to validation dataset configuration
            
        Returns:
            Validation metrics
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            logger.info("Running model validation...")
            metrics = self.model.val(data=data_config)
            
            # Extract key metrics
            results = {
                'mAP50': float(metrics.box.map50),
                'mAP50-95': float(metrics.box.map),
                'precision': float(metrics.box.p),
                'recall': float(metrics.box.r)
            }
            
            logger.info(f"Validation results: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
    
    def draw_detections(self, 
                       image: np.ndarray, 
                       detections: List[Dict[str, Any]],
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2) -> np.ndarray:
        """
        Draw bounding boxes on image.
        
        Args:
            image: Input image
            detections: List of detections from detect()
            color: Color for bounding boxes (BGR)
            thickness: Line thickness
            
        Returns:
            Image with drawn bounding boxes
        """
        result_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw confidence score
            label = f"Plate: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_image


def create_data_config(train_path: str, 
                      val_path: str, 
                      save_path: str = 'data/license_plate_config.yaml') -> str:
    """
    Create a YAML configuration file for YOLO training.
    
    Args:
        train_path: Path to training images and labels
        val_path: Path to validation images and labels
        save_path: Path to save the configuration file
        
    Returns:
        Path to created configuration file
    """
    config_content = f"""# License Plate Detection Dataset Configuration
path: {os.path.abspath(os.path.dirname(train_path))}
train: {os.path.basename(train_path)}
val: {os.path.basename(val_path)}

# Classes
nc: 1  # number of classes
names: ['license_plate']  # class names
"""
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Write configuration
    with open(save_path, 'w') as f:
        f.write(config_content)
    
    logger.info(f"Data configuration saved to: {save_path}")
    return save_path


if __name__ == "__main__":
    # Example usage
    detector = LicensePlateDetector()
    
    # Load a test image (you would replace this with actual image loading)
    test_image = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Run detection
    detections = detector.detect(test_image, confidence=0.5)
    print(f"Found {len(detections)} license plates")