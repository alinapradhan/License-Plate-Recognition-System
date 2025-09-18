"""
Data Preparation Script for Kaggle Car Plate Detection Dataset
==============================================================

Downloads, processes, and prepares the Kaggle Car Plate Detection dataset
for training YOLOv8 models.

Dataset: https://www.kaggle.com/datasets/andrewmvd/car-plate-detection
"""

import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
import argparse
from typing import List, Dict, Any, Tuple
import shutil
from sklearn.model_selection import train_test_split
import requests
import zipfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetPreparer:
    """
    Prepares the Kaggle Car Plate Detection dataset for YOLO training.
    """
    
    def __init__(self, data_dir: str = "data", train_split: float = 0.8):
        """
        Initialize dataset preparer.
        
        Args:
            data_dir: Base directory for data
            train_split: Fraction of data to use for training
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.train_split = train_split
        
        # Create directories
        self._create_directories()
        
        # Dataset info
        self.dataset_info = {
            "name": "Car License Plate Detection",
            "source": "https://www.kaggle.com/datasets/andrewmvd/car-plate-detection",
            "classes": ["license_plate"],
            "total_images": 0,
            "train_images": 0,
            "val_images": 0,
            "annotations": 0
        }
    
    def _create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.raw_dir,
            self.processed_dir,
            self.processed_dir / "images" / "train",
            self.processed_dir / "images" / "val",
            self.processed_dir / "labels" / "train",
            self.processed_dir / "labels" / "val",
            self.processed_dir / "annotations",
            self.data_dir / "models"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def download_dataset(self, kaggle_dataset: str = "andrewmvd/car-plate-detection") -> bool:
        """
        Download dataset from Kaggle.
        
        Note: This requires kaggle API credentials to be set up.
        Instructions: https://github.com/Kaggle/kaggle-api
        
        Args:
            kaggle_dataset: Kaggle dataset identifier
            
        Returns:
            True if download successful
        """
        try:
            import kaggle
            
            logger.info(f"Downloading dataset: {kaggle_dataset}")
            
            # Download dataset
            kaggle.api.dataset_download_files(
                kaggle_dataset,
                path=str(self.raw_dir),
                unzip=True
            )
            
            logger.info("Dataset download completed")
            return True
            
        except ImportError:
            logger.error("Kaggle API not installed. Install with: pip install kaggle")
            return False
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            logger.info("Manual download instructions:")
            logger.info(f"1. Go to https://www.kaggle.com/datasets/{kaggle_dataset}")
            logger.info("2. Click 'Download' button")
            logger.info(f"3. Extract files to: {self.raw_dir}")
            return False
    
    def parse_xml_annotation(self, xml_path: str) -> List[Dict[str, Any]]:
        """
        Parse Pascal VOC XML annotation file.
        
        Args:
            xml_path: Path to XML annotation file
            
        Returns:
            List of bounding box annotations
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            annotations = []
            
            # Get image dimensions
            size = root.find('size')
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                # Default dimensions if not found
                width = 640
                height = 480
                logger.warning(f"Image dimensions not found in {xml_path}, using defaults")
            
            # Parse objects
            for obj in root.findall('object'):
                name = obj.find('name').text
                
                # Get bounding box
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                # Convert to YOLO format (normalized coordinates)
                x_center = (xmin + xmax) / 2.0 / width
                y_center = (ymin + ymax) / 2.0 / height
                bbox_width = (xmax - xmin) / width
                bbox_height = (ymax - ymin) / height
                
                annotations.append({
                    'class': name,
                    'class_id': 0,  # license_plate is class 0
                    'bbox_pascal': [xmin, ymin, xmax, ymax],
                    'bbox_yolo': [x_center, y_center, bbox_width, bbox_height],
                    'image_width': width,
                    'image_height': height
                })
            
            return annotations
            
        except Exception as e:
            logger.error(f"Failed to parse XML annotation {xml_path}: {e}")
            return []
    
    def create_yolo_annotation(self, annotations: List[Dict[str, Any]], output_path: str) -> bool:
        """
        Create YOLO format annotation file.
        
        Args:
            annotations: List of bounding box annotations
            output_path: Path to save YOLO annotation file
            
        Returns:
            True if successful
        """
        try:
            with open(output_path, 'w') as f:
                for ann in annotations:
                    bbox_yolo = ann['bbox_yolo']
                    class_id = ann['class_id']
                    
                    # Format: class_id x_center y_center width height
                    f.write(f"{class_id} {bbox_yolo[0]:.6f} {bbox_yolo[1]:.6f} "
                           f"{bbox_yolo[2]:.6f} {bbox_yolo[3]:.6f}\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create YOLO annotation {output_path}: {e}")
            return False
    
    def visualize_annotations(self, image_path: str, annotations: List[Dict[str, Any]], 
                            output_path: str) -> bool:
        """
        Visualize annotations on image for verification.
        
        Args:
            image_path: Path to image file
            annotations: List of annotations
            output_path: Path to save visualized image
            
        Returns:
            True if successful
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Cannot load image: {image_path}")
                return False
            
            # Draw bounding boxes
            for ann in annotations:
                bbox = ann['bbox_pascal']
                x1, y1, x2, y2 = bbox
                
                # Draw rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{ann['class']}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save visualized image
            cv2.imwrite(output_path, image)
            return True
            
        except Exception as e:
            logger.error(f"Failed to visualize annotations: {e}")
            return False
    
    def process_dataset(self, visualize_samples: bool = True, sample_count: int = 10) -> bool:
        """
        Process the raw dataset into YOLO format.
        
        Args:
            visualize_samples: Whether to create visualization samples
            sample_count: Number of samples to visualize
            
        Returns:
            True if processing successful
        """
        try:
            logger.info("Processing dataset...")
            
            # Find all annotation files
            annotation_files = list(self.raw_dir.glob("**/*.xml"))
            
            if not annotation_files:
                logger.error("No XML annotation files found in raw directory")
                return False
            
            logger.info(f"Found {len(annotation_files)} annotation files")
            
            # Process each annotation
            processed_data = []
            valid_files = []
            
            for xml_file in annotation_files:
                # Find corresponding image file
                image_name = xml_file.stem
                
                # Try different image extensions
                image_file = None
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    potential_image = xml_file.parent / f"{image_name}{ext}"
                    if potential_image.exists():
                        image_file = potential_image
                        break
                
                if not image_file:
                    logger.warning(f"No image found for annotation: {xml_file}")
                    continue
                
                # Parse annotation
                annotations = self.parse_xml_annotation(str(xml_file))
                
                if not annotations:
                    logger.warning(f"No valid annotations in: {xml_file}")
                    continue
                
                processed_data.append({
                    'image_file': image_file,
                    'xml_file': xml_file,
                    'annotations': annotations,
                    'image_name': image_name
                })
                
                valid_files.append((str(image_file), str(xml_file)))
            
            if not processed_data:
                logger.error("No valid image-annotation pairs found")
                return False
            
            logger.info(f"Processing {len(processed_data)} valid image-annotation pairs")
            
            # Split into train and validation
            train_data, val_data = train_test_split(
                processed_data, 
                train_size=self.train_split, 
                random_state=42
            )
            
            logger.info(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")
            
            # Process training data
            self._process_split(train_data, "train")
            
            # Process validation data
            self._process_split(val_data, "val")
            
            # Create visualizations
            if visualize_samples:
                self._create_visualizations(train_data[:sample_count], "train")
                self._create_visualizations(val_data[:min(sample_count//2, len(val_data))], "val")
            
            # Update dataset info
            self.dataset_info.update({
                "total_images": len(processed_data),
                "train_images": len(train_data),
                "val_images": len(val_data),
                "annotations": sum(len(data['annotations']) for data in processed_data)
            })
            
            # Save dataset info
            self._save_dataset_info()
            
            # Create YOLO config file
            self._create_yolo_config()
            
            logger.info("Dataset processing completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Dataset processing failed: {e}")
            return False
    
    def _process_split(self, data: List[Dict[str, Any]], split: str) -> None:
        """Process a data split (train or val)."""
        for item in data:
            # Copy image
            image_dest = self.processed_dir / "images" / split / f"{item['image_name']}.jpg"
            
            # Load and save image (ensures consistent format)
            image = cv2.imread(str(item['image_file']))
            if image is not None:
                cv2.imwrite(str(image_dest), image)
            
            # Create YOLO annotation
            label_dest = self.processed_dir / "labels" / split / f"{item['image_name']}.txt"
            self.create_yolo_annotation(item['annotations'], str(label_dest))
    
    def _create_visualizations(self, data: List[Dict[str, Any]], split: str) -> None:
        """Create visualization samples."""
        viz_dir = self.processed_dir / "visualizations" / split
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        for i, item in enumerate(data):
            viz_path = viz_dir / f"{item['image_name']}_annotated.jpg"
            self.visualize_annotations(
                str(item['image_file']), 
                item['annotations'], 
                str(viz_path)
            )
    
    def _save_dataset_info(self) -> None:
        """Save dataset information."""
        info_file = self.processed_dir / "dataset_info.json"
        
        with open(info_file, 'w') as f:
            json.dump(self.dataset_info, f, indent=2)
        
        logger.info(f"Dataset info saved to: {info_file}")
    
    def _create_yolo_config(self) -> str:
        """Create YOLO dataset configuration file."""
        config_content = f"""# License Plate Detection Dataset Configuration
# Generated by dataset preparation script

# Dataset paths
path: {self.processed_dir.absolute()}
train: images/train
val: images/val

# Classes
nc: 1  # number of classes
names: ['license_plate']  # class names

# Dataset statistics
total_images: {self.dataset_info['total_images']}
train_images: {self.dataset_info['train_images']}
val_images: {self.dataset_info['val_images']}
total_annotations: {self.dataset_info['annotations']}
"""
        
        config_path = self.processed_dir / "dataset.yaml"
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        logger.info(f"YOLO config saved to: {config_path}")
        return str(config_path)
    
    def validate_dataset(self) -> Dict[str, Any]:
        """
        Validate the processed dataset.
        
        Returns:
            Validation report
        """
        report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        try:
            # Check directory structure
            required_dirs = [
                self.processed_dir / "images" / "train",
                self.processed_dir / "images" / "val", 
                self.processed_dir / "labels" / "train",
                self.processed_dir / "labels" / "val"
            ]
            
            for dir_path in required_dirs:
                if not dir_path.exists():
                    report["errors"].append(f"Missing directory: {dir_path}")
                    report["valid"] = False
            
            if not report["valid"]:
                return report
            
            # Count files
            train_images = list((self.processed_dir / "images" / "train").glob("*.jpg"))
            train_labels = list((self.processed_dir / "labels" / "train").glob("*.txt"))
            val_images = list((self.processed_dir / "images" / "val").glob("*.jpg"))
            val_labels = list((self.processed_dir / "labels" / "val").glob("*.txt"))
            
            # Check counts match
            if len(train_images) != len(train_labels):
                report["errors"].append(
                    f"Training images ({len(train_images)}) != labels ({len(train_labels)})"
                )
            
            if len(val_images) != len(val_labels):
                report["errors"].append(
                    f"Validation images ({len(val_images)}) != labels ({len(val_labels)})"
                )
            
            # Update statistics
            report["statistics"] = {
                "train_images": len(train_images),
                "train_labels": len(train_labels),
                "val_images": len(val_images),
                "val_labels": len(val_labels),
                "total_images": len(train_images) + len(val_images)
            }
            
            # Check for empty label files
            empty_labels = 0
            for label_file in train_labels + val_labels:
                if label_file.stat().st_size == 0:
                    empty_labels += 1
            
            if empty_labels > 0:
                report["warnings"].append(f"{empty_labels} empty label files found")
            
            report["statistics"]["empty_labels"] = empty_labels
            
            if report["errors"]:
                report["valid"] = False
            
            logger.info(f"Dataset validation: {'PASSED' if report['valid'] else 'FAILED'}")
            
        except Exception as e:
            report["valid"] = False
            report["errors"].append(f"Validation failed: {e}")
        
        return report
    
    def get_dataset_config_path(self) -> str:
        """Get path to YOLO dataset configuration file."""
        return str(self.processed_dir / "dataset.yaml")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Prepare Kaggle Car Plate Detection dataset")
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--download", action="store_true", help="Download dataset from Kaggle")
    parser.add_argument("--train-split", type=float, default=0.8, help="Training split ratio")
    parser.add_argument("--visualize", action="store_true", help="Create visualization samples")
    parser.add_argument("--sample-count", type=int, default=10, help="Number of visualization samples")
    
    args = parser.parse_args()
    
    # Initialize dataset preparer
    preparer = DatasetPreparer(args.data_dir, args.train_split)
    
    # Download dataset if requested
    if args.download:
        if not preparer.download_dataset():
            logger.error("Failed to download dataset. Please download manually.")
            return
    
    # Check if raw data exists
    if not any(preparer.raw_dir.glob("**/*.xml")):
        logger.error(f"No annotation files found in {preparer.raw_dir}")
        logger.info("Please download the dataset first or use --download flag")
        return
    
    # Process dataset
    if preparer.process_dataset(args.visualize, args.sample_count):
        # Validate processed dataset
        report = preparer.validate_dataset()
        
        if report["valid"]:
            logger.info("Dataset ready for training!")
            logger.info(f"Config file: {preparer.get_dataset_config_path()}")
            logger.info(f"Statistics: {report['statistics']}")
        else:
            logger.error("Dataset validation failed:")
            for error in report["errors"]:
                logger.error(f"  - {error}")
    
    else:
        logger.error("Dataset processing failed")


if __name__ == "__main__":
    main()