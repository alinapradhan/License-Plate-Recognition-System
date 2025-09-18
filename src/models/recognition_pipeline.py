"""
License Plate Recognition Pipeline
=================================

Main pipeline that combines plate detection and OCR for complete
license plate recognition functionality.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import json
import os

from .plate_detector import LicensePlateDetector
from .plate_ocr import LicensePlateOCR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LicensePlateRecognitionPipeline:
    """
    Complete pipeline for license plate detection and recognition.
    """
    
    def __init__(self, 
                 detector_model_path: Optional[str] = None,
                 detection_confidence: float = 0.5,
                 ocr_confidence: float = 0.3,
                 ocr_languages: List[str] = ['en'],
                 use_gpu: bool = True):
        """
        Initialize the recognition pipeline.
        
        Args:
            detector_model_path: Path to trained detection model
            detection_confidence: Confidence threshold for detection
            ocr_confidence: Confidence threshold for OCR
            ocr_languages: Languages for OCR
            use_gpu: Whether to use GPU acceleration
        """
        self.detection_confidence = detection_confidence
        self.ocr_confidence = ocr_confidence
        
        # Initialize detector
        logger.info("Initializing license plate detector...")
        self.detector = LicensePlateDetector(
            model_path=detector_model_path,
            device='auto' if use_gpu else 'cpu'
        )
        
        # Initialize OCR
        logger.info("Initializing OCR system...")
        self.ocr = LicensePlateOCR(
            languages=ocr_languages,
            gpu=use_gpu
        )
        
        logger.info("Pipeline initialization complete")
    
    def process_image(self, 
                     image: np.ndarray,
                     return_intermediate: bool = False) -> List[Dict[str, Any]]:
        """
        Process a single image for license plate recognition.
        
        Args:
            image: Input image (BGR format)
            return_intermediate: Whether to return intermediate processing results
            
        Returns:
            List of recognition results with plates, text, and metadata
        """
        results = []
        
        try:
            # Step 1: Detect license plates
            logger.debug("Running plate detection...")
            detections = self.detector.detect(
                image, 
                confidence=self.detection_confidence,
                return_crops=True
            )
            
            if not detections:
                logger.info("No license plates detected")
                return results
            
            # Step 2: Process each detected plate
            for i, detection in enumerate(detections):
                logger.debug(f"Processing plate {i+1}/{len(detections)}")
                
                result = {
                    'plate_id': i,
                    'detection_bbox': detection['bbox'],
                    'detection_confidence': detection['confidence'],
                    'timestamp': datetime.now().isoformat(),
                    'text': None,
                    'text_confidence': 0.0,
                    'valid_plate': False
                }
                
                # Add intermediate results if requested
                if return_intermediate:
                    result['crop_image'] = detection.get('crop')
                
                # Step 3: Extract text if crop available
                if 'crop' in detection and detection['crop'].size > 0:
                    try:
                        # Extract text from the cropped plate
                        extracted_text = self.ocr.extract_best_text(
                            detection['crop'], 
                            min_confidence=self.ocr_confidence
                        )
                        
                        if extracted_text:
                            result['text'] = extracted_text
                            result['valid_plate'] = self.ocr.validate_license_plate(extracted_text)
                            
                            # Get detailed OCR results for confidence score
                            detailed_ocr = self.ocr.extract_text(
                                detection['crop'],
                                confidence_threshold=self.ocr_confidence
                            )
                            
                            if detailed_ocr:
                                # Use the highest confidence from detailed results
                                result['text_confidence'] = max(
                                    r['confidence'] for r in detailed_ocr
                                )
                            
                            logger.info(f"Extracted text: '{extracted_text}' (valid: {result['valid_plate']})")
                        else:
                            logger.info("No valid text extracted from plate")
                            
                    except Exception as e:
                        logger.error(f"OCR failed for plate {i}: {e}")
                        result['error'] = str(e)
                
                results.append(result)
            
            logger.info(f"Processed {len(results)} plates, {sum(1 for r in results if r['valid_plate'])} valid")
            return results
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise
    
    def process_video_frame(self, 
                           frame: np.ndarray,
                           frame_number: int = 0) -> Dict[str, Any]:
        """
        Process a single video frame.
        
        Args:
            frame: Video frame (BGR format)
            frame_number: Frame number for tracking
            
        Returns:
            Frame processing results
        """
        frame_result = {
            'frame_number': frame_number,
            'timestamp': datetime.now().isoformat(),
            'plates': [],
            'total_plates': 0,
            'valid_plates': 0
        }
        
        try:
            # Process the frame
            plates = self.process_image(frame)
            frame_result['plates'] = plates
            frame_result['total_plates'] = len(plates)
            frame_result['valid_plates'] = sum(1 for p in plates if p['valid_plate'])
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            frame_result['error'] = str(e)
        
        return frame_result
    
    def draw_results(self, 
                    image: np.ndarray, 
                    results: List[Dict[str, Any]],
                    draw_text: bool = True,
                    color_valid: Tuple[int, int, int] = (0, 255, 0),
                    color_invalid: Tuple[int, int, int] = (0, 165, 255)) -> np.ndarray:
        """
        Draw detection and recognition results on image.
        
        Args:
            image: Input image
            results: Results from process_image()
            draw_text: Whether to draw extracted text
            color_valid: Color for valid plates (BGR)
            color_invalid: Color for invalid plates (BGR)
            
        Returns:
            Image with drawn results
        """
        result_image = image.copy()
        
        for result in results:
            bbox = result['detection_bbox']
            x1, y1, x2, y2 = bbox
            
            # Choose color based on validity
            color = color_valid if result['valid_plate'] else color_invalid
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            labels = []
            labels.append(f"Conf: {result['detection_confidence']:.2f}")
            
            if draw_text and result['text']:
                labels.append(f"Text: {result['text']}")
                if result['text_confidence'] > 0:
                    labels.append(f"OCR: {result['text_confidence']:.2f}")
            
            # Draw labels
            y_offset = y1 - 10
            for label in labels:
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                
                # Background rectangle for text
                cv2.rectangle(result_image, 
                             (x1, y_offset - label_size[1] - 5),
                             (x1 + label_size[0], y_offset), 
                             color, -1)
                
                # Text
                cv2.putText(result_image, label, (x1, y_offset - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                y_offset -= label_size[1] + 8
        
        return result_image
    
    def process_video_file(self, 
                          video_path: str,
                          output_path: Optional[str] = None,
                          skip_frames: int = 1,
                          max_frames: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process a video file for license plate recognition.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            skip_frames: Process every N frames
            max_frames: Maximum frames to process
            
        Returns:
            List of frame processing results
        """
        logger.info(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties: {width}x{height}, {fps}fps, {total_frames} frames")
        
        # Setup output video if requested
        out_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_results = []
        frame_number = 0
        processed_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames if requested
                if frame_number % skip_frames != 0:
                    frame_number += 1
                    continue
                
                # Process frame
                result = self.process_video_frame(frame, frame_number)
                frame_results.append(result)
                
                # Draw results and save frame if output requested
                if out_writer:
                    annotated_frame = self.draw_results(frame, result['plates'])
                    out_writer.write(annotated_frame)
                
                processed_frames += 1
                frame_number += 1
                
                # Progress logging
                if processed_frames % 30 == 0:
                    logger.info(f"Processed {processed_frames} frames")
                
                # Check max frames limit
                if max_frames and processed_frames >= max_frames:
                    break
            
        finally:
            cap.release()
            if out_writer:
                out_writer.release()
        
        logger.info(f"Video processing complete. Processed {processed_frames} frames")
        return frame_results
    
    def save_results(self, 
                    results: List[Dict[str, Any]], 
                    output_path: str,
                    format: str = 'json') -> None:
        """
        Save processing results to file.
        
        Args:
            results: Results to save
            output_path: Output file path
            format: Output format ('json', 'csv')
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format.lower() == 'json':
            # Remove non-serializable items (like numpy arrays)
            clean_results = []
            for result in results:
                clean_result = result.copy()
                if 'crop_image' in clean_result:
                    del clean_result['crop_image']
                clean_results.append(clean_result)
            
            with open(output_path, 'w') as f:
                json.dump(clean_results, f, indent=2)
                
        elif format.lower() == 'csv':
            import pandas as pd
            
            # Flatten results for CSV
            flat_results = []
            for result in results:
                if isinstance(result, dict) and 'plates' in result:
                    # Video frame result
                    for plate in result['plates']:
                        flat_result = {
                            'frame_number': result.get('frame_number'),
                            'frame_timestamp': result.get('timestamp'),
                            **plate
                        }
                        if 'crop_image' in flat_result:
                            del flat_result['crop_image']
                        flat_results.append(flat_result)
                else:
                    # Single image result
                    clean_result = result.copy()
                    if 'crop_image' in clean_result:
                        del clean_result['crop_image']
                    flat_results.append(clean_result)
            
            df = pd.DataFrame(flat_results)
            df.to_csv(output_path, index=False)
        
        logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    pipeline = LicensePlateRecognitionPipeline()
    
    # Process a test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    results = pipeline.process_image(test_image)
    
    print(f"Found {len(results)} license plates")
    for i, result in enumerate(results):
        print(f"Plate {i+1}: {result['text']} (valid: {result['valid_plate']})")