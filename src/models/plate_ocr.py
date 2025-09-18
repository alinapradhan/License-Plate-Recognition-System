"""
OCR Text Recognition for License Plates
=======================================

This module provides OCR functionality for extracting text from detected
license plate regions using EasyOCR and preprocessing techniques.
"""

import cv2
import numpy as np
import easyocr
import re
from typing import List, Dict, Any, Tuple, Optional
import logging
from PIL import Image, ImageEnhance
import string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LicensePlateOCR:
    """
    OCR system specifically designed for license plate text recognition.
    """
    
    def __init__(self, languages: List[str] = ['en'], gpu: bool = True):
        """
        Initialize the OCR reader.
        
        Args:
            languages: List of languages for OCR recognition
            gpu: Whether to use GPU acceleration
        """
        self.languages = languages
        self.gpu = gpu and self._check_gpu_available()
        self.reader = None
        self.initialize_reader()
    
    def _check_gpu_available(self) -> bool:
        """Check if GPU is available for EasyOCR."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def initialize_reader(self) -> None:
        """Initialize the EasyOCR reader."""
        try:
            logger.info(f"Initializing EasyOCR with languages: {self.languages}, GPU: {self.gpu}")
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
            logger.info("EasyOCR reader initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR reader: {e}")
            raise
    
    def preprocess_plate_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess license plate image for better OCR results.
        
        Args:
            image: Input plate image (BGR format)
            
        Returns:
            Preprocessed image
        """
        if len(image.shape) == 3:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize image if too small
        height, width = gray.shape
        if height < 32 or width < 100:
            scale_factor = max(32 / height, 100 / width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Noise reduction
        gray = cv2.medianBlur(gray, 3)
        
        # Sharpen the image
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        gray = cv2.filter2D(gray, -1, kernel)
        
        # Thresholding
        # Try adaptive thresholding first
        thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Also try Otsu's thresholding
        _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Use the one with better characteristics (more connected components in expected range)
        contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Choose threshold based on number of contours (characters should be 5-8)
        if 3 <= len(contours1) <= 10:
            processed = thresh1
        elif 3 <= len(contours2) <= 10:
            processed = thresh2
        else:
            # Use adaptive if both are out of range
            processed = thresh1
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        
        return processed
    
    def extract_text(self, 
                    image: np.ndarray, 
                    preprocess: bool = True,
                    confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Extract text from license plate image.
        
        Args:
            image: Input image (BGR format)
            preprocess: Whether to apply preprocessing
            confidence_threshold: Minimum confidence for text detection
            
        Returns:
            List of detected text with bounding boxes and confidence scores
        """
        if self.reader is None:
            raise RuntimeError("OCR reader not initialized")
        
        try:
            # Preprocess if requested
            if preprocess:
                processed_image = self.preprocess_plate_image(image)
            else:
                processed_image = image
            
            # Run OCR
            results = self.reader.readtext(processed_image)
            
            # Filter and format results
            extracted_texts = []
            for (bbox, text, confidence) in results:
                if confidence >= confidence_threshold:
                    # Clean up the text
                    cleaned_text = self.clean_text(text)
                    if cleaned_text:  # Only add non-empty results
                        extracted_texts.append({
                            'text': cleaned_text,
                            'raw_text': text,
                            'confidence': float(confidence),
                            'bbox': bbox
                        })
            
            logger.info(f"Extracted {len(extracted_texts)} text regions")
            return extracted_texts
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw text from OCR
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = text.strip()
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text)      # Normalize whitespace
        
        # Convert to uppercase (standard for license plates)
        text = text.upper()
        
        # Remove obviously wrong characters that might be OCR errors
        # Keep only alphanumeric characters
        text = re.sub(r'[^A-Z0-9\s]', '', text)
        
        return text.strip()
    
    def validate_license_plate(self, text: str) -> bool:
        """
        Validate if extracted text looks like a license plate.
        
        Args:
            text: Extracted text
            
        Returns:
            True if text appears to be a valid license plate
        """
        if not text:
            return False
        
        # Remove spaces for validation
        clean_text = text.replace(' ', '')
        
        # Check length (most license plates are 4-8 characters)
        if len(clean_text) < 3 or len(clean_text) > 10:
            return False
        
        # Should contain both letters and numbers (most common pattern)
        has_letters = bool(re.search(r'[A-Z]', clean_text))
        has_numbers = bool(re.search(r'[0-9]', clean_text))
        
        # Most license plates have both letters and numbers
        if not (has_letters and has_numbers):
            # Some plates might be all numbers or all letters, so be less strict
            if len(clean_text) < 4:
                return False
        
        # Check for common patterns (this can be extended based on region)
        # Examples: ABC123, 123ABC, AB12CD, etc.
        common_patterns = [
            r'^[A-Z]{2,3}[0-9]{3,4}$',  # ABC123, AB1234
            r'^[0-9]{3,4}[A-Z]{2,3}$',  # 123ABC, 1234AB  
            r'^[A-Z][0-9]{2,3}[A-Z]{2,3}$',  # A123BC
            r'^[A-Z]{2}[0-9]{2}[A-Z]{2}$',  # AB12CD
            r'^[0-9]{4,6}$',  # All numbers
            r'^[A-Z]{4,6}$',  # All letters
        ]
        
        for pattern in common_patterns:
            if re.match(pattern, clean_text):
                return True
        
        # If no specific pattern matches, check if it's reasonable
        # At least 50% of characters should be alphanumeric
        alnum_count = sum(1 for c in clean_text if c.isalnum())
        if alnum_count / len(clean_text) >= 0.8:
            return True
        
        return False
    
    def extract_best_text(self, 
                         image: np.ndarray,
                         min_confidence: float = 0.3) -> Optional[str]:
        """
        Extract the most likely license plate text from an image.
        
        Args:
            image: Input plate image
            min_confidence: Minimum confidence threshold
            
        Returns:
            Best extracted license plate text, or None if no valid text found
        """
        # Try with preprocessing
        texts_processed = self.extract_text(image, preprocess=True, 
                                          confidence_threshold=min_confidence)
        
        # Try without preprocessing as backup
        texts_raw = self.extract_text(image, preprocess=False, 
                                    confidence_threshold=min_confidence)
        
        # Combine results
        all_texts = texts_processed + texts_raw
        
        # Filter valid license plates and sort by confidence
        valid_texts = []
        for text_info in all_texts:
            if self.validate_license_plate(text_info['text']):
                valid_texts.append(text_info)
        
        if not valid_texts:
            return None
        
        # Sort by confidence and return the best one
        valid_texts.sort(key=lambda x: x['confidence'], reverse=True)
        return valid_texts[0]['text']
    
    def batch_extract(self, 
                     images: List[np.ndarray],
                     min_confidence: float = 0.3) -> List[Optional[str]]:
        """
        Extract text from multiple license plate images.
        
        Args:
            images: List of plate images
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of extracted texts (or None for failed extractions)
        """
        results = []
        for i, image in enumerate(images):
            try:
                text = self.extract_best_text(image, min_confidence)
                results.append(text)
                logger.info(f"Image {i+1}/{len(images)}: {'Success' if text else 'Failed'}")
            except Exception as e:
                logger.error(f"Failed to extract text from image {i+1}: {e}")
                results.append(None)
        
        return results


def enhance_image_pil(image: np.ndarray, 
                     brightness: float = 1.2, 
                     contrast: float = 1.5,
                     sharpness: float = 1.3) -> np.ndarray:
    """
    Enhance image using PIL for better OCR results.
    
    Args:
        image: Input image (BGR format)
        brightness: Brightness enhancement factor
        contrast: Contrast enhancement factor
        sharpness: Sharpness enhancement factor
        
    Returns:
        Enhanced image
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    
    # Apply enhancements
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(brightness)
    
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(contrast)
    
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(sharpness)
    
    # Convert back to BGR
    enhanced_rgb = np.array(pil_image)
    enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
    
    return enhanced_bgr


if __name__ == "__main__":
    # Example usage
    ocr = LicensePlateOCR()
    
    # Load a test image (you would replace this with actual plate crop)
    test_image = np.zeros((50, 200, 3), dtype=np.uint8)
    
    # Extract text
    result = ocr.extract_best_text(test_image)
    print(f"Extracted text: {result}")