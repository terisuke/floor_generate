"""
PDF Dimension Extractor
Extract dimensions from architectural PDF floor plans
"""

import cv2
import re
import numpy as np
import pytesseract
import easyocr
from pdf2image import convert_from_path
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DimensionExtractor:
    """Extract dimensions from architectural PDF drawings"""
    
    def __init__(self):
        """Initialize OCR readers"""
        try:
            self.reader = easyocr.Reader(['ja', 'en'])
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.warning(f"EasyOCR initialization failed: {e}")
            self.reader = None
        
        # Common dimension patterns in Japanese architectural drawings
        self.dimension_patterns = [
            r'(\d{1,2}),(\d{3})',      # 9,100形式
            r'(\d{4,5})',               # 9100形式
            r'(\d+)×(\d+)',             # 横×縦形式
            r'(\d+)\s*[xX×]\s*(\d+)',   # Various multiplication symbols
            r'(\d+\.?\d*)\s*[mM㎡]',    # Meters or square meters
        ]
    
    def extract_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract dimension information from PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dimension dictionaries
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=300)
            logger.info(f"Converted PDF to {len(images)} images")
            
            all_dimensions = []
            
            for page_num, img in enumerate(images):
                logger.info(f"Processing page {page_num + 1}/{len(images)}")
                
                # Convert PIL Image to numpy array
                img_array = np.array(img)
                
                # Preprocess image
                processed = self.preprocess_image(img_array)
                
                # Extract dimensions
                dimensions = self.detect_dimensions(processed)
                
                # Add page number to each dimension
                for dim in dimensions:
                    dim['page'] = page_num + 1
                
                all_dimensions.extend(dimensions)
            
            # Validate and filter dimensions
            validated = self.validate_dimensions(all_dimensions)
            
            logger.info(f"Extracted {len(validated)} valid dimensions from {len(all_dimensions)} total")
            return validated
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return []
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to connect text
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return morph
    
    def detect_dimensions(self, image: np.ndarray) -> List[Dict]:
        """
        Detect dimension numbers from image
        
        Args:
            image: Preprocessed image
            
        Returns:
            List of detected dimensions
        """
        dimensions = []
        
        # Try EasyOCR if available
        if self.reader:
            try:
                results = self.reader.readtext(image)
                
                for (bbox, text, confidence) in results:
                    if confidence > 0.7:
                        # Try to match dimension patterns
                        for pattern in self.dimension_patterns:
                            matches = re.findall(pattern, text)
                            
                            for match in matches:
                                dim_info = self.parse_dimension_match(match, text)
                                if dim_info:
                                    dim_info.update({
                                        'bbox': bbox,
                                        'confidence': confidence,
                                        'original_text': text,
                                        'method': 'easyocr'
                                    })
                                    dimensions.append(dim_info)
                                    
            except Exception as e:
                logger.warning(f"EasyOCR failed: {e}")
        
        # Fallback to Tesseract
        if not dimensions:
            try:
                text = pytesseract.image_to_string(image, lang='jpn+eng')
                lines = text.split('\n')
                
                for line in lines:
                    for pattern in self.dimension_patterns:
                        matches = re.findall(pattern, line)
                        
                        for match in matches:
                            dim_info = self.parse_dimension_match(match, line)
                            if dim_info:
                                dim_info.update({
                                    'confidence': 0.5,  # Lower confidence for Tesseract
                                    'original_text': line,
                                    'method': 'tesseract'
                                })
                                dimensions.append(dim_info)
                                
            except Exception as e:
                logger.warning(f"Tesseract failed: {e}")
        
        return dimensions
    
    def parse_dimension_match(self, match, text: str) -> Optional[Dict]:
        """
        Parse regex match to extract dimension value
        
        Args:
            match: Regex match object or tuple
            text: Original text
            
        Returns:
            Dimension info dictionary or None
        """
        try:
            if isinstance(match, tuple):
                if len(match) == 2:
                    # Check if it's a comma-separated number (e.g., 9,100)
                    if ',' in text:
                        value = int(match[0]) * 1000 + int(match[1])
                    else:
                        # It's a width x height format
                        return {
                            'type': 'area',
                            'width': int(match[0]),
                            'height': int(match[1]),
                            'value': [int(match[0]), int(match[1])]
                        }
                else:
                    return None
            else:
                # Single number
                value = int(match)
            
            # Check if it's a reasonable dimension (100mm to 50000mm)
            if 100 <= value <= 50000:
                return {
                    'type': 'linear',
                    'value': value
                }
                
        except (ValueError, IndexError):
            pass
            
        return None
    
    def is_valid_dimension(self, value) -> bool:
        """
        Check if dimension value is reasonable for architecture
        
        Args:
            value: Dimension value or list of values
            
        Returns:
            True if valid
        """
        if isinstance(value, list):
            # For area dimensions, check both values
            return all(100 <= v <= 50000 for v in value)
        else:
            # For linear dimensions
            return 100 <= value <= 50000
    
    def validate_dimensions(self, dimensions: List[Dict]) -> List[Dict]:
        """
        Validate and filter extracted dimensions
        
        Args:
            dimensions: List of raw dimensions
            
        Returns:
            List of validated dimensions
        """
        validated = []
        
        for dim in dimensions:
            # Skip low confidence
            if dim.get('confidence', 0) < 0.5:
                continue
            
            # Validate value
            value = dim.get('value')
            if value and self.is_valid_dimension(value):
                validated.append(dim)
        
        # Remove duplicates (same value and similar position)
        unique_dims = []
        for dim in validated:
            is_duplicate = False
            
            for existing in unique_dims:
                if (dim['value'] == existing['value'] and 
                    dim.get('page') == existing.get('page')):
                    # Check if bounding boxes are close (if available)
                    if 'bbox' in dim and 'bbox' in existing:
                        # Simple check - could be improved
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_dims.append(dim)
        
        return unique_dims


def main():
    """Test the dimension extractor"""
    extractor = DimensionExtractor()
    
    # Test with sample PDF
    pdf_path = "data/raw_pdfs/sample.pdf"
    if Path(pdf_path).exists():
        dimensions = extractor.extract_from_pdf(pdf_path)
        
        print(f"\nExtracted {len(dimensions)} dimensions:")
        for i, dim in enumerate(dimensions):
            print(f"\n{i+1}. Type: {dim['type']}")
            print(f"   Value: {dim['value']}")
            print(f"   Confidence: {dim['confidence']:.2f}")
            print(f"   Method: {dim['method']}")
            print(f"   Text: {dim.get('original_text', 'N/A')}")
    else:
        print(f"PDF file not found: {pdf_path}")


if __name__ == "__main__":
    main()
