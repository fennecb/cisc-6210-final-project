"""
Menu image OCR module - custom image processing implementation.
Extracts text from menu images before sending to LLM.
"""
import os
from pathlib import Path
from typing import Optional, List, Dict
from PIL import Image
import io

from src.utils.logger import setup_logger
from src.utils.cache import CacheManager

logger = setup_logger(__name__)

class MenuOCR:
    """
    OCR system for extracting text from menu images.
    Uses Tesseract OCR (open-source) before resorting to LLM vision.
    """
    
    def __init__(self, use_cache: bool = True):
        """Initialize OCR system."""
        self.use_cache = use_cache
        self.cache = CacheManager() if use_cache else None
        
        # Check if pytesseract is available
        self.tesseract_available = self._check_tesseract()
        
        # Check if EasyOCR is available
        self.easyocr_available = self._check_easyocr()
        self.easyocr_reader = None
    
    def _check_tesseract(self) -> bool:
        """Check if Tesseract OCR is available."""
        try:
            import pytesseract
            # Try to get version
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR available")
            return True
        except Exception as e:
            logger.warning(f"Tesseract OCR not available: {e}")
            return False
    
    def _check_easyocr(self) -> bool:
        """Check if EasyOCR is available."""
        try:
            import easyocr
            logger.info("EasyOCR available")
            return True
        except ImportError:
            logger.warning("EasyOCR not available")
            return False
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results.
        
        Args:
            image: PIL Image
        
        Returns:
            Preprocessed image
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large (for faster processing)
        max_dimension = 2000
        if max(image.size) > max_dimension:
            ratio = max_dimension / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    
    def extract_text_tesseract(self, image_path: str) -> Optional[str]:
        """
        Extract text using Tesseract OCR.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Extracted text or None
        """
        if not self.tesseract_available:
            return None
        
        try:
            import pytesseract
            
            # Load and preprocess image
            image = Image.open(image_path)
            image = self._preprocess_image(image)
            
            # Extract text
            text = pytesseract.image_to_string(image, lang='eng')
            
            # Clean up text
            text = text.strip()
            
            if text:
                logger.info(f"Tesseract extracted {len(text)} characters")
                return text
            
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
        
        return None
    
    def extract_text_easyocr(self, image_path: str) -> Optional[str]:
        """
        Extract text using EasyOCR (more accurate but slower).
        
        Args:
            image_path: Path to image file
        
        Returns:
            Extracted text or None
        """
        if not self.easyocr_available:
            return None
        
        try:
            import easyocr
            
            # Initialize reader (lazy loading)
            if self.easyocr_reader is None:
                logger.info("Initializing EasyOCR reader...")
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
            
            # Load and preprocess image
            image = Image.open(image_path)
            image = self._preprocess_image(image)
            
            # Convert to numpy array
            import numpy as np
            image_np = np.array(image)
            
            # Extract text
            results = self.easyocr_reader.readtext(image_np)
            
            # Combine all text
            text_parts = [result[1] for result in results]
            text = '\n'.join(text_parts)
            
            if text:
                logger.info(f"EasyOCR extracted {len(text)} characters")
                return text
            
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
        
        return None
    
    def extract_text(self, image_path: str, method: str = 'auto') -> Optional[str]:
        """
        Extract text from menu image.
        
        Args:
            image_path: Path to image file
            method: OCR method ('tesseract', 'easyocr', 'auto')
        
        Returns:
            Extracted text or None
        """
        cache_key = f"ocr:{image_path}:{method}"
        
        # Check cache
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                logger.info(f"Using cached OCR result for: {image_path}")
                return cached
        
        text = None
        
        if method == 'tesseract' or (method == 'auto' and self.tesseract_available):
            text = self.extract_text_tesseract(image_path)
        
        if text is None and (method == 'easyocr' or method == 'auto'):
            text = self.extract_text_easyocr(image_path)
        
        # Cache result
        if text and self.cache:
            self.cache.set(cache_key, text)
        
        return text
    
    def extract_from_url(self, image_url: str, save_dir: str = "data/temp") -> Optional[str]:
        """
        Download image from URL and extract text.
        
        Args:
            image_url: URL of image
            save_dir: Directory to save temporary image
        
        Returns:
            Extracted text or None
        """
        try:
            import requests
            
            # Create temp directory
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate filename from URL hash
            import hashlib
            url_hash = hashlib.md5(image_url.encode()).hexdigest()
            temp_path = os.path.join(save_dir, f"{url_hash}.jpg")
            
            # Download if not exists
            if not os.path.exists(temp_path):
                response = requests.get(image_url, timeout=15)
                response.raise_for_status()
                
                with open(temp_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Downloaded image from URL")
            
            # Extract text
            return self.extract_text(temp_path)
        
        except Exception as e:
            logger.error(f"Error processing image URL: {e}")
            return None
    
    def batch_extract(self, image_paths: List[str]) -> Dict[str, str]:
        """
        Extract text from multiple images.
        
        Args:
            image_paths: List of image file paths
        
        Returns:
            Dictionary mapping image path to extracted text
        """
        results = {}
        
        for image_path in image_paths:
            text = self.extract_text(image_path)
            if text:
                results[image_path] = text
        
        logger.info(f"Extracted text from {len(results)}/{len(image_paths)} images")
        return results
    
    def get_image_info(self, image_path: str) -> Dict:
        """
        Get basic information about an image.
        
        Args:
            image_path: Path to image
        
        Returns:
            Dictionary with image metadata
        """
        try:
            image = Image.open(image_path)
            return {
                'size': image.size,
                'mode': image.mode,
                'format': image.format,
                'width': image.width,
                'height': image.height
            }
        except Exception as e:
            logger.error(f"Error getting image info: {e}")
            return {}
