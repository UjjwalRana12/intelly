import os
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from docx import Document
from pdf2image import convert_from_path
import base64
from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
from google.cloud import vision
import io

from extract_text import extract_death_certificate_details, print_extracted_details, save_extracted_details

load_dotenv()

def image_to_text(image_path):
    """Extract text using Google Vision API"""
    try:
        # Initialize Google Vision client
        client = vision.ImageAnnotatorClient()
        
        # Load the image into memory
        with io.open(image_path, "rb") as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        # Use document_text_detection for better accuracy with documents
        response = client.document_text_detection(image=image)
        
        if response.error.message:
            raise Exception(f'Google Vision API error: {response.error.message}')
        
        # Extract full text
        if response.full_text_annotation:
            text_content = response.full_text_annotation.text
        else:
            # Fallback to basic text detection
            texts = response.text_annotations
            if texts:
                text_content = texts[0].description
            else:
                text_content = ""
        
        logging.info("OCR applied successfully using Google Vision API for image: %s", image_path)
        logging.info("Extracted text length: %d", len(text_content))
        return text_content
        
    except Exception as e:
        logging.error("Error in OCR using Google Vision API for image: %s", image_path)
        logging.error("Error details: %s", e)
        return "No text extracted from image"

def preprocess_image(image_path):
    """Lightweight preprocessing - Google Vision API handles most issues"""
    try:
        # Load image
        img = Image.open(image_path)

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Light enhancement only - Google Vision is smart enough for the rest
        img = ImageEnhance.Brightness(img).enhance(1.2)   # Slight brightness boost
        img = ImageEnhance.Contrast(img).enhance(1.3)     # Slight contrast boost

        # Save the lightly processed image
        base_name, extension = os.path.splitext(image_path)
        preprocessed_path = f"{base_name}_preprocessed{extension}"
        img.save(preprocessed_path)

        return preprocessed_path
        
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        return image_path

def image_to_text_enhanced(image_path):
    """Enhanced OCR with Google Vision API and preprocessing"""
    try:
        preprocessed_path = preprocess_image(image_path)
        
        print("  Extracting text with Google Vision API...")
        result = image_to_text(preprocessed_path)
        
        # Simple validation - check if result seems reasonable
        if result != "No text extracted from image" and len(result.strip()) > 100:
            print(f"  ✅ Extraction successful ({len(result)} characters)")
            return result
        else:
            print("  ⚠️ Preprocessed result seems incomplete, trying original...")
            fallback_result = image_to_text(image_path)
            if len(fallback_result.strip()) > len(result.strip()):
                return fallback_result
            else:
                return result
        
    except Exception as e:
        logging.error(f"Enhanced Google Vision OCR failed: {e}")
        return "No text extracted from image"

def process_death_certificate(image_path):
    """Process a single death certificate image and extract details"""
    
    # Get base filename without extension for output files
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    print(f"\n{'='*60}")
    print(f"PROCESSING: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found: {image_path}")
        return False
    
    try:
        
        print(" Step 1: OCR processing...")
        result = image_to_text_enhanced(image_path)
        
        if result == "No text extracted from image":
            print("Failed to extract text from image")
            return False
        
        print("✅ Text extraction successful")
        
        print("Step 2: Extracting structured details using LLM...")
        details = extract_death_certificate_details(result)
        
        print("Step 3: Displaying extracted details...")
        print_extracted_details(details)
        
        print(" Step 4: Saving results...")
        
        # Create output folder if it doesn't exist
        output_folder = "extracted_results"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Save complete results with structured details
        complete_filename = os.path.join(output_folder, f"{base_name}_complete.txt")
        save_extracted_details(details, complete_filename, result)

        # Save raw text only
        raw_filename = os.path.join(output_folder, f"{base_name}_raw.txt")
        with open(raw_filename, "w", encoding="utf-8") as f:
            f.write(result)

        # Check for enhanced image
        enhanced_path = f"{os.path.splitext(image_path)[0]}_preprocessed{os.path.splitext(image_path)[1]}"
        
        print(f"Results saved:")
        print(f"Raw text: {raw_filename}")
        print(f"Complete details: {complete_filename}")
        if os.path.exists(enhanced_path):
            print(f"Enhanced image: {os.path.basename(enhanced_path)}")
        
        return True
        
    except Exception as e:
        print(f"ERROR processing {os.path.basename(image_path)}: {e}")
        logging.error(f"Error processing {image_path}: {e}")
        return False


if __name__ == "__main__":
    
    image_path = r"C:\Users\HP\OneDrive\Desktop\test_ocr\output_images_by_pdfs\Luvenia_DC 1_page_1.png"
    process_death_certificate(image_path)