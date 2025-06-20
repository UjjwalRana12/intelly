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
# import cv2
# import numpy as np
# from PIL import Image, ImageEnhance
# import os

from extract_text import extract_death_certificate_details, print_extracted_details, save_extracted_details

import openai
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

openai.api_key = api_key

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def image_to_text(image_path):
    try:
        base64_img = f"data:image/png;base64,{encode_image(image_path)}"

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": """You are an expert OCR system specialized in death certificates and legal documents. Extract ALL text from this image with maximum accuracy and consistency.

                            CRITICAL REQUIREMENTS:
                            - Read every single word, number, date, and character visible in the image
                            - Preserve exact formatting, line breaks, and spacing as they appear
                            - Include ALL form fields, labels, values, checkboxes, and annotations
                            - Capture headers, footers, signatures, stamps, and watermarks
                            - Maintain table structure and field alignment precisely
                            - Include partially visible or faded text with your best interpretation
                            - Process the entire document systematically from top to bottom
                            - Do not skip any sections, even if they appear empty or unclear
                            
                            CONSISTENCY INSTRUCTIONS:
                            - Use the same text extraction approach every time
                            - Maintain consistent formatting standards
                            - Apply consistent interpretation rules for unclear text
                            
                            OUTPUT FORMAT:
                            Return ONLY the extracted text with no explanations, commentary, or metadata.
                            Preserve the original document structure and formatting exactly."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_img, "detail": "high"},
                        },
                    ],
                }
            ],
            max_tokens=4000,
            temperature=0,  # Keep at 0 for consistency
        )

        plain_text_content = response.choices[0].message.content
        logging.info("OCR applied successfully using openai in image: %s", image_path)
        logging.info("Extracted text length: %d", len(plain_text_content))
        return plain_text_content
    except Exception as e:
        logging.error("Error in OCR using openai in image: %s", image_path)
        logging.error("Error details: %s", e)
        return "No text extracted from image"

def preprocess_image(image_path):
    """Enhanced preprocessing for more consistent OCR results"""
    try:
        # Load image
        img = Image.open(image_path)

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Get image dimensions for adaptive processing
        width, height = img.size
        
        # Resize if image is too large (for consistency and performance)
        max_dimension = 2000
        if max(width, height) > max_dimension:
            scale = max_dimension / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.LANCZOS)

        # Enhanced brightness and contrast with fixed values for consistency
        img = ImageEnhance.Brightness(img).enhance(1.3)   # Increased brightness
        img = ImageEnhance.Contrast(img).enhance(1.6)     # Higher contrast for better text clarity
        img = ImageEnhance.Sharpness(img).enhance(1.4)    # Moderate sharpness

        # Convert to OpenCV format
        img_array = np.array(img)

        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur to reduce noise before denoising
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # More aggressive denoising with consistent parameters
        denoised = cv2.fastNlMeansDenoising(blurred, h=12, templateWindowSize=7, searchWindowSize=21)

        # Apply morphological operations to clean up text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

        # Use adaptive threshold for more consistent results across different images
        thresh = cv2.adaptiveThreshold(
            cleaned, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            15,  # Increased block size for better text handling
            4    # Increased C value for better threshold
        )

        # Final morphological operation to connect broken characters
        final_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        final_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, final_kernel)

        # Save preprocessed image
        base_name, extension = os.path.splitext(image_path)
        preprocessed_path = f"{base_name}_preprocessed{extension}"
        cv2.imwrite(preprocessed_path, final_thresh)

        return preprocessed_path
        
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        return image_path

def image_to_text_enhanced(image_path):
    """Enhanced OCR with validation for consistency"""
    try:
        preprocessed_path = preprocess_image(image_path)
        
        print("  Extracting text with OpenAI...")
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
        logging.error(f"Enhanced OCR failed: {e}")
        return "No text extracted from image"


# def ocr_image(image_path):
#     """
#     Perform OCR on the given image and return the extracted text.
#     Handles encoding errors gracefully.
#     """
#     import pytesseract
#     from PIL import Image
#     import re

#     try:
#         image = Image.open(image_path)
#     except Exception as e:
#         logging.error("Failed to open image %s: %s", image_path, e)
#         return ""

#     try:
#         # Optional: correct rotation
#         image = correct_rotation_to_upright(image)

#         custom_config = r'--oem 3 --psm 6'
#         text = pytesseract.image_to_string(image, config=custom_config)
#         logging.info("Raw OCR text:\n%s", text)

#         # Handle encoding issues
#         text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')

#         # Optional: clean non-ASCII characters
#         text = re.sub(r'[^\x00-\x7F]+', ' ', text)

      

        
               

#         logging.info("Processed OCR text:\n%s", text)
#         return text

#     except Exception as e:
#         logging.error("OCR failed for %s: %s", image_path, e)
#         return ""   


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
        
        print(" Step 1:OCR processing...")
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