import os
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from docx import Document
from pdf2image import convert_from_path
from openai import OpenAI 
import base64

from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np

from extract_text import extract_death_certificate_details, print_extracted_details, save_extracted_details

load_dotenv()  

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def preprocess_image(image_path):
    """Preprocess image to improve OCR accuracy"""
    
    img = Image.open(image_path)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.2)
    
    # Convert to numpy array for OpenCV processing
    img_array = np.array(img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    base_name = os.path.splitext(image_path)[0]
    extension = os.path.splitext(image_path)[1]
    preprocessed_path = f"{base_name}_preprocessed{extension}"
    
    cv2.imwrite(preprocessed_path, thresh)
    
    return preprocessed_path

def image_to_text(image_path):
    try:
        base64_img = f"data:image/png;base64,{encode_image(image_path)}"

        # Enhanced prompt for better OCR
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": """Extract ALL text from this image with high accuracy. 
                            Please:
                            - Preserve original formatting, line breaks, and spacing
                            - Include all text, even if partially visible or unclear
                            - Maintain the structure of tables, lists, and paragraphs
                            - If text is unclear, provide your best interpretation
                            - Do not add explanations, just return the extracted text"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": base64_img, "detail": "high"},
                        },
                    ],
                }
            ],
            max_tokens=4000,
            temperature=0,
        )

        plain_text_content = response.choices[0].message.content
        logging.info("OCR applied successfully using openai in image: %s", image_path)
        return plain_text_content
    except Exception as e:
        logging.error("Error in OCR using openai in image: %s", image_path)
        logging.error("Error details: %s", e)
        return "No text extracted from image"

def image_to_text_enhanced(image_path):
    """Enhanced OCR with preprocessing"""
    try:
        preprocessed_path = preprocess_image(image_path)
        
        preprocessed_result = image_to_text(preprocessed_path)
        
        return preprocessed_result
    except Exception as e:
        logging.error(f"Enhanced OCR failed: {e}")
        return "No text extracted from image"

def image_to_text_original(image_path):
    """OCR on original image without preprocessing"""
    try:
        print("Processing original image...")
        original_result = image_to_text(image_path)
        print("Using original image result")
        return original_result
    except Exception as e:
        logging.error(f"Original OCR failed: {e}")
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

# Example usage
if __name__ == "__main__":
    
    image_path = r"C:\Users\HP\OneDrive\Desktop\test_ocr\output_images_by_pdfs\will copy and death certificate 1_page_2.png"
    process_death_certificate(image_path)

