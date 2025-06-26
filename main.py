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
import openai

from extract_text import extract_death_certificate_details, print_extracted_details, save_extracted_details

load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

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

def structure_text_with_openai(raw_text):
    """Use OpenAI to structure and align the OCR text properly"""
    try:
        prompt = f"""
        You are an expert document formatter specializing in death certificates and legal documents. 
        The following text was extracted from a death certificate using OCR and needs to be properly structured and aligned.

        RAW OCR TEXT:
        {raw_text}

        INSTRUCTIONS:
        1. Organize the text into a properly formatted death certificate structure
        2. Align field labels with their corresponding values using consistent spacing
        3. Group related information together (personal details, death details, etc.)
        4. Maintain proper spacing and indentation for readability
        5. Fix any obvious OCR errors in common words (but preserve all data)
        6. Preserve all important information - don't remove anything
        7. Format it like an actual death certificate with clear sections

        SECTIONS TO ORGANIZE:
        - Document Header/Title
        - Certificate Information
        - Decedent Information (Name, DOB, DOD, etc.)
        - Death Details (Location, Time, Cause, etc.)
        - Personal Information (SSN, Address, Race, Sex, etc.)
        - Family/Next of Kin Information
        - Medical Information
        - Certification/Signatures
        - Any other relevant sections

        OUTPUT FORMAT:
        Provide a clean, well-structured version of the death certificate text with:
        - Clear section headers
        - Proper alignment of labels and values
        - Consistent spacing and formatting
        - Easy to read and process structure
        - Preserve all original data
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert document formatter for legal documents. Focus on preserving all data while improving structure and readability."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3000,
            temperature=0.1
        )

        structured_text = response.choices[0].message.content
        logging.info("Text successfully structured with OpenAI")
        return structured_text

    except Exception as e:
        logging.error(f"Error structuring text with OpenAI: {e}")
        return raw_text  # Return original text if formatting fails

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
    """Enhanced OCR with Google Vision API, preprocessing, and OpenAI structuring"""
    try:
        preprocessed_path = preprocess_image(image_path)
        
        print("  Step 1: Extracting raw text with Google Vision API...")
        raw_result = image_to_text(preprocessed_path)
        
        # Simple validation - check if result seems reasonable
        if raw_result == "No text extracted from image" or len(raw_result.strip()) < 50:
            print("  ⚠️ Preprocessed result seems incomplete, trying original...")
            raw_result = image_to_text(image_path)
        
        if raw_result == "No text extracted from image" or len(raw_result.strip()) < 50:
            return raw_result, "No text extracted from image"
        
        print(f"  ✅ Raw extraction successful ({len(raw_result)} characters)")
        
        print("  Step 2: Structuring text with OpenAI...")
        structured_result = structure_text_with_openai(raw_result)
        
        print(f"  ✅ Text structuring successful ({len(structured_result)} characters)")
        
        return raw_result, structured_result
        
    except Exception as e:
        logging.error(f"Enhanced OCR with structuring failed: {e}")
        return "No text extracted from image", "No text extracted from image"

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
        
        print("Step 1: OCR processing and structuring...")
        raw_text, structured_text = image_to_text_enhanced(image_path)
        
        if structured_text == "No text extracted from image":
            print("Failed to extract text from image")
            return False
        
        print("✅ Text extraction and structuring successful")
        
        print("Step 2: Extracting structured details using LLM...")
        # Use structured text for better field extraction
        details = extract_death_certificate_details(structured_text)
        
        print("Step 3: Displaying extracted details...")
        print_extracted_details(details)
        
        print("Step 4: Saving results...")
        
        # Create output folder if it doesn't exist
        output_folder = "extracted_results"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Save raw OCR text
        raw_filename = os.path.join(output_folder, f"{base_name}_raw_ocr.txt")
        with open(raw_filename, "w", encoding="utf-8") as f:
            f.write(raw_text)

        # Save structured text
        structured_filename = os.path.join(output_folder, f"{base_name}_structured.txt")
        with open(structured_filename, "w", encoding="utf-8") as f:
            f.write(structured_text)

        # Save complete results with structured details
        complete_filename = os.path.join(output_folder, f"{base_name}_complete.txt")
        save_extracted_details(details, complete_filename, structured_text)

        # Check for enhanced image
        enhanced_path = f"{os.path.splitext(image_path)[0]}_preprocessed{os.path.splitext(image_path)[1]}"
        
        print(f"Results saved:")
        print(f"  Raw OCR text: {raw_filename}")
        print(f"  Structured text: {structured_filename}")
        print(f"  Complete details: {complete_filename}")
        if os.path.exists(enhanced_path):
            print(f"  Enhanced image: {os.path.basename(enhanced_path)}")
        
        return True
        
    except Exception as e:
        print(f"ERROR processing {os.path.basename(image_path)}: {e}")
        logging.error(f"Error processing {image_path}: {e}")
        return False


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ocr_processing.log'),
            logging.StreamHandler()
        ]
    )
    
    image_path = r"C:\Users\HP\OneDrive\Desktop\test_ocr\output_images_by_pdfs\COUNTY_OF_OAKLAND_page_1_preprocessed.png"
    process_death_certificate(image_path)