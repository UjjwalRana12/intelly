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
        logging.info("Extracted text: %s", plain_text_content[:1000])
        return plain_text_content
    except Exception as e:
        logging.error("Error in OCR using openai in image: %s", image_path)
        logging.error("Error details: %s", e)
        return "No text extracted from image"

def preprocess_image(image_path):
    """Enhanced preprocessing with noise removal, brightness, and morphological cleanup"""
    
    print(f"🔧 Starting enhanced preprocessing: {os.path.basename(image_path)}")
    
    # Create preprocessed_images folder for organization
    preprocessed_folder = "preprocessed_images"
    if not os.path.exists(preprocessed_folder):
        os.makedirs(preprocessed_folder)
    
    img = Image.open(image_path)
    print(f"📊 Original image: {img.size}, Mode: {img.mode}")
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
        print("✅ Converted to RGB")
    
    # Step 1: Brightness enhancement (NEW)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.3)  # Increase brightness by 30%
    print("✅ Brightness enhanced (+30%)")
    
    # Step 2: Enhanced contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.8)  # Increased from 1.5 for better text clarity
    print("✅ Contrast enhanced (+80%)")
    
    # Step 3: Enhanced sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.4)  # Increased from 1.2
    print("✅ Sharpness enhanced (+40%)")
    
    # Step 4: Initial noise reduction with median filter (NEW)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    print("✅ Initial noise reduction applied")
    
    # Convert to numpy array for OpenCV processing
    img_array = np.array(img)
    
    # Convert to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Step 5: Advanced noise reduction with bilateral filter (NEW)
    # Preserves edges while removing noise
    denoised = cv2.bilateralFilter(img_bgr, 9, 75, 75)
    print("✅ Advanced bilateral noise reduction")
    
    # Convert to grayscale
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    
    # Step 6: Histogram equalization for better contrast distribution (NEW)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    equalized = clahe.apply(gray)
    print("✅ Histogram equalization applied")
    
    # Step 7: Additional noise reduction with Gaussian blur
    # Using smaller kernel than before for better text preservation
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
    print("✅ Gaussian blur noise reduction")
    
    # Step 8: Advanced thresholding combination (NEW)
    # Method 1: Adaptive threshold
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Method 2: OTSU threshold
    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Combine both thresholding methods for better results
    combined_thresh = cv2.addWeighted(adaptive_thresh, 0.7, otsu_thresh, 0.3, 0)
    print("✅ Combined adaptive + OTSU thresholding")
    
    # Step 9: Morphological cleanup operations (NEW)
    # Remove small noise particles
    kernel_small = np.ones((1,1), np.uint8)
    opened = cv2.morphologyEx(combined_thresh, cv2.MORPH_OPEN, kernel_small)
    print("✅ Small noise particles removed")
    
    # Fill small gaps in text
    kernel_medium = np.ones((2,2), np.uint8)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_medium)
    print("✅ Small gaps in text filled")
    
    # Step 10: Remove very small connected components (salt and pepper noise)
    final_cleaned = remove_small_noise_components(closed, min_area=15)
    print("✅ Tiny noise components removed")
    
    # Step 11: Final smoothing to clean up any remaining artifacts
    kernel_final = np.ones((1,1), np.uint8)
    final_result = cv2.morphologyEx(final_cleaned, cv2.MORPH_CLOSE, kernel_final)
    
    # Save to organized folder
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    extension = os.path.splitext(image_path)[1]
    preprocessed_path = os.path.join(preprocessed_folder, f"{base_name}_enhanced_preprocessed{extension}")
    
    cv2.imwrite(preprocessed_path, final_result)
    print(f"💾 Enhanced preprocessing complete: {preprocessed_path}")
    print(f"🎯 Applied: Brightness ↑ Contrast ↑ Noise ↓ Morphological Cleanup ✓")
    
    return preprocessed_path

def remove_small_noise_components(image, min_area=15):
    """Remove small connected components (noise) while preserving text"""
    
    try:
        # Find all connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            image, connectivity=8, ltype=cv2.CV_32S
        )
        
        # Create output image
        cleaned_image = np.zeros_like(image)
        
        # Keep components that are large enough (likely text)
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                # Keep this component
                cleaned_image[labels == i] = 255
        
        print(f"   🧹 Removed {num_labels - 1 - np.sum(stats[1:, cv2.CC_STAT_AREA] >= min_area)} small noise components")
        return cleaned_image
        
    except Exception as e:
        print(f"   ⚠️ Connected components cleanup failed: {e}")
        return image

def preprocess_image_adaptive(image_path):
    """Adaptive preprocessing that adjusts based on image characteristics"""
    
    print(f"🔧 Adaptive preprocessing: {os.path.basename(image_path)}")
    
    img = Image.open(image_path)
    
    # Analyze image characteristics
    img_array = np.array(img.convert('L'))  # Convert to grayscale for analysis
    mean_brightness = np.mean(img_array)
    contrast_level = np.std(img_array)
    
    print(f"📊 Image analysis - Brightness: {mean_brightness:.1f}, Contrast: {contrast_level:.1f}")
    
    # Adaptive brightness enhancement
    if mean_brightness < 100:  # Dark image
        brightness_factor = 1.6
        contrast_factor = 2.0
        print("🌙 Dark image detected - strong enhancement")
    elif mean_brightness < 150:  # Moderately dark
        brightness_factor = 1.3
        contrast_factor = 1.8
        print("🌤️ Moderately dark - medium enhancement")
    elif mean_brightness > 200:  # Very bright
        brightness_factor = 0.95
        contrast_factor = 1.5
        print("☀️ Bright image - gentle enhancement")
    else:  # Normal
        brightness_factor = 1.1
        contrast_factor = 1.6
        print("🌞 Normal brightness - standard enhancement")
    
    # Apply adaptive enhancements
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.4)
    
    # Continue with rest of processing...
    img = img.filter(ImageFilter.MedianFilter(size=3))
    
    img_array = np.array(img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    denoised = cv2.bilateralFilter(img_bgr, 9, 75, 75)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    equalized = clahe.apply(gray)
    
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
    
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    combined_thresh = cv2.addWeighted(adaptive_thresh, 0.7, otsu_thresh, 0.3, 0)
    
    # Morphological cleanup
    kernel_small = np.ones((1,1), np.uint8)
    opened = cv2.morphologyEx(combined_thresh, cv2.MORPH_OPEN, kernel_small)
    kernel_medium = np.ones((2,2), np.uint8)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_medium)
    final_cleaned = remove_small_noise_components(closed, min_area=15)
    
    # Save result
    preprocessed_folder = "preprocessed_images"
    if not os.path.exists(preprocessed_folder):
        os.makedirs(preprocessed_folder)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    extension = os.path.splitext(image_path)[1]
    preprocessed_path = os.path.join(
        preprocessed_folder, 
        f"{base_name}_adaptive_bright{brightness_factor:.1f}_preprocessed{extension}"
    )
    
    cv2.imwrite(preprocessed_path, final_cleaned)
    print(f"💾 Adaptive preprocessing complete: {preprocessed_path}")
    
    return preprocessed_path

# Update your image_to_text_enhanced function for better error handling
def image_to_text_enhanced(image_path):
    """Enhanced OCR with fallback options"""
    
    try:
        print("🚀 Starting enhanced OCR with preprocessing...")
        
        # Try enhanced preprocessing
        preprocessed_path = preprocess_image(image_path)
        preprocessed_result = image_to_text(preprocessed_path)
        
        if preprocessed_result != "No text extracted from image":
            print("✅ Enhanced preprocessing successful!")
            return preprocessed_result
        
        print("⚠️ Enhanced preprocessing failed, trying adaptive...")
        
        # Fallback to adaptive preprocessing
        adaptive_path = preprocess_image_adaptive(image_path)
        adaptive_result = image_to_text(adaptive_path)
        
        if adaptive_result != "No text extracted from image":
            print("✅ Adaptive preprocessing successful!")
            return adaptive_result
        
        print("⚠️ All preprocessing failed, trying original image...")
        
        # Last resort: original image
        original_result = image_to_text(image_path)
        return original_result
        
    except Exception as e:
        print(f"❌ Enhanced OCR failed: {e}")
        logging.error(f"Enhanced OCR failed: {e}")
        
        # Emergency fallback
        try:
            return image_to_text(image_path)
        except:
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


if __name__ == "__main__":
    
    image_path = r"C:\Users\HP\OneDrive\Desktop\test_ocr\output_images_by_pdfs\bruce_alan_data 1_page_1.png"
    process_death_certificate(image_path)

