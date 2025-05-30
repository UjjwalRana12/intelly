import os
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from docx import Document

from pdf2image import convert_from_path

from openai import OpenAI 
import base64
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import shutil
from langchain.chains import LLMChain

import re

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np

# Initialize OpenAI client
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
    
    # Create preprocessed filename by inserting '_preprocessed' before the extension
    base_name = os.path.splitext(image_path)[0]
    extension = os.path.splitext(image_path)[1]
    preprocessed_path = f"{base_name}_preprocessed{extension}"
    
    cv2.imwrite(preprocessed_path, thresh)
    
    print(f"Enhanced image saved to: {preprocessed_path}")
    
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
        print("plain_text_content:", plain_text_content)
        logging.info("OCR applied successfully using openai in image: %s", image_path)
        return plain_text_content
    except Exception as e:
        logging.error("Error in OCR using openai in image: %s", image_path)
        logging.error("Error details: %s", e)
        return "No text extracted from image"

def image_to_text_enhanced(image_path):
    """Enhanced OCR with preprocessing"""
    try:
        # Create preprocessed image directly
        print("Creating preprocessed image...")
        preprocessed_path = preprocess_image(image_path)
        
        # Use preprocessed image for OCR
        print("Trying OCR with preprocessed image...")
        preprocessed_result = image_to_text(preprocessed_path)
        
        print("Using preprocessed image result")
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
    

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    image_path = r"C:\Users\HP\OneDrive\Desktop\test_ocr\beverly jean.jpg"
    
    if os.path.exists(image_path):
        print("Processing image with OpenAI OCR...")
        
        
        result = image_to_text_enhanced(image_path)

        
        
        print(f"\nExtracted text:\n{'-'*50}")
        print(result)
        print(f"{'-'*50}")
        
        
        with open("beverly.txt", "w", encoding="utf-8") as f:
            f.write(result)
        print("Text saved to 'beverly.txt'")
        
        
        
        enhanced_path = image_path.replace('.png', '_preprocessed.png')
        if os.path.exists(enhanced_path):
            print(f"\nEnhanced image saved at: {enhanced_path}")
        
    else:
        print("Image file not found!")

