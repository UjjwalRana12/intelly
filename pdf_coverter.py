import os
from pdf2image import convert_from_path
import pytesseract
import logging
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def pdf_to_images_with_rotation(pdf_path, output_dir=None, dpi=250, poppler_path=None):
    
    try:
        
        if output_dir is None:
            output_dir = os.path.dirname(pdf_path)
        
        
        os.makedirs(output_dir, exist_ok=True)
        
        
        if poppler_path:
            pages = convert_from_path(pdf_path, poppler_path=poppler_path, dpi=dpi)
        else:
            pages = convert_from_path(pdf_path, dpi=dpi)
        
        image_paths = []
        
        for i, page in enumerate(pages):
            # Save original page as an image
            base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
            image_path = os.path.join(output_dir, f"{base_filename}_page_{i + 1}.png")
            page.save(image_path, "PNG")
            
            # Correct rotation and save rotated image
            rotated_image_path = os.path.join(output_dir, f"{base_filename}_page_{i + 1}_rotated.png")
            rotated_image = correct_rotation_to_upright(page)
            rotated_image.save(rotated_image_path, "PNG")
            
            image_paths.append((image_path, rotated_image_path))
            print(f"Processed page {i + 1}: {rotated_image_path}")
        
        return image_paths
        
    except Exception as e:
        logging.error(f"Error converting PDF to images: {e}")
        return []


def correct_rotation_to_upright(image):
    
    try:
        
        osd = pytesseract.image_to_osd(image)
        rotation_angle = int(osd.split("Rotate:")[1].split("\n")[0].strip())
        
        if rotation_angle:
            # Rotate image to correct orientation
            image = image.rotate(360 - rotation_angle, expand=True)
            logging.info(f"Image rotated by {rotation_angle} degrees to upright.")
        
    except Exception as e:
        logging.error(f"Error correcting rotation: {e}")
    
    return image


# Example usage:
if __name__ == "__main__":
    
    pdf_file = r"C:\Users\HP\OneDrive\Desktop\test_ocr\pdfs\bruce_alan_data 1.pdf"
    
    
    image_paths = pdf_to_images_with_rotation(
        pdf_path=pdf_file,
        output_dir="output_images_by_pdfs",
        dpi=400,
         
    )
    
    # Print results
    for original, rotated in image_paths:
        print(f"Original: {original}")
        print(f"Rotated: {rotated}")