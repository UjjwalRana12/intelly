from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

def process_single_page(page_data):
    """Process a single page and return its text"""
    i, page, pdf_path = page_data
    try:
        # Save each page as an image
        image_path = f"{pdf_path}_page_{i + 1}.png"
        page.save(image_path, "PNG")

        #image rotation
        image_path1 = f"{pdf_path}_page_{i + 1}_rotated.png"
        rotated_image = correct_rotation_to_upright(page)
        rotated_image.save(image_path1, "PNG")

        # Use image_to_text function for text extraction
        print(f"Processing page {i + 1}...")
        preprocessed_path = preprocess_image(image_path1)
        logging.info(f"preprocessed_path:{preprocessed_path}")
        text = image_to_text(preprocessed_path)
        
        return i, text  # Return page number and text for ordering
        
    except Exception as e:
        logging.error(f"Error processing page {i + 1}: {e}")
        return i, ""  # Return empty text on error

def pdf_to_text(pdf_path):
    """Extracts text from a scanned PDF using image-to-text processing with parallel processing."""
    logging.info(f"pdf_to_text called with: {pdf_path}")
    
    try:
        # Convert PDF pages to images
        try:
            pages = convert_from_path(pdf_path, poppler_path=r"C:\Release-24.08.0-0\poppler-24.08.0\Library\bin", dpi=400)
            
            # Prepare data for parallel processing
            page_data = [(i, page, pdf_path) for i, page in enumerate(pages)]
            
            # Process pages in parallel
            all_text = [None] * len(pages)  # Initialize list with correct size
            
            # Use ThreadPoolExecutor for parallel processing
            max_workers = min(4, len(pages))  # Limit to 4 threads or number of pages
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_page = {executor.submit(process_single_page, data): data[0] for data in page_data}
                
                # Collect results as they complete
                for future in as_completed(future_to_page):
                    try:
                        page_index, text = future.result()
                        all_text[page_index] = text  # Store text in correct order
                        logging.info(f"Completed processing page {page_index + 1}")
                    except Exception as e:
                        page_index = future_to_page[future]
                        logging.error(f"Page {page_index + 1} generated an exception: {e}")
                        all_text[page_index] = ""  # Set empty text for failed pages

            # Filter out None values and combine all text
            all_text = [text for text in all_text if text is not None]
            combined_text = "\n\n".join(all_text)
            logging.info("Combined text from all pages completed")
            logging.info("image to text:%s", combined_text[:1000])  # Log first 1000 chars

            # Call extraction functions ONCE on the combined text
            logging.info("Starting extraction process...")
            details = extract_death_certificate_details(combined_text)
            logging.info(f"extract_death_certificate_details details:{details}")
            print("Step 3: Displaying extracted details...")
            print_extracted_details(details)
            
            return details

        except Exception as e:
            logging.error("Error in PDF to image conversion:%s", e)
            # Fallback to direct OCR
            combined_text = image_to_text(pdf_path)
            logging.info("Fallback image to text:%s", combined_text)
            
            if combined_text and combined_text != "No text":
                details = extract_death_certificate_details(combined_text)
                logging.info(f"Fallback extract_death_certificate_details details:{details}")
                print_extracted_details(details)
                return details
            else:
                return DeathCertificateDetails()  # Return empty details object

    except Exception as e:
        logging.error("Error in pdf to text function:%s", e)
        return DeathCertificateDetails()  # Return empty details object