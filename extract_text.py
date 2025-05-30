import os
import logging
import json
import re
from typing import Optional
from pydantic import BaseModel, Field, validator
from openai import OpenAI
from dotenv import load_dotenv

# Initialize OpenAI client
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class DeathCertificateDetails(BaseModel):
    """Pydantic model for death certificate information"""
    decedent_name: Optional[str] = Field(None, description="Full name of the deceased person")
    date_of_birth: Optional[str] = Field(None, description="Date of birth in MM/DD/YYYY format")
    date_of_death: Optional[str] = Field(None, description="Date of death in MM/DD/YYYY format")
    location_of_death: Optional[str] = Field(None, description="Place where death occurred")
    county: Optional[str] = Field(None, description="County where death occurred")
    social_security_number: Optional[str] = Field(None, description="Social Security Number")
    time_pronounced_dead: Optional[str] = Field(None, description="Time when death was pronounced")
    
    @validator('social_security_number')
    def validate_ssn(cls, v):
        if v is None:
            return v
        # Remove any formatting and validate SSN pattern
        cleaned_ssn = re.sub(r'[^\d]', '', str(v))
        if len(cleaned_ssn) == 9:
            return f"{cleaned_ssn[:3]}-{cleaned_ssn[3:5]}-{cleaned_ssn[5:]}"
        return v
    
    @validator('date_of_birth', 'date_of_death')
    def validate_date_format(cls, v):
        if v is None:
            return v
        # Basic date validation - you can make this more sophisticated
        date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        if re.match(date_pattern, str(v)):
            return v
        return v
    
    @validator('time_pronounced_dead')
    def validate_time_format(cls, v):
        if v is None:
            return v
        # Basic time validation
        time_pattern = r'\d{1,2}:\d{2}(?:\s*[ap]m)?'
        if re.search(time_pattern, str(v).lower()):
            return v
        return v

def extract_death_certificate_details(text: str) -> DeathCertificateDetails:
    """Use LLM to extract specific details from death certificate text with Pydantic validation"""
    try:
        # Create a prompt for extracting specific information
        prompt = f"""
        You are an expert at extracting information from death certificates. 
        Please extract the following specific details from the given text:

        1. Decedent's name (full name of the deceased)
        2. Date of birth (in MM/DD/YYYY format if possible)
        3. Date of death (in MM/DD/YYYY format if possible)
        4. Location of death (hospital, home, etc.)
        5. County (county where death occurred)
        6. Social security number (XXX-XX-XXXX format)
        7. Time pronounced dead (HH:MM AM/PM format if possible)

        Text to analyze:
        {text}

        Please respond in the following JSON format:
        {{
            "decedent_name": "extracted name or null",
            "date_of_birth": "extracted date or null", 
            "date_of_death": "extracted date or null",
            "location_of_death": "extracted location or null",
            "county": "extracted county or null",
            "social_security_number": "extracted SSN or null",
            "time_pronounced_dead": "extracted time or null"
        }}

        If any information is not found, use null. Only return the JSON, no additional text.
        Be very careful to extract exact information as it appears in the document.
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1000,
            temperature=0,
        )

        # Parse the JSON response
        result = response.choices[0].message.content.strip()
        
        # Clean up the response if it has markdown formatting
        if result.startswith("```json"):
            result = result.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON and validate with Pydantic
        raw_data = json.loads(result)
        details = DeathCertificateDetails(**raw_data)
        
        logging.info("Successfully extracted and validated death certificate details")
        return details

    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON response: {e}")
        return DeathCertificateDetails()
    except Exception as e:
        logging.error(f"Error extracting details with LLM: {e}")
        return DeathCertificateDetails()

def print_extracted_details(details: DeathCertificateDetails):
    """Print the extracted details in a formatted way"""
    print(f"\n{'='*50}")
    print("EXTRACTED DEATH CERTIFICATE DETAILS")
    print(f"{'='*50}")
    
    # Convert Pydantic model to dict for easy iteration
    details_dict = details.dict()
    
    for key, value in details_dict.items():
        formatted_key = key.replace("_", " ").title()
        print(f"{formatted_key:<25}: {value or 'Not found'}")
    
    print(f"{'='*50}")
    
    # Show validation errors if any
    try:
        details.dict()  # This will raise validation errors if any
    except Exception as e:
        print(f"Validation warnings: {e}")

def save_extracted_details(details: DeathCertificateDetails, filename: str, raw_text: str = None):
    """Save extracted details to a file"""
    with open(filename, "w", encoding="utf-8") as f:
        if raw_text:
            f.write("RAW EXTRACTED TEXT:\n")
            f.write("="*50 + "\n")
            f.write(raw_text)
            f.write("\n\n")
        
        f.write("EXTRACTED DETAILS:\n")
        f.write("="*50 + "\n")
        
        # Convert Pydantic model to dict for saving
        details_dict = details.dict()
        for key, value in details_dict.items():
            formatted_key = key.replace("_", " ").title()
            f.write(f"{formatted_key}: {value or 'Not found'}\n")
        
        # Save as JSON for easy parsing later
        f.write("\n\nJSON FORMAT:\n")
        f.write("="*30 + "\n")
        f.write(details.json(indent=2))
    
    print(f"Extracted details saved to '{filename}'")

if __name__ == "__main__":
    # Example usage - read from existing text file
    logging.basicConfig(level=logging.INFO)
    
    # Try to read from beverly.txt if it exists
    if os.path.exists("beverly.txt"):
        with open("beverly.txt", "r", encoding="utf-8") as f:
            text = f.read()
        
        print("Extracting details from existing text file...")
        details = extract_death_certificate_details(text)
        print_extracted_details(details)
        
        # Save with extracted details
        save_extracted_details(details, "beverly_details.txt", text)
    else:
        print("beverly.txt not found. Please run main.py first to extract text from image.")