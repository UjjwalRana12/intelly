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
    """Pydantic model for comprehensive death certificate and probate information"""
    
    petitioner_full_name: Optional[str] = Field(None, description="Petitioner's full name")
    petitioner_address: Optional[str] = Field(None, description="Petitioner's address")
    petitioner_phone_number: Optional[str] = Field(None, description="Petitioner's phone number")
    petitioner_date_of_birth: Optional[str] = Field(None, description="Petitioner's date of birth")
    petitioner_relationship_to_decedent: Optional[str] = Field(None, description="Petitioner's relationship to the decedent")
    
    
    decedent_full_name: Optional[str] = Field(None, description="Full name of the deceased person")
    date_of_birth: Optional[str] = Field(None, description="Date of birth")
    date_of_death: Optional[str] = Field(None, description="Date of death")
    location_of_death: Optional[str] = Field(None, description="Place where death occurred")
    county: Optional[str] = Field(None, description="County where death occurred")
    last_4_digits_ssn: Optional[str] = Field(None, description="Last 4 digits of Social Security Number")
    drivers_license_number: Optional[str] = Field(None, description="Driver's License Number or State-Issued ID Number")
    passport_number: Optional[str] = Field(None, description="Passport Number")
    other_identifying_details: Optional[str] = Field(None, description="Other identifying details")
    time_of_death: Optional[str] = Field(None, description="Time when death occurred")
    
    
    estimated_value_real_estate: Optional[str] = Field(None, description="Estimated value of the decedent's real estate")
    estimated_value_personal_estate: Optional[str] = Field(None, description="Estimated value of personal estate (other assets)")
    
    
    application_previously_filed: Optional[str] = Field(None, description="Was an application previously filed, and was a personal representative appointed informally?")
    personal_representative_previously_appointed: Optional[str] = Field(None, description="Has a personal representative been previously appointed?")
    representative_full_name: Optional[str] = Field(None, description="Representative's full name")
    representative_relationship_to_decedent: Optional[str] = Field(None, description="Representative's relationship to the decedent")
    representative_address: Optional[str] = Field(None, description="Representative's address")
    representative_city_state_zip: Optional[str] = Field(None, description="Representative's city, state, zip")
    
    
    date_of_decedent_will: Optional[str] = Field(None, description="Date of decedent's will")
    date_of_decedent_codicil: Optional[str] = Field(None, description="Date of decedent's codicil")
    will_and_codicils_offered_for_probate: Optional[str] = Field(None, description="Is/are the will and codicils offered for probate?")
    authenticated_copy_of_will_and_codicil: Optional[str] = Field(None, description="Is there any authenticated copy of the will and codicil?")
    
    
    type_of_fiduciary: Optional[str] = Field(None, description="Type of fiduciary")
    period_of_fiduciary_service: Optional[str] = Field(None, description="Period of fiduciary service")
    description_of_real_property_or_business_interest: Optional[str] = Field(None, description="Description of real property or business interest")
    mailing_address_of_informant: Optional[str] = Field(None, description="Mailing address of informant")
    
    @validator('last_4_digits_ssn')
    def validate_last_4_ssn(cls, v):
        if v is None:
            return v
        # Extract only digits and take last 4
        cleaned_ssn = re.sub(r'[^\d]', '', str(v))
        if len(cleaned_ssn) >= 4:
            return cleaned_ssn[-4:]
        return v
    
    @validator('date_of_birth', 'date_of_death', 'petitioner_date_of_birth', 'date_of_decedent_will', 'date_of_decedent_codicil')
    def validate_date_format(cls, v):
        if v is None:
            return v
        return str(v).strip()
    
    @validator('time_of_death')
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
        # Create a comprehensive prompt for extracting all information
        prompt = f"""
        You are an expert at extracting information from death certificates and probate documents. 
        Please extract the following specific details from the given text:

        PETITIONER INFORMATION:
        1. Petitioner's Full Name
        2. Petitioner's Address
        3. Petitioner's Phone Number
        4. Petitioner's Date of Birth
        5. Petitioner's Relationship to the Decedent

        DECEDENT INFORMATION:
        6. Decedent's Full Name (full name of the deceased)
        7. Date of Birth
        8. Date of Death
        9. Location of Death (Just Return What is Written in the Document)
        10. County (county where death occurred)
        11. Last 4 Digits of the Social Security Number
        12. Driver's License Number or State-Issued ID Number
        13. Passport Number
        14. Other Identifying Details
        15. Time of Death

        ESTATE INFORMATION:
        16. Estimated Value of the Decedent's Real Estate
        17. Estimated Value of the Personal Estate (Other Assets)

        PREVIOUS APPLICATIONS AND REPRESENTATIVES:
        18. Was an application previously filed, and was a personal representative appointed informally?
        19. Has a personal representative been previously appointed?
        20. Representative's Full Name
        21. Representative's Relationship to the Decedent
        22. Representative's Address
        23. Representative's City, State, Zip

        WILL AND CODICIL INFORMATION:
        24. Date of Decedent's Will
        25. Date of Decedent's Codicil
        26. Is/are the will and codicils offered for probate?
        27. Is there any authenticated copy of the will and codicil?

        FIDUCIARY INFORMATION:
        28. Type of Fiduciary
        29. Period of Fiduciary Service
        30. Description of Real Property or Business Interest
        31. Mailing Address of Informant

        Text to analyze:
        {text}

        Please respond in the following JSON format:
        {{
            "petitioner_full_name": "extracted name or null",
            "petitioner_address": "extracted address or null",
            "petitioner_phone_number": "extracted phone or null",
            "petitioner_date_of_birth": "extracted date or null",
            "petitioner_relationship_to_decedent": "extracted relationship or null",
            "decedent_full_name": "extracted name or null",
            "date_of_birth": "extracted date or null",
            "date_of_death": "extracted date or null",
            "location_of_death": "extracted location or null",
            "county": "extracted county or null",
            "last_4_digits_ssn": "extracted last 4 digits or null",
            "drivers_license_number": "extracted license number or null",
            "passport_number": "extracted passport number or null",
            "other_identifying_details": "extracted details or null",
            "time_of_death": "extracted time or null",
            "estimated_value_real_estate": "extracted value or null",
            "estimated_value_personal_estate": "extracted value or null",
            "application_previously_filed": "extracted answer or null",
            "personal_representative_previously_appointed": "extracted answer or null",
            "representative_full_name": "extracted name or null",
            "representative_relationship_to_decedent": "extracted relationship or null",
            "representative_address": "extracted address or null",
            "representative_city_state_zip": "extracted city state zip or null",
            "date_of_decedent_will": "extracted date or null",
            "date_of_decedent_codicil": "extracted date or null",
            "will_and_codicils_offered_for_probate": "extracted answer or null",
            "authenticated_copy_of_will_and_codicil": "extracted answer or null",
            "type_of_fiduciary": "extracted type or null",
            "period_of_fiduciary_service": "extracted period or null",
            "description_of_real_property_or_business_interest": "extracted description or null",
            "mailing_address_of_informant": "extracted address or null"
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
            max_tokens=2000,
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
        
        logging.info("Successfully extracted and validated comprehensive details")
        return details

    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON response: {e}")
        return DeathCertificateDetails()
    except Exception as e:
        logging.error(f"Error extracting details with LLM: {e}")
        return DeathCertificateDetails()

def print_extracted_details(details: DeathCertificateDetails):
    """Print the extracted details in a formatted way"""
    print(f"\n{'='*60}")
    print("EXTRACTED COMPREHENSIVE DETAILS")
    print(f"{'='*60}")
    
    # Convert Pydantic model to dict for easy iteration
    details_dict = details.dict()
    
    # Group fields for better display
    sections = {
        "PETITIONER INFORMATION": [
            "petitioner_full_name", "petitioner_address", "petitioner_phone_number",
            "petitioner_date_of_birth", "petitioner_relationship_to_decedent"
        ],
        "DECEDENT INFORMATION": [
            "decedent_full_name", "date_of_birth", "date_of_death", "location_of_death",
            "county", "last_4_digits_ssn", "drivers_license_number", "passport_number",
            "other_identifying_details", "time_of_death"
        ],
        "ESTATE INFORMATION": [
            "estimated_value_real_estate", "estimated_value_personal_estate"
        ],
        "REPRESENTATIVES": [
            "application_previously_filed", "personal_representative_previously_appointed",
            "representative_full_name", "representative_relationship_to_decedent",
            "representative_address", "representative_city_state_zip"
        ],
        "WILL AND CODICIL": [
            "date_of_decedent_will", "date_of_decedent_codicil",
            "will_and_codicils_offered_for_probate", "authenticated_copy_of_will_and_codicil"
        ],
        "FIDUCIARY INFORMATION": [
            "type_of_fiduciary", "period_of_fiduciary_service",
            "description_of_real_property_or_business_interest", "mailing_address_of_informant"
        ]
    }
    
    for section_name, fields in sections.items():
        print(f"\n{section_name}:")
        print("-" * len(section_name))
        for field in fields:
            if field in details_dict:
                formatted_key = field.replace("_", " ").title()
                print(f"{formatted_key:<40}: {details_dict[field] or 'Not found'}")
    
    print(f"\n{'='*60}")
    
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
        # Use json.dumps for formatting instead of details.json(indent=2)
        import json
        f.write(json.dumps(details.dict(), indent=2))
    
    print(f"Extracted details saved to '{filename}'")

# if __name__ == "__main__":
    
#     logging.basicConfig(level=logging.INFO)
    
    
#     if os.path.exists("beverly.txt"):
#         with open("beverly.txt", "r", encoding="utf-8") as f:
#             text = f.read()
        
#         print("Extracting details from existing text file...")
#         details = extract_death_certificate_details(text)
#         print_extracted_details(details)
        
#         # Save with extracted details
#         save_extracted_details(details, "beverly_details.txt", text)
#     else:
#         print("beverly.txt not found. Please run main.py first to extract text from image.")