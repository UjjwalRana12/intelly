def image_to_text(image_path):
    try:
 
        # image_url = base_url + "/" + image_path.replace("\\", "/")
 
        base64_img = f"data:image/png;base64,{encode_image(image_path)}"
       
   
        # Send the image as base64 in a chat completion request
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",  # Specify the model
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
   
        # Extract and return the plain text content from the response
        plain_text_content = response["choices"][0]["message"]["content"]
        print(plain_text_content)
        logging.info("plain_text_content extracted :%s", plain_text_content)
        logging.info("OCR applied successfully using openai in image :%s", image_path)
        return plain_text_content
    except Exception as e:
        logging.info("Error in ocr for using openai in image :%s", image_path)
        logging.info("Error in ocr for using openai :%s", e)
        return "No text"
 
 
form_questions_for_pretrail = [
    "Petitioner's Full Name",
    "Address",
    "Phone Number",
    "Petitioner's Date of Birth",
    "Petitioner's Relationship to the Decedent",
    "Decedent's Full Name",
    "Date of Birth",
    "Date of Death",
    "Location of Death",
    "County",
    "Last 4 Digits of the Social Security Number",
    "Driver's License Number or State-Issued ID Number",
    "Passport Number",
    "Other Identifying Details",
    "Estimated Value of the Decedent's Real Estate",
    "Estimated Value of the Personal Estate (Other Assets)",
    "Time of Death",
    "Was an application previously filed, and was a personal representative appointed informally?",
    "Has a personal representative been previously appointed?",
    "Representative's Full Name",
    "Representative's Relationship to the Decedent",
    "Representative's Address",
    "Representative's City, State, Zip",
    "Date of Decedent's Will",
    "Date of Decedent's Codicil",
    "Is/are the will and codicils offered for probate?",
    "Is there any authenticated copy of the will and codicil?",
    "Type of Fiduciary",
    "Period of Fiduciary Service",
    "Description of Real Property or Business Interest",
    "Mailing Address of Informant"
]
 
def get_form_ques_data(form_ques, combined_text):
 
    try:
        extraction_prompt = ChatPromptTemplate.from_template(f"""
            You are an expert in processing Probate Case Documents like Death Certificate or Will. Based on the document text provided, extract:
            {form_ques}
 
            If data is not provided in the text for any question, please generate 'Not provided' only.
            Document Text:
            {{text}}
 
            Generate the response in the following format:
            {form_ques[0]}: Value 1
            {form_ques[1]}: Value 2
 
        """)
 
 
        # Create the LLM model instance
        llm = ChatOpenAI(temperature=0, model='gpt-4o')
 
        # Create the LLM chain with the prompt
        response_chain = LLMChain(llm=llm, prompt=extraction_prompt)
 
        # Run the text through the chain and get the response
        response = response_chain.run({"text": combined_text})
        logging.info("response:%s", response)
 
    except Exception as e:
        logging.info("error in extracting form questions answer: %s", e)
 
 
    print("generalized patttern match")
    # Step 6: Generate dynamic patterns for each field
    try:
        extracted_info = {}
        not_matched={}
        for field in form_ques:
            #field_name = field.strip().split("(")[0].strip()  # Strip any extra info like (DOB) or (DOD)
            field_name = field.strip()
            print(field_name)
           
            #logging.info("field names: %s",field_name)
            #regex_pattern = rf"{field_name}:\s*([^\n]+)"
            regex_pattern = rf"{re.escape(field_name.split('(')[0].strip())}(?: \(.*?\))?:\s*([^\n]+)"
           
            # field_name = field.split('?')[0].strip()  # Remove question mark for label matching
            # logging.info("field names: %s",field_name)
            # regex_pattern = rf"{re.escape(field_name)}:\s*([^\n]+)"
            # Find match for each field in the response
            match = re.search(regex_pattern, response, re.IGNORECASE)
            logging.info(f"matched regex:==>{match}")
            #logging.info("match:%s",  match.group(1).strip())
            # s = match.group(1).strip()
            if match :
                extracted_info[field_name] = match.group(1).strip()
            else:
                not_matched[field_name] = "Not Matched"
                extracted_info[field_name] = ""
           
 
 
        # Step 7: Print and return the extracted information
        for field, value in extracted_info.items():
            print(f"{field}: {value}")
    except Exception as e:
        logging.info("error in regex pattern atching : %s", e)
 
    return extracted_info,not_matched

