from dotenv import load_dotenv
import os
import json
import spacy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pandas as pd
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Get the API key from .env file
api_key = os.getenv("GROQ_API_KEY")

# Define headers for API requests
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Check if API key is provided
if api_key is None:
    raise ValueError("GROQ_API_KEY is not set in the .env file.")

# Initialize the ChatGroq model
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=60,
    max_retries=2,
)
nlp = spacy.load("en_core_web_sm")
medicine_dataset = pd.read_csv("medicine_dataset.csv") 
medicine_names = medicine_dataset["Medicine Name"].tolist() 

# Function to correct medicine names using SpaCy and FuzzyWuzzy
def correct_medicine_names(extracted_medicines):
    corrected_medicines = []
    for medicine in extracted_medicines:
        doc = nlp(medicine)
        for token in doc:
            best_match, similarity = process.extractOne(token.text, medicine_names, scorer=fuzz.ratio)
            if similarity > 85:  
                corrected_medicines.append(best_match)
            else:
                corrected_medicines.append(token.text)
    return corrected_medicines
def extract_json(content):
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end == -1:
            raise ValueError("No valid JSON block found in the AI response.")
        json_str = content[start:end].strip()
        extracted_data = json.loads(json_str)
        if "pharmacy" in extracted_data:
            for medicine_entry in extracted_data["pharmacy"]:
                if "Medicine Name" in medicine_entry:
                    medicine_name = medicine_entry["Medicine Name"]
                    corrected_name = correct_medicine_names([medicine_name])
                    medicine_entry["Medicine Name"] = corrected_name[0]
        
        return extracted_data
    except Exception as e:
        print("Error extracting JSON:", str(e))
        raise ValueError("Failed to extract valid JSON from AI response.")

def get_ai_msg(patient_prompt):
    messages = [
        {
            "role": "system",
            "content": """
              You are a medical assistant from INDIA country capable of extracting structured details from unstructured clinical text. Your task is to identify specific categories of information from the provided text and if you get any other language other than english then please translate it to english. and provide the structured details in a JSON format.
            The categories you need to extract are:
            1. **Status**: Identify the patient's health status (e.g., "S" for stable, "C" for critical).
            2. **Pharmacy**: Extract detailed medication information. (Extract each name separately)
            3. **Services**: Extract medical services/tests. (Extract each name separately)
            4. **Diagnosis**: Extract the patient's diagnosis. (Extract each name separately)
            5. **Allergies**: Extract the patient's allergies. (Extract each name separately)
            6. **Chief Complaint**: Extract the patient's chief complaint.
            7. **Summary**: Summarize the patient's condition.
            Provide the response in the following structured JSON format and in the same order as the categories above:
            {
                    "status": "S",
                    "pharmacy": [
                        {
                            "Medicine Name": "",
                            "Dosage Amount": "",
                            "Dosage Unit": "",
                            "Frequency": "",
                            "Time": "",
                            "Duration": "",
                            "Instructions": ""
                        }
                    ],
                    "service": [
                        {
                            "Test or Service Name": "",
                            "Date & Time": "",
                            "Instructions": ""
                        }
                    ],
                    "diagnosis":[ 
                    {
                        "diagnosis_name": "",
                        "ICD_Code": "" (if not provided then please extract it using the internet)
                    }
                    ],
                    "allergies": [
                        {
                            "allergies_name": "",
                            "ICD_Code": "" (if not provided then please extract it using the internet)
                        }
                    ],
                    "Chief Complaint": [ 
                    {
                        "Chief Complaint": "(Provide the chief complaint of the patient from the text.)",
                        "Duration": "Extract the duration of the chief complaint."        
                    }
                    ],
                    "Summary": "(Considering the above details, please provide a summary of the patient's condition.)"
                }
            }
            """
        },
        {
            "role": "user",
            "content": patient_prompt
        }
    ]

    ai_msg = llm.invoke(messages)
    # return ai_msg.content
    # print("Raw AI Response:", ai_msg.content)
    return extract_json(ai_msg.content)