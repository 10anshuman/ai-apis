from dotenv import load_dotenv
import os
import json
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

if api_key is None:
    raise ValueError("GROQ_API_KEY is not set in the .env file.")

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=60,
    max_retries=2,
)

def extract_json(content):
    try:
        # Try to find the JSON part in the raw response
        start = content.find("{")
        end = content.rfind("}") + 1
        if start == -1 or end == -1:
            raise ValueError("No valid JSON block found in the AI response.")
        json_str = content[start:end].strip()
        return json.loads(json_str)
    except Exception as e:
        print("Error extracting JSON:", str(e))
        raise ValueError("Failed to extract valid JSON from AI response.")

def get_ai_msg(patient_prompt):
    messages = [
        {
            "role": "system",
            "content": """
              You are a medical assistant from INDIA country capable of extracting structured details from unstructured clinical text. Your task is to identify specific categories of information from the provided text.
            The categories you need to extract are:
            1. **Status**: Identify the patient's health status (e.g., "S" for stable, "C" for critical).
            2. **Pharmacy**: Extract detailed medication information.
            3. **Services**: Extract medical services/tests.
            4. **Diagnosis**: Extract the patient's diagnosis.
            5. **Allergies**: Extract the patient's allergies.
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
                        "Name": "",
                        "ICD Code": "" (if not provided then please extract it using the internet)
                    }
                    ],
                    "allergies": [
                        {
                            "Name": "",
                            "ICD Code": "" (if not provided then please extract it using the internet)
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