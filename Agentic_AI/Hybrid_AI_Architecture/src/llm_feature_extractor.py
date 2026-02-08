import json
import requests
from config import OLLAMA_URL, LLM_MODEL

def extract_fincrime_features(text: str):

    prompt = f"""
You are an AML investigator.

Extract structured financial crime indicators.
Return ONLY valid JSON.

Required JSON format:
{{
"structuring_risk": float 0-1,
"mule_account_risk": float 0-1,
"offshore_risk": float 0-1,
"rapid_movement_of_funds": float 0-1,
"third_party_usage": float 0-1,
"urgency": integer 0-3,
"suspicious_entities": integer
}}

Narrative:
{text}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "temperature": 0
        },
        timeout=120
    )

    output = response.json()["response"]

    try:
        start = output.find("{")
        end = output.rfind("}") + 1
        return json.loads(output[start:end])
    except:
        return {
            "structuring_risk": 0,
            "mule_account_risk": 0,
            "offshore_risk": 0,
            "rapid_movement_of_funds": 0,
            "third_party_usage": 0,
            "urgency": 0,
            "suspicious_entities": 0
        }
