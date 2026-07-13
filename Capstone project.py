# %%
import os
import json
import requests

from dotenv import load_dotenv
from jsonschema import validate, ValidationError

load_dotenv()

api_key = os.getenv("LLM_API_KEY")

print("API Loaded:", api_key is not None)

# %%
def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        print("Error:", response.status_code)
        print(response.text)
        return None

    return response.json()["choices"][0]["message"]["content"]

# %%
system_prompt = "You are a helpful assistant."

user_prompt = "Reply with only the word: hello"

response = call_llm(system_prompt, user_prompt)

print(response)

# %%
import os

print(os.getcwd())

# %%
import pandas as pd

df = pd.read_csv("../outputs/cleaned_data.csv")

print(df.shape)
df.head()

# %%
# Select three patient records

records = df.iloc[[0, 1, 2]]

records

# %%
import json

record_list = records.to_dict(orient="records")

print(json.dumps(record_list[0], indent=4))

# %%
system_prompt = """
You are an Alzheimer's patient risk assessment assistant.

Your task is to assess one patient record and return ONLY valid JSON.

Use the following rubric:

- risk_tier: low, medium, or high
- flag_for_review: true or false
- primary_signal: the strongest reason for the assessment
- confidence: low, medium, or high
- recommended_action: one short recommendation

Return ONLY valid JSON.

Worked Example

Input:
{
 "Age":82,
 "MMSE":15,
 "MemoryComplaints":1,
 "Confusion":1,
 "Diagnosis":1
}

Output:
{
 "risk_tier":"high",
 "flag_for_review":true,
 "primary_signal":"Low MMSE with confirmed diagnosis",
 "confidence":"high",
 "recommended_action":"Refer to neurologist for comprehensive evaluation"
}
"""

# %%
user_prompt = f"""
Assess the following patient record.

Return ONLY valid JSON.

Patient Record:

{json.dumps(record_list[0], indent=2)}
"""

# %%
response = call_llm(
    system_prompt,
    user_prompt,
    temperature=0
)

print(response)

# %%
schema = {
    "type": "object",
    "properties": {
        "risk_tier": {
            "type": "string"
        },
        "flag_for_review": {
            "type": "boolean"
        },
        "primary_signal": {
            "type": "string"
        },
        "confidence": {
            "type": "string"
        },
        "recommended_action": {
            "type": "string"
        }
    },
    "required": [
        "risk_tier",
        "flag_for_review",
        "primary_signal",
        "confidence",
        "recommended_action"
    ]
}

# %%
response = response.strip()

print(response)

# %%
response = response.replace("```json", "")
response = response.replace("```", "")
response = response.strip()

# %%
assessment = json.loads(response)

assessment

# %%
try:
    validate(instance=assessment, schema=schema)
    print("✅ Validation Passed")
except ValidationError as e:
    print("❌ Validation Failed")
    print(e)

# %%
fallback = {
    "risk_tier": None,
    "flag_for_review": None,
    "primary_signal": None,
    "confidence": None,
    "recommended_action": None
}

try:
    assessment = json.loads(response)
    validate(instance=assessment, schema=schema)
    print("✅ Validation Passed")
    print(assessment)

except (json.JSONDecodeError, ValidationError) as e:
    print("❌ Validation Failed")
    print(e)
    assessment = fallback

print("\nFinal Output:")
print(assessment)

# %%
import re

def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'

    return bool(
        re.search(email_pattern, text) or
        re.search(phone_pattern, text)
    )

# %%
test1 = "Email: rethi@gmail.com"

test2 = "Age:73 BMI:22 MMSE:21"

print("Test 1:", has_pii(test1))
print("Test 2:", has_pii(test2))

# %%
user_input = "Email: rethi@gmail.com"

if has_pii(user_input):
    print("Input blocked: PII detected.")
else:
    print(call_llm(system_prompt, user_input))

# %%
user_input = json.dumps(record_list[1], indent=2)

if has_pii(user_input):
    print("Input blocked: PII detected.")
else:
    print(call_llm(system_prompt, user_input))

# %%
for i, record in enumerate(record_list):

    user_prompt = f"""
Assess the following patient.

Return ONLY valid JSON.

{json.dumps(record, indent=2)}
"""

    response = call_llm(system_prompt, user_prompt)

    print("=" * 60)
    print(f"Patient {i+1}")
    print(response)

# %%
patient = json.dumps(record_list[0], indent=2)

response0 = call_llm(system_prompt, patient, temperature=0)

response07 = call_llm(system_prompt, patient, temperature=0.7)

print("Temperature = 0")
print(response0)

print("\n")

print("Temperature = 0.7")
print(response07)

# %%
patient = json.dumps(record_list[0], indent=2)

response0 = call_llm(system_prompt, patient, temperature=0)

response07 = call_llm(system_prompt, patient, temperature=0.7)

print("Temperature = 0")
print(response0)

print("\n")

print("Temperature = 0.7")
print(response07)

# %%
results = []

for i, record in enumerate(record_list):

    user_prompt = f"""
Assess the following patient record.

Return ONLY valid JSON.

{json.dumps(record, indent=2)}
"""

    if has_pii(user_prompt):
        results.append({
            "Patient": i + 1,
            "LLM_Output": "Blocked",
            "Validation": "Not Run",
            "Guardrail": "Blocked"
        })
        continue

    response = call_llm(system_prompt, user_prompt, temperature=0)

    clean_response = response.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(clean_response)
        validate(instance=parsed, schema=schema)

        validation = "Pass"

    except Exception:
        parsed = fallback
        validation = "Fail"

    results.append({
        "Patient": i + 1,
        "LLM_Output": parsed,
        "Validation": validation,
        "Guardrail": "Passed"
    })

print(results)

# %%
results_df = pd.DataFrame(results)

results_df


