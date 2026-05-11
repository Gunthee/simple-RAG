import requests
from openai import OpenAI

#setup LLM
TYPHOON_API_KEY = "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" #Replace with your actual API key
TYPHOON_BASE_URL = "https://api.opentyphoon.ai/v1"
TYPHOON_MODEL = "typhoon-v2.5-30b-a3b-instruct"


def call_typhoon(prompt):
    
    client = OpenAI(
    api_key= TYPHOON_API_KEY,
    base_url= TYPHOON_BASE_URL
    )

    response = client.chat.completions.create(
    model = TYPHOON_MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
        ],
    )

    return response.choices[0].message.content

prompt = "ช่วยคิดชื่อแมวภาษาไทยให้หน่อยได้ไหม"
response = call_typhoon(prompt)
print(response)