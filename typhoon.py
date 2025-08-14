import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import requests
import json
from dotenv import load_dotenv
load_dotenv()
import os
token = os.environ["typhoon_api_key"]
def extract_text_from_image(image_path, api_key, params):
    url = "https://api.opentyphoon.ai/v1/ocr"
    
    with open(image_path, 'rb') as file:
        files = {'file': file}
        data = {
            'params': json.dumps(params)
        }
        
        headers = {
            'Authorization': f'Bearer {api_key}'
        }
        
        response = requests.post(url, files=files, data=data, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            usage = result.get("usage") or {}
            print(
                "[usage] prompt =", usage.get("prompt_tokens"),
                "completion =", usage.get("completion_tokens"),
                "total =", usage.get("total_tokens"),
            )
            # Extract text from successful results
            extracted_texts = []
            for page_result in result.get('results', []):
                if page_result.get('success') and page_result.get('message'):
                    content = page_result['message']['choices'][0]['message']['content']
                    try:
                        # Try to parse as JSON if it's structured output
                        parsed_content = json.loads(content)
                        text = parsed_content.get('natural_text', content)
                    except json.JSONDecodeError:
                        text = content
                    extracted_texts.append(text)
                elif not page_result.get('success'):
                    print(f"Error processing {page_result.get('filename', 'unknown')}: {page_result.get('error', 'Unknown error')}")
            
            return '\n'.join(extracted_texts)
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None

# Usage
api_key = "<YOUR_API_KEY>"
image_path = "https://static.thairath.co.th/media/FcvsRgKyX10OHanMmOPrFKepuCKbynOAPg80XISIM4bqOmvbhkAgGXfWu8.webp"  # or path/to/your/document.pdf
model = "typhoon-ocr-preview"
params = {
    "model": model,
    "task_type": "default",
    "max_tokens": 16000,
    "temperature": 0.1,
    "top_p": 0.6,
    "repetition_penalty": 1.2
}
extracted_text = extract_text_from_image(image_path, token, params)
print(extracted_text)

