from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
token = os.environ["openai_api_key2"]
# client = OpenAI(api_key=token)

# response = client.responses.create(
#     model="gpt-5",
#     input=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "input_text",
#                     "text": "What teams are playing in this image?",
#                 },
#                 {
#                     "type": "input_image",
#                     "image_url": "https://c8.alamy.com/comp/2RXDX75/low-priority-rubber-stamp-seal-vector-2RXDX75.jpg"
#                 }
#             ]
#         }
#     ]
# )

# print(response.output_text)

# from openai import OpenAI
import base64

client = OpenAI(api_key=token) 

response = client.responses.create(
    model="gpt-4.1-mini",
    input="Generate an image of gray tabby cat hugging an otter with an orange scarf",
    tools=[{"type": "image_generation"}],
)

#  Save the image to a file
image_data = [
    output.result
    for output in response.output
    if output.type == "image_generation_call"
]

if image_data:
    image_base64 = image_data[0]
    with open("cat_and_otter.png", "wb") as f:
        f.write(base64.b64decode(image_base64))