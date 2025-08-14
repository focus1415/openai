import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    
from dotenv import load_dotenv
load_dotenv()
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    SystemMessage, UserMessage,
    TextContentItem, ImageContentItem, ImageUrl
)
from azure.core.credentials import AzureKeyCredential

endpoint = "https://models.github.ai/inference"   # ถูกต้องแล้ว (endpoint ใหม่นี้แทนตัวเก่า)
model = "openai/gpt-4.1"                          # รุ่นที่รองรับภาพ/มัลติโหมด
token = os.environ["GITHUB_TOKEN"]

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

# แนบภาพจากไฟล์ในเครื่อง
# img = ImageUrl.load(image_file="https://cdn.pixabay.com/photo/2018/08/04/11/30/draw-3583548_1280.png", image_format="jpeg")
img = ImageUrl(url="https://static.thairath.co.th/media/FcvsRgKyX10OHanMmOPrFKepuCKbynOAPg80XISIM4bqOmvbhkAgGXfWu8.webp")
# หากมี URL ของรูป ก็ใช้แทนได้ (ปลดคอมเมนต์บรรทัดล่าง)
# img = ImageUrl(url="https://example.com/photo.jpg")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    UserMessage(content=[
        TextContentItem(text="Perform OCR on all visible text, detect the language"),#what is this picture , and translate to thai
        ImageContentItem(image_url=img),
    ]),
]

resp = client.complete(messages=messages, model=model, max_tokens=300)

# บางครั้ง content จะเป็น list ของชิ้นส่วนข้อความ
content = resp.choices[0].message.content
if isinstance(content, list):
    print("".join(getattr(c, "text", "") for c in content))
else:
    print(content)
# แสดงจำนวน token ที่ใช้
resp = client.complete(messages=messages, model=model)
u = resp.usage
print("prompt:", u.prompt_tokens, "completion:", u.completion_tokens, "total:", u.total_tokens)