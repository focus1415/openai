import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()
import os
import time
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    SystemMessage, UserMessage,
    TextContentItem, ImageContentItem, ImageUrl
)
from azure.core.credentials import AzureKeyCredential

endpoint = "https://models.github.ai/inference"   # endpoint ใหม่
model = "openai/gpt-4.1"                          # รุ่นมัลติโหมด
token = os.environ["GITHUB_TOKEN"]

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

# แนบภาพจาก URL (แนะนำให้ใช้ URL ที่เข้าถึงได้สาธารณะ)
img = ImageUrl(url="https://static.thairath.co.th/media/FcvsRgKyX10OHanMmOPrFKepuCKbynOAPg80XISIM4bqOmvbhkAgGXfWu8.webp")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    UserMessage(content=[
        TextContentItem(text="Perform OCR on all visible text, detect the language"),
        ImageContentItem(image_url=img),
    ]),
]

# ====== จับเวลารวมทั้งสคริปต์ ======
t_total_start = time.perf_counter()

# ====== จับเวลาเฉพาะการเรียก API ======
t_api_start = time.perf_counter()
resp = client.complete(messages=messages, model=model, max_tokens=300)
t_api = time.perf_counter() - t_api_start

# แสดงผลลัพธ์ข้อความ
content = resp.choices[0].message.content
if isinstance(content, list):
    print("".join(getattr(c, "text", "") for c in content))
else:
    print(content)

# แสดงจำนวน token ที่ใช้ (ไม่ต้องเรียก API ซ้ำ)
u = getattr(resp, "usage", None)
if u:
    print(f"prompt: {getattr(u, 'prompt_tokens', None)} "
          f"completion: {getattr(u, 'completion_tokens', None)} "
          f"total: {getattr(u, 'total_tokens', None)}")
    # คำนวณความเร็ว tokens/sec เฉพาะช่วง API call
    if getattr(u, "total_tokens", None) and t_api > 0:
        tps = u.total_tokens / t_api
        print(f"throughput: {tps:.2f} tokens/sec")

# รายงานเวลา
print(f"API latency: {t_api*1000:.1f} ms")
print(f"Total runtime: {(time.perf_counter() - t_total_start):.3f} s")
