import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import time
import mimetypes
from typing import List, Tuple, Any

from dotenv import load_dotenv
load_dotenv()

# === Azure AI Inference (GitHub Models endpoint) ===
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    SystemMessage, UserMessage, AssistantMessage,
    TextContentItem, ImageContentItem, ImageUrl
)
from azure.core.credentials import AzureKeyCredential

# === Minimal file parsers ===
from PyPDF2 import PdfReader
from PIL import Image

import gradio as gr

# -------- Config --------
ENDPOINT = os.getenv("ENDPOINT", "https://models.github.ai/inference")
MODEL = os.getenv("MODEL", "openai/gpt-4.1")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

client = ChatCompletionsClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(GITHUB_TOKEN),
)

# -------- Helpers --------
IMAGE_EXT_MAP = {
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".png": "png",
    ".webp": "webp",
    ".bmp": "bmp",
}

TEXT_LIKE = {".txt", ".md", ".csv", ".log"}
PDF_LIKE = {".pdf"}

def is_image(path: str) -> bool:
    mt, _ = mimetypes.guess_type(path)
    return (mt or "").startswith("image/")

def read_text_from_file(path: str, max_chars: int = 20000) -> str:
    _, ext = os.path.splitext(path.lower())
    if ext in TEXT_LIKE:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()[:max_chars]

    if ext in PDF_LIKE:
        text = []
        try:
            reader = PdfReader(path)
            for page in reader.pages:
                text.append(page.extract_text() or "")
        except Exception as e:
            return f"[PDF read error: {e}]"
        return ("\n".join(text))[:max_chars]

    # ไม่รองรับไฟล์ชนิดอื่นตรงๆ: แนบเป็นชื่อไฟล์เฉยๆ
    return f"[Unsupported file type: {os.path.basename(path)}]"

def image_to_content_item(path: str) -> ImageContentItem:
    _, ext = os.path.splitext(path.lower())
    img_fmt = IMAGE_EXT_MAP.get(ext, "png")  # default png
    # สร้าง data URL ผ่าน SDK โดยอัตโนมัติ
    img_url = ImageUrl.load(image_file=path, image_format=img_fmt)
    return ImageContentItem(image_url=img_url)

def to_text_item(text: str) -> TextContentItem:
    return TextContentItem(text=text)

def history_to_messages(history: List[Any]) -> List[Any]:
    """
    แปลง history จาก gr.ChatInterface ให้เป็นรายการข้อความสำหรับ Azure
    รองรับทั้งรูปแบบ [(user, assistant), ...] และ/หรือ รูปแบบ dict messages
    """
    msgs = []
    if not history:
        return msgs

    # กรองเอา 8 เทิร์นล่าสุดพอ (กัน prompt โตเกิน)
    trimmed = history[-8:]

    for turn in trimmed:
        # รูปแบบ tuple: (user_text, assistant_text)
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            u, a = turn[0] or "", turn[1] or ""
            if isinstance(u, str) and u.strip():
                msgs.append(UserMessage(content=[to_text_item(u)]))
            elif isinstance(u, dict) and "text" in u:
                # ถ้าเผื่อ history เก็บ dict ของ multimodal
                text_u = u.get("text") or ""
                msgs.append(UserMessage(content=[to_text_item(text_u)]))

            if isinstance(a, str) and a.strip():
                msgs.append(AssistantMessage(content=[to_text_item(a)]))

        # รูปแบบ dict ตาม type="messages" (ถ้าใช้ในอนาคต)
        elif isinstance(turn, dict) and "role" in turn and "content" in turn:
            role = turn["role"]
            content = turn["content"]
            # แปลงเป็น TextContent อย่างง่าย
            text_join = ""
            if isinstance(content, list):
                # ดึงเฉพาะข้อความ
                text_join = " ".join([c.get("text", "") for c in content if isinstance(c, dict)])
            elif isinstance(content, str):
                text_join = content

            if role == "user":
                msgs.append(UserMessage(content=[to_text_item(text_join)]))
            elif role == "assistant":
                msgs.append(AssistantMessage(content=[to_text_item(text_join)]))

    return msgs

# -------- Core chat function for Gradio --------
def chat_fn(message, history, system_prompt, model_name, max_tokens, temperature):
    """
    message: dict {"text": str, "files": [tmp_paths]}
    history: previous chat display (list)
    """
    t0 = time.perf_counter()

    # เตรียมข้อความระบบ
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append(SystemMessage(content=system_prompt.strip()))

    # นำ history เดิมเข้าเป็นบริบท (ข้อความล้วน)
    messages.extend(history_to_messages(history))

    # แตกไฟล์จาก message (image & docs)
    user_text = (message or {}).get("text") or ""
    files = (message or {}).get("files") or []

    # แยก image กับ non-image
    image_files = [f for f in files if is_image(f)]
    other_files = [f for f in files if not is_image(f)]

    user_contents = []
    if user_text.strip():
        user_contents.append(to_text_item(user_text.strip()))

    # แนบรูปเข้าคอนเทนต์ (รองรับหลายรูป)
    for path in image_files:
        try:
            user_contents.append(image_to_content_item(path))
        except Exception as e:
            user_contents.append(to_text_item(f"[Image load error: {os.path.basename(path)}: {e}]"))

    # Extract ข้อความจากไฟล์ที่ไม่ใช่รูป (txt/pdf ฯลฯ)
    for path in other_files:
        snippet = read_text_from_file(path, max_chars=12000)
        header = f"\n--- file: {os.path.basename(path)} ---\n"
        user_contents.append(to_text_item(header + snippet))

    # ถ้าไม่มีอะไรเลย ก็ส่งข้อความเปล่าไป (กัน error)
    if not user_contents:
        user_contents = [to_text_item("")]

    messages.append(UserMessage(content=user_contents))

    # เรียกโมเดล
    resp = client.complete(
        messages=messages,
        model=model_name or MODEL,
        max_tokens=int(max_tokens),
        temperature=float(temperature),
    )

    # ดึงข้อความตอบกลับ (รองรับกรณี content เป็น list ของ content items)
    msg = resp.choices[0].message
    content = msg.content
    if isinstance(content, list):
        out = "".join(getattr(c, "text", "") for c in content)
    else:
        out = str(content)

    latency_ms = (time.perf_counter() - t0) * 1000
    # แสดง latency เล็กน้อยท้ายข้อความ (ช่วยดีบัก)
    out += f"\n\n---\n(latency: {latency_ms:.0f} ms)"

    return out

# -------- UI --------
with gr.Blocks(theme="soft") as demo:
    gr.Markdown(
        "## 📎 Multimodal Chatbot (Gradio + Azure AI Inference)\n"
        "- พิมพ์ถาม + แนบรูป/ไฟล์ ได้โดยตรงในช่องอินพุต\n"
        "- โมเดลเริ่มต้น: `openai/gpt-4.1`\n"
    )

    chat = gr.ChatInterface(
        fn=chat_fn,
        multimodal=True,  # ส่ง text+files ได้
        title="Multimodal ChatBot",
        description="แนบรูปหรือไฟล์ในช่องอินพุตได้เลย จากนั้นพิมพ์คำถาม",
        additional_inputs=[
            gr.Textbox(value="You are a helpful assistant.", label="System Prompt"),
            gr.Textbox(value=MODEL, label="Model"),
            gr.Slider(64, 2048, value=512, step=64, label="max_tokens"),
            gr.Slider(0.0, 1.0, value=0.2, step=0.1, label="temperature"),
        ],
        # อย่าใส่ retry_btn / undo_btn / clear_btn ใน v5
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_api=False)