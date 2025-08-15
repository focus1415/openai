import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import time
import mimetypes
from typing import List, Any, Dict

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
from PIL import Image  # noqa: F401  # (เพียงเพื่อให้แน่ใจว่า PIL พร้อมใช้งาน)

import gradio as gr

# -------- Config --------
ENDPOINT = os.getenv("ENDPOINT", "https://models.github.ai/inference")
MODEL = os.getenv("MODEL", "openai/gpt-4.1")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

client = ChatCompletionsClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(GITHUB_TOKEN),
)

# === Token usage (session accumulators) ===
SESSION_PROMPT_TOKENS = 0
SESSION_COMPLETION_TOKENS = 0
SESSION_TOTAL_TOKENS = 0

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

# โมเดลให้เลือกใน dropdown (ปรับตามสิทธิ์ที่คุณมี)
MODEL_CHOICES = [
    "openai/gpt-4.1",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "gpt-4.1-mini",      # ถ้ามีสิทธิ์ใช้
    "phi-3.5-mini-instruct",  # ตัวอย่าง non-OpenAI บางตัว
    "meta/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "deepseek/DeepSeek-R1-0528",  # DeepSeek R1
    "xai/grok-3",
    # เพิ่ม/ลบได้ตามต้องการ
]


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
                text_u = u.get("text") or ""
                msgs.append(UserMessage(content=[to_text_item(text_u)]))

            if isinstance(a, str) and a.strip():
                msgs.append(AssistantMessage(content=[to_text_item(a)]))

        # รูปแบบ dict ตาม type="messages"
        elif isinstance(turn, dict) and "role" in turn and "content" in turn:
            role = turn["role"]
            content = turn["content"]
            text_join = ""
            if isinstance(content, list):
                text_join = " ".join([c.get("text", "") for c in content if isinstance(c, dict)])
            elif isinstance(content, str):
                text_join = content
            if role == "user":
                msgs.append(UserMessage(content=[to_text_item(text_join)]))
            elif role == "assistant":
                msgs.append(AssistantMessage(content=[to_text_item(text_join)]))

    return msgs

def parse_rate_limit_headers(h: Dict[str, str]) -> str:
    """
    รับ headers (lowercase keys) แล้วสรุปผลคงเหลือของ rate-limit ถ้ามี
    รองรับทั้งรูปแบบ x-ratelimit-*-requests/tokens และแบบ generic
    """
    rl_req_rem   = h.get("x-ratelimit-remaining-requests") or h.get("x-ratelimit-remaining-requests-per-minute")
    rl_req_lim   = h.get("x-ratelimit-limit-requests") or h.get("x-ratelimit-limit-requests-per-minute")
    rl_req_reset = h.get("x-ratelimit-reset-requests") or h.get("x-ratelimit-reset-requests-per-minute")

    rl_tok_rem   = h.get("x-ratelimit-remaining-tokens") or h.get("x-ratelimit-remaining")
    rl_tok_lim   = h.get("x-ratelimit-limit-tokens") or h.get("x-ratelimit-limit")
    rl_tok_reset = h.get("x-ratelimit-reset-tokens") or h.get("x-ratelimit-reset")

    retry_after  = h.get("retry-after")

    parts = []
    if rl_req_rem or rl_req_lim:
        parts.append(f"reqs: {rl_req_rem or '?'} / {rl_req_lim or '?'} (reset {rl_req_reset or '?'})")
    if rl_tok_rem or rl_tok_lim:
        parts.append(f"tokens/min: {rl_tok_rem or '?'} / {rl_tok_lim or '?'} (reset {rl_tok_reset or '?'})")
    if retry_after:
        parts.append(f"retry-after: {retry_after}s")

    return "rate-limit — " + (" | ".join(parts) if parts else "headers not provided by endpoint")

# -------- Core chat function for Gradio --------
def chat_fn(message, history, system_prompt, model_name, max_tokens, temperature):
    """
    message: dict {"text": str, "files": [tmp_paths]}
    history: previous chat display (list)
    """
    global SESSION_PROMPT_TOKENS, SESSION_COMPLETION_TOKENS, SESSION_TOTAL_TOKENS

    # --- เก็บ headers จาก response ผ่าน hook ---
    last_headers = {}
    def _hook(resp):
        try:
            h = resp.http_response.headers  # azure.core.rest.HttpResponse.headers
            nonlocal last_headers
            last_headers = {k.lower(): v for k, v in h.items()}
        except Exception as e:
            last_headers = {"_hook_error": str(e)}

    # เตรียมข้อความระบบ + history
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append(SystemMessage(content=system_prompt.strip()))
    messages.extend(history_to_messages(history))

    # แตกไฟล์จาก message (image & docs)
    user_text = (message or {}).get("text") or ""
    files = (message or {}).get("files") or []
    image_files = [f for f in files if is_image(f)]
    other_files = [f for f in files if not is_image(f)]

    user_contents = []
    if user_text.strip():
        user_contents.append(to_text_item(user_text.strip()))

    for path in image_files:
        try:
            user_contents.append(image_to_content_item(path))
        except Exception as e:
            user_contents.append(to_text_item(f"[Image load error: {os.path.basename(path)}: {e}]"))

    for path in other_files:
        snippet = read_text_from_file(path, max_chars=12000)
        header = f"\n--- file: {os.path.basename(path)} ---\n"
        user_contents.append(to_text_item(header + snippet))

    if not user_contents:
        user_contents = [to_text_item("")]

    messages.append(UserMessage(content=user_contents))

    # เรียกโมเดล + วัด API latency
    t_api_start = time.perf_counter()
    try:
        resp = client.complete(
            messages=messages,
            model=model_name or MODEL,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            raw_response_hook=_hook,    # <<<< ดึง headers ที่นี่
        )
    except Exception as e:
        return f"❌ API error: {e}"

    t_api_ms = (time.perf_counter() - t_api_start) * 1000.0

    # ดึงข้อความตอบกลับ
    msg = resp.choices[0].message
    content = msg.content
    if isinstance(content, list):
        out = "".join(getattr(c, "text", "") for c in content)
    else:
        out = str(content)

    # === Token usage ===
    u = getattr(resp, "usage", None)
    per_prompt = per_completion = per_total = None
    tps = None
    if u:
        per_prompt = getattr(u, "prompt_tokens", None)
        per_completion = getattr(u, "completion_tokens", None)
        per_total = getattr(u, "total_tokens", None)

        if isinstance(per_prompt, int):
            SESSION_PROMPT_TOKENS += per_prompt
        if isinstance(per_completion, int):
            SESSION_COMPLETION_TOKENS += per_completion
        if isinstance(per_total, int):
            SESSION_TOTAL_TOKENS += per_total

        if per_total and t_api_ms > 0:
            tps = per_total / (t_api_ms / 1000.0)

    # === Rate limit (จาก headers ถ้ามี) ===
    rate_line = parse_rate_limit_headers(last_headers)

    # === สรุปท้ายข้อความ ===
    report_lines = [f"(api latency: {t_api_ms:.0f} ms)"]
    if per_total is not None:
        report_lines.append(
            f"tokens — prompt:{per_prompt} | completion:{per_completion} | total:{per_total}"
        )
        report_lines.append(
            f"session — prompt:{SESSION_PROMPT_TOKENS} | completion:{SESSION_COMPLETION_TOKENS} | total:{SESSION_TOTAL_TOKENS}"
        )
        if tps is not None:
            report_lines.append(f"throughput: {tps:.2f} tok/s")
    else:
        report_lines.append("tokens — (no usage returned by API)")
    report_lines.append(rate_line)

    out += "\n\n---\n" + " | ".join(report_lines)
    return out

# -------- UI --------
with gr.Blocks(theme="soft") as demo:
    gr.Markdown(
        "## 📎 Multimodal Chatbot (Gradio + Azure AI Inference)\n"
        "- พิมพ์ถาม + แนบรูป/ไฟล์ ได้โดยตรงในช่องอินพุต\n"
        "- โมเดลเริ่มต้น: `openai/gpt-4.1`\n"
    )

    # สร้างคอมโพเนนต์ไว้ก่อน จะได้อ้างอิงซ้ำ/แก้ไขง่าย
    sys_tb = gr.Textbox(value="You are a helpful assistant.", label="System Prompt")
    model_dd = gr.Dropdown(
        choices=MODEL_CHOICES,
        value=MODEL,                 # ค่าตั้งต้นจาก .env
        allow_custom_value=True,     # ให้พิมพ์ชื่อโมเดลอื่นเองได้
        label="Model",
        interactive=True,
    )
    max_tokens_sl = gr.Slider(64, 4096, value=512, step=64, label="max_tokens")
    temp_sl = gr.Slider(0.0, 1.0, value=0.2, step=0.1, label="temperature")

    chat = gr.ChatInterface(
        fn=chat_fn,
        multimodal=True,
        title="Multimodal ChatBot",
        description="แนบรูปหรือไฟล์ในช่องอินพุตได้เลย จากนั้นพิมพ์คำถาม",
        additional_inputs=[sys_tb, model_dd, max_tokens_sl, temp_sl],
    )

if __name__ == "__main__":
    # ปล. ปิดเซิร์ฟเวอร์ด้วย Ctrl+C ในเทอร์มินัล
    demo.launch(server_name="127.0.0.1", server_port=7860, show_api=False)
