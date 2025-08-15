import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import time
import mimetypes
from typing import List, Any, Dict, Tuple

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
from PIL import Image  # noqa: F401

import gradio as gr

# -------- Profiles (อ่านจาก .env) --------
# ปลอดภัยกว่าให้ผู้ใช้เลือก "โปรไฟล์" แทนพิมพ์ token ตรง ๆ
PROFILES: Dict[str, Dict[str, Any]] = {
    # โปรไฟล์ตัวอย่าง 1: GitHub Models (main)
    "GitHub/Main": {
        "endpoint": os.getenv("ENDPOINT_GH", "https://models.github.ai/inference"),
        "token": os.getenv("GITHUB_TOKEN"),
        "models": [
            os.getenv("MODEL", "openai/gpt-4.1"),
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "gpt-4.1-mini",      # ถ้ามีสิทธิ์ใช้
            "phi-3.5-mini-instruct",  # ตัวอย่าง non-OpenAI บางตัว
            "meta/Llama-4-Maverick-17B-128E-Instruct-FP8",
            "deepseek/DeepSeek-R1-0528",  # DeepSeek R1
            "xai/grok-3",
        ],
        "default_model": os.getenv("MODEL", "openai/gpt-4.1"),
    },
    # โปรไฟล์ตัวอย่าง 2: GitHub Models (alt)
    "huggingface": {
        "endpoint": os.getenv("ENDPOINT_HF", "https://router.huggingface.co/v1"),
        "token": os.getenv("HF_TOKEN"),
        "models": [
            "openai/gpt-oss-120b:fireworks-ai",
            
        ],
        "default_model": "openai/gpt-oss-120b:fireworks-ai",
    },
    # เพิ่มโปรไฟล์ของคุณเองด้านล่างนี้ได้อีกตามต้องการ
    # "Azure/Prod": {
    #     "endpoint": os.getenv("AZURE_INFERENCE_ENDPOINT"),
    #     "token": os.getenv("AZURE_INFERENCE_TOKEN"),
    #     "models": ["phi-3.5-mini-instruct"],
    #     "default_model": "phi-3.5-mini-instruct",
    # },
}

DEFAULT_PROFILE = next(iter(PROFILES.keys()))  # ชื่อโปรไฟล์แรก

# -------- Client cache ต่อโปรไฟล์ --------
CLIENT_CACHE: Dict[Tuple[str, str], ChatCompletionsClient] = {}

def get_client(endpoint: str, token: str) -> ChatCompletionsClient:
    key = (endpoint, token)
    if key not in CLIENT_CACHE:
        CLIENT_CACHE[key] = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(token),
        )
    return CLIENT_CACHE[key]

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
    return f"[Unsupported file type: {os.path.basename(path)}]"

def image_to_content_item(path: str) -> ImageContentItem:
    _, ext = os.path.splitext(path.lower())
    img_fmt = IMAGE_EXT_MAP.get(ext, "png")  # default png
    img_url = ImageUrl.load(image_file=path, image_format=img_fmt)
    return ImageContentItem(image_url=img_url)

def to_text_item(text: str) -> TextContentItem:
    return TextContentItem(text=text)

def history_to_messages(history: List[Any]) -> List[Any]:
    msgs = []
    if not history:
        return msgs
    trimmed = history[-8:]
    for turn in trimmed:
        if isinstance(turn, (list, tuple)) and len(turn) == 2:
            u, a = turn[0] or "", turn[1] or ""
            if isinstance(u, str) and u.strip():
                msgs.append(UserMessage(content=[to_text_item(u)]))
            elif isinstance(u, dict) and "text" in u:
                text_u = u.get("text") or ""
                msgs.append(UserMessage(content=[to_text_item(text_u)]))
            if isinstance(a, str) and a.strip():
                msgs.append(AssistantMessage(content=[to_text_item(a)]))
        elif isinstance(turn, dict) and "role" in turn and "content" in turn:
            role = turn["role"]; content = turn["content"]
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

# -------- Core chat function --------
def chat_fn(message, history, system_prompt, profile_key, model_name, max_tokens, temperature):
    """
    profile_key: ชื่อโปรไฟล์จาก dropdown (ชี้ไปที่ endpoint + token)
    model_name: โมเดลที่เลือก (จะตั้งค่า default ให้ตามโปรไฟล์)
    """
    global SESSION_PROMPT_TOKENS, SESSION_COMPLETION_TOKENS, SESSION_TOTAL_TOKENS

    # --- หา endpoint/token จากโปรไฟล์ ---
    profile = PROFILES.get(profile_key or "", {})
    endpoint = profile.get("endpoint")
    token = profile.get("token")

    if not endpoint or not token:
        return f"❌ โปรไฟล์ '{profile_key}' ยังไม่ตั้งค่า endpoint/token ใน .env (หรือค่าว่าง)"

    # --- เตรียม client ตามโปรไฟล์ ---
    client = get_client(endpoint, token)

    # --- เก็บ headers จาก response ผ่าน hook ---
    last_headers: Dict[str, str] = {}
    def _hook(resp):
        try:
            h = resp.http_response.headers
            nonlocal last_headers
            last_headers = {k.lower(): v for k, v in h.items()}
        except Exception as e:
            last_headers = {"_hook_error": str(e)}

    # --- เตรียมข้อความระบบ + history ---
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append(SystemMessage(content=system_prompt.strip()))
    messages.extend(history_to_messages(history))

    # --- แตกไฟล์จาก message (image & docs) ---
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

    # --- เรียกโมเดล + วัด API latency ---
    t_api_start = time.perf_counter()
    try:
        resp = client.complete(
            messages=messages,
            model=model_name,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            raw_response_hook=_hook,
        )
    except Exception as e:
        return f"❌ API error ({profile_key}): {e}"

    t_api_ms = (time.perf_counter() - t_api_start) * 1000.0

    # --- ดึงข้อความตอบกลับ ---
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

    # === Rate limit ===
    rate_line = parse_rate_limit_headers(last_headers)

    # === สรุปท้ายข้อความ ===
    report_lines = [f"(api latency: {t_api_ms:.0f} ms) [profile: {profile_key}]"]
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
        "- เลือกโปรไฟล์ (token+endpoint) และโมเดลได้ | แนบรูป/ไฟล์ได้\n"
    )

    sys_tb = gr.Textbox(value="You are a helpful assistant.", label="System Prompt")

    profile_dd = gr.Dropdown(
        choices=list(PROFILES.keys()),
        value=DEFAULT_PROFILE,
        label="Credential Profile",
        interactive=True,
    )

    # ตั้งค่าตามโปรไฟล์เริ่มต้น
    init_models = PROFILES[DEFAULT_PROFILE]["models"]
    init_default = PROFILES[DEFAULT_PROFILE]["default_model"]

    model_dd = gr.Dropdown(
        choices=init_models,
        value=init_default,
        allow_custom_value=True,  # ถ้าต้องการพิมพ์เอง
        label="Model",
        interactive=True,
    )

    max_tokens_sl = gr.Slider(64, 4096, value=512, step=64, label="max_tokens")
    temp_sl = gr.Slider(0.0, 1.0, value=0.2, step=0.1, label="temperature")

    # อัปเดตรายการโมเดลอัตโนมัติเมื่อเปลี่ยนโปรไฟล์
    def on_profile_change(pkey: str):
        p = PROFILES.get(pkey, {})
        models = p.get("models", [])
        default_model = p.get("default_model", models[0] if models else "")
        return gr.update(choices=models, value=default_model)

    profile_dd.change(fn=on_profile_change, inputs=profile_dd, outputs=model_dd)

    chat = gr.ChatInterface(
        fn=chat_fn,
        multimodal=True,
        title="Multimodal ChatBot",
        description="เลือกโปรไฟล์ + โมเดล แล้วแชทได้เลย (แนบรูป/ไฟล์ได้)",
        additional_inputs=[sys_tb, profile_dd, model_dd, max_tokens_sl, temp_sl],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_api=False)
