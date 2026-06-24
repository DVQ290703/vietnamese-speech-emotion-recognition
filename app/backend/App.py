"""
=============================================================
 Vietnamese Speech Emotion Recognition — Gradio demo
 - Audio: Whisper FT (LoRA) -> Fusion (wav2vec2 + PhoBERT) -> cảm xúc
 - Text : PhoBERT-large-5class -> cảm xúc
=============================================================
"""
import os
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torchaudio
import numpy as np
import gradio as gr
from datetime import datetime
from collections import deque
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, login
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import (AutoModel, AutoTokenizer,
                          Wav2Vec2Model, Wav2Vec2FeatureExtractor,
                          AutoModelForSequenceClassification,
                          WhisperProcessor, WhisperForConditionalGeneration)

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

# ====== CẤU HÌNH ======
WHISPER_REPO = "qdovan03/whisper-small-vi-lora-asr"
WHISPER_BASE = "openai/whisper-small"
FUSION_REPO  = "qdovan03/vietspeech-fusion-ser"
TEXT_REPO    = "qdovan03/phobert-large-vsmec-5class"
AUDIO_MODEL  = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
TEXT_BASE    = "vinai/phobert-base"
SR           = 16000
MAX_AUDIO_S  = 8
MAX_TEXT_LEN = 128
MAX_HISTORY  = 50
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
ID2LABEL     = {0:"Enjoyment 😊", 1:"Sadness 😢", 2:"Anger 😠",
                3:"Surprise 😲", 4:"Other 😐"}
NUM_LABELS   = 5
# ======================

EMOTION_COLORS = {
    "Enjoyment 😊": "#4CAF50",
    "Sadness 😢":   "#2196F3",
    "Anger 😠":     "#F44336",
    "Surprise 😲":  "#FF9800",
    "Other 😐":     "#9E9E9E",
}
EMOTION_BG = {
    "Enjoyment 😊": "#f0faf0",
    "Sadness 😢":   "#f0f6ff",
    "Anger 😠":     "#fff5f5",
    "Surprise 😲":  "#fff8f0",
    "Other 😐":     "#f5f5f5",
}

history: deque = deque(maxlen=MAX_HISTORY)

# ---------- Kiến trúc fusion ----------
class MultimodalSER(nn.Module):
    def __init__(self, num_labels=NUM_LABELS, hidden=256):
        super().__init__()
        self.audio = Wav2Vec2Model.from_pretrained(AUDIO_MODEL)
        self.audio = get_peft_model(self.audio, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj","v_proj"]))
        adim = self.audio.config.hidden_size
        self.text = AutoModel.from_pretrained(TEXT_BASE)
        self.text = get_peft_model(self.text, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["query","value"]))
        tdim = self.text.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(adim+tdim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, num_labels))
    def forward(self, audio_in, text_in):
        a = self.audio(**audio_in).last_hidden_state.mean(dim=1)
        t = self.text(**text_in).last_hidden_state[:, 0]
        return self.classifier(torch.cat([a, t], dim=-1))

# ---------- Lazy load ----------
_models = {}

def get_whisper():
    if "whisper" not in _models:
        proc = WhisperProcessor.from_pretrained(WHISPER_BASE, use_fast=False)
        base = WhisperForConditionalGeneration.from_pretrained(WHISPER_BASE)
        model = PeftModel.from_pretrained(base, WHISPER_REPO).to(DEVICE).eval()
        forced = proc.get_decoder_prompt_ids(language="vi", task="transcribe")
        _models["whisper"] = (proc, model, forced)
    return _models["whisper"]

def get_fusion():
    if "fusion" not in _models:
        model = MultimodalSER().to(DEVICE)
        path = hf_hub_download(FUSION_REPO, "best_fusion.pt")
        state = torch.load(path, map_location="cpu", mmap=True)
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state); model.eval()
        fe  = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_MODEL)
        tok = AutoTokenizer.from_pretrained(TEXT_BASE, use_fast=False)
        _models["fusion"] = (model, fe, tok)
    return _models["fusion"]

def get_text():
    if "text" not in _models:
        tok = AutoTokenizer.from_pretrained(TEXT_REPO, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(TEXT_REPO).to(DEVICE).eval()
        _models["text"] = (tok, model)
    return _models["text"]

def probs_dict(logits):
    p = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
    return {ID2LABEL[i]: float(p[i]) for i in range(NUM_LABELS)}

# ---------- Model inference ----------
def classify_text(text: str) -> dict:
    tok, model = get_text()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_TEXT_LEN).to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits
    return probs_dict(logits)

def transcribe_audio(audio_path: str) -> str:
    wav, sr = torchaudio.load(audio_path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    wav = wav.mean(0)[: MAX_AUDIO_S * SR]
    proc, wmodel, forced = get_whisper()
    feats = proc(wav.numpy(), sampling_rate=SR, return_tensors="pt").input_features.to(DEVICE)
    with torch.no_grad():
        ids = wmodel.generate(feats, forced_decoder_ids=forced)
    return proc.batch_decode(ids, skip_special_tokens=True)[0], wav

def classify_audio(wav, transcript: str) -> dict:
    fmodel, fe, ftok = get_fusion()
    audio_in = fe(wav.numpy(), sampling_rate=SR, return_tensors="pt", padding=True).to(DEVICE)
    text_in  = ftok(transcript, return_tensors="pt", truncation=True,
                    max_length=MAX_TEXT_LEN, padding=True).to(DEVICE)
    with torch.no_grad():
        logits = fmodel(dict(audio_in), dict(text_in))
    return probs_dict(logits)

# ---------- HTML builders ----------
def build_result_html(label_scores: dict, top_label: str) -> str:
    bars = ""
    for label, score in label_scores.items():
        color  = EMOTION_COLORS.get(label, "#9E9E9E")
        pct    = score * 100
        weight = "font-weight:700;" if label == top_label else ""
        bars += f"""
        <div style="margin:6px 0;">
          <div style="display:flex;justify-content:space-between;{weight}">
            <span>{label}</span><span>{pct:.1f}%</span>
          </div>
          <div style="background:#e0e0e0;border-radius:4px;height:12px;margin-top:3px;">
            <div style="background:{color};width:{pct:.1f}%;height:12px;border-radius:4px;"></div>
          </div>
        </div>"""
    top_color = EMOTION_COLORS.get(top_label, "#9E9E9E")
    emoji = top_label.split()[-1] if top_label else "🤔"
    name  = top_label.split(" ")[0] if top_label else "?"
    return f"""
    <div style="font-family:sans-serif;padding:16px;border-radius:10px;border:1px solid #e0e0e0;background:#fafafa;">
      <div style="text-align:center;margin-bottom:16px;">
        <div style="font-size:2.5rem;">{emoji}</div>
        <div style="font-size:1.2rem;font-weight:700;color:{top_color};margin-top:4px;">{name}</div>
      </div>
      {bars}
    </div>"""

def build_error_html(msg: str) -> str:
    return f"<p style='color:#e53935;font-family:sans-serif;padding:12px;border:1px solid #ffcdd2;border-radius:8px;background:#fff5f5;'>❌ {msg}</p>"

def build_history_html() -> str:
    if not history:
        return "<p style='color:#888;font-family:sans-serif;text-align:center;padding:32px;'>Chưa có lịch sử phân tích.</p>"
    rows = ""
    for i, entry in enumerate(history):
        color  = EMOTION_COLORS.get(entry["emotion"], "#9E9E9E")
        bg     = EMOTION_BG.get(entry["emotion"], "#f5f5f5")
        icon   = "📝" if entry["source"] == "text" else "🎤"
        preview = entry["text"][:77] + "…" if len(entry["text"]) > 80 else entry["text"]
        rows += f"""
        <tr style="background:{'white' if i % 2 == 0 else '#f9f9f9'};">
          <td style="padding:10px 12px;color:#666;white-space:nowrap;">{entry["time"]}</td>
          <td style="padding:10px 12px;font-size:1.1rem;text-align:center;">{icon}</td>
          <td style="padding:10px 12px;max-width:400px;word-break:break-word;">{preview}</td>
          <td style="padding:10px 12px;">
            <span style="background:{bg};color:{color};border:1px solid {color};border-radius:16px;
                         padding:3px 10px;font-weight:600;white-space:nowrap;font-size:0.9rem;">
              {entry["emotion"]}
            </span>
          </td>
          <td style="padding:10px 12px;color:#555;">{entry["confidence"]}</td>
        </tr>"""
    return f"""
    <div style="font-family:sans-serif;overflow-x:auto;">
      <table style="width:100%;border-collapse:collapse;font-size:0.95rem;">
        <thead>
          <tr style="background:#f0f0f0;text-align:left;">
            <th style="padding:10px 12px;">Thời gian</th>
            <th style="padding:10px 12px;text-align:center;">Nguồn</th>
            <th style="padding:10px 12px;">Văn bản</th>
            <th style="padding:10px 12px;">Cảm xúc</th>
            <th style="padding:10px 12px;">Độ tin cậy</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
      <p style="color:#999;font-size:0.82rem;margin-top:8px;text-align:right;">
        {len(history)} bản ghi (tối đa {MAX_HISTORY})
      </p>
    </div>"""

def add_history(source: str, text: str, label_scores: dict, top_label: str):
    history.appendleft({
        "time":       datetime.now().strftime("%H:%M:%S"),
        "source":     source,
        "text":       text,
        "emotion":    top_label,
        "confidence": f"{label_scores.get(top_label, 0) * 100:.1f}%",
    })

# ---------- Handlers ----------
def analyze_text(text: str):
    text = (text or "").strip()
    if not text:
        return build_error_html("Vui lòng nhập văn bản."), build_history_html()
    try:
        label_scores = classify_text(text)
        top_label    = max(label_scores, key=label_scores.get)
        add_history("text", text, label_scores, top_label)
        return build_result_html(label_scores, top_label), build_history_html()
    except Exception as e:
        return build_error_html(str(e)), build_history_html()

def analyze_voice(audio):
    if audio is None:
        return build_error_html("Vui lòng ghi âm hoặc tải file lên."), "", build_history_html()
    try:
        transcript, wav = transcribe_audio(audio)
        if not transcript:
            return build_error_html("Không nhận ra nội dung giọng nói. Vui lòng thử lại."), "(không nhận ra nội dung)", build_history_html()
        label_scores = classify_audio(wav, transcript)
        top_label    = max(label_scores, key=label_scores.get)
        add_history("voice", transcript, label_scores, top_label)
        return build_result_html(label_scores, top_label), transcript, build_history_html()
    except Exception as e:
        return build_error_html(str(e)), "", build_history_html()

def refresh_history():
    return build_history_html()

def clear_history():
    history.clear()
    return build_history_html()

# ---------- UI ----------
CSS = """
#title { text-align: center; }
footer { display: none !important; }
"""

with gr.Blocks(css=CSS, title="Nhận diện cảm xúc tiếng Việt") as demo:
    gr.Markdown(
        """
        # 🎭 Nhận Diện Cảm Xúc Tiếng Việt
        Phân tích cảm xúc từ **văn bản** hoặc **giọng nói** tiếng Việt.
        """,
        elem_id="title",
    )

    with gr.Tabs():
        with gr.TabItem("📝 Nhập văn bản"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="Nhập văn bản tiếng Việt",
                        placeholder="Ví dụ: Hôm nay tôi rất vui vì được gặp bạn bè...",
                        lines=5,
                    )
                    text_btn = gr.Button("🔍 Phân tích cảm xúc", variant="primary")
                with gr.Column():
                    text_result = gr.HTML(label="Kết quả")
            gr.Examples(
                examples=[
                    ["Hôm nay tôi rất vui vì được gặp lại những người bạn cũ!"],
                    ["Tôi buồn lắm, không biết làm sao nữa..."],
                    ["Sao anh ta lại làm vậy?! Tôi tức giận lắm!"],
                    ["Ôi trời ơi, tôi không ngờ chuyện này lại xảy ra!"],
                    ["Hôm nay thời tiết bình thường, tôi đi làm như mọi ngày."],
                ],
                inputs=text_input,
                label="Ví dụ",
            )

        with gr.TabItem("🎤 Giọng nói"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(
                        label=f"Ghi âm hoặc tải file (tối đa {MAX_AUDIO_S}s)",
                        type="filepath",
                        sources=["microphone", "upload"],
                    )
                    voice_btn = gr.Button("🔍 Phân tích cảm xúc", variant="primary")
                    transcript_box = gr.Textbox(
                        label="📄 Văn bản nhận dạng",
                        interactive=False,
                        placeholder="Nội dung giọng nói sẽ hiển thị ở đây...",
                    )
                with gr.Column():
                    voice_result = gr.HTML(label="Kết quả cảm xúc")
            gr.Markdown(
                "**Hướng dẫn:** Nhấn 🎤 để ghi âm, hoặc tải file âm thanh lên. "
                "Nói tiếng Việt rõ ràng để đạt kết quả tốt nhất."
            )

        with gr.TabItem("📋 Lịch sử"):
            history_html = gr.HTML(value=build_history_html(), label="Lịch sử phân tích")
            with gr.Row():
                refresh_btn = gr.Button("🔄 Làm mới", variant="secondary")
                clear_btn   = gr.Button("🗑️ Xóa lịch sử", variant="stop")
            refresh_btn.click(fn=refresh_history, outputs=history_html)
            clear_btn.click(fn=clear_history,     outputs=history_html)

    gr.Markdown(
        "---\n**Nhãn cảm xúc:** Enjoyment 😊 · Sadness 😢 · Anger 😠 · Surprise 😲 · Other 😐\n\n"
        "⚠️ Nhãn huấn luyện là pseudo-label (emotion2vec) — kết quả mang tính demo."
    )

    text_btn.click(
        fn=analyze_text,
        inputs=text_input,
        outputs=[text_result, history_html],
    )
    voice_btn.click(
        fn=analyze_voice,
        inputs=audio_input,
        outputs=[voice_result, transcript_box, history_html],
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)
