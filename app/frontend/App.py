"""
Vietnamese Speech Emotion Recognition — Gradio demo
- Audio: Whisper FT (LoRA) -> Fusion (wav2vec2 + PhoBERT) -> cảm xúc
- Text : PhoBERT-large-5class -> cảm xúc
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
from huggingface_hub import login
import gradio as gr

from src.inference.text import classify_text
from src.inference.audio import transcribe_audio, classify_audio
from src.utils.misc import add_history, history, MAX_HISTORY
from src.utils.html import build_result_html, build_error_html, build_history_html

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

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

MAX_AUDIO_S = 8

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
