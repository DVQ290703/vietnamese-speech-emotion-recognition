# Vietnamese Speech Emotion Recognition

Nhận diện cảm xúc tiếng Việt từ **giọng nói** và **văn bản**, sử dụng mô hình multimodal kết hợp Whisper ASR + wav2vec2 + PhoBERT.

## Kiến trúc

```
Giọng nói (audio)
  ├── Whisper Small (fine-tuned LoRA)  →  transcript
  └── wav2vec2-base-vietnamese-250h   ─┐
                                        ├── Fusion Model → Cảm xúc
PhoBERT-base (fine-tuned LoRA)        ─┘

Văn bản (text)
  └── PhoBERT-large (fine-tuned)  →  Cảm xúc
```

## Nhãn cảm xúc

| Nhãn | Mô tả |
|---|---|
| Enjoyment 😊 | Vui vẻ |
| Sadness 😢 | Buồn bã |
| Anger 😠 | Tức giận |
| Surprise 😲 | Ngạc nhiên |
| Other 😐 | Trung tính / khác |

## Models trên Hugging Face

| Model | Repo |
|---|---|
| Whisper ASR (LoRA) | `qdovan03/whisper-small-vi-lora-asr` |
| Fusion SER | `qdovan03/vietspeech-fusion-ser` |
| PhoBERT text classifier | `qdovan03/phobert-large-vsmec-5class` |

## Cài đặt

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Cấu hình

Tạo file `.env` ở thư mục gốc:

```
HF_TOKEN=hf_...
```

## Chạy ứng dụng

```bash
python app/backend/App.py
```

Mở trình duyệt tại `http://localhost:8000`

## Notebooks

| Notebook | Nội dung |
|---|---|
| `01-finetune-whisper-ipynb.ipynb` | Fine-tune Whisper Small với LoRA |
| `02-train-phobert.ipynb` | Huấn luyện PhoBERT phân loại cảm xúc |
| `03-fusion-model.ipynb` | Huấn luyện Fusion Model (audio + text) |

## Dataset

- ASR: [NhutP/VietSpeech](https://huggingface.co/datasets/NhutP/VietSpeech)
- SER: VSMEC (pseudo-label bằng emotion2vec)

> ⚠️ Nhãn huấn luyện SER là pseudo-label — kết quả mang tính demo.
