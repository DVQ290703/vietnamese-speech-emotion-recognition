import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils.misc import probs_dict

TEXT_REPO   = "qdovan03/phobert-large-vsmec-5class"
MAX_TEXT_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_models = {}


def _get_text_model():
    if "text" not in _models:
        tok   = AutoTokenizer.from_pretrained(TEXT_REPO, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(TEXT_REPO).to(DEVICE).eval()
        _models["text"] = (tok, model)
    return _models["text"]


def classify_text(text: str) -> dict:
    tok, model = _get_text_model()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_TEXT_LEN).to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits
    return probs_dict(logits)
