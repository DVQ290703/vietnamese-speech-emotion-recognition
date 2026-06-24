import torch
from datetime import datetime
from collections import deque

NUM_LABELS = 5
MAX_HISTORY = 50

ID2LABEL = {
    0: "Enjoyment 😊",
    1: "Sadness 😢",
    2: "Anger 😠",
    3: "Surprise 😲",
    4: "Other 😐",
}

history: deque = deque(maxlen=MAX_HISTORY)


def probs_dict(logits) -> dict:
    p = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
    return {ID2LABEL[i]: float(p[i]) for i in range(NUM_LABELS)}


def add_history(source: str, text: str, label_scores: dict, top_label: str):
    history.appendleft({
        "time":       datetime.now().strftime("%H:%M:%S"),
        "source":     source,
        "text":       text,
        "emotion":    top_label,
        "confidence": f"{label_scores.get(top_label, 0) * 100:.1f}%",
    })
