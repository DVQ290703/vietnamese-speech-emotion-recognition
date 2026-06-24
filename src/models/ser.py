import torch.nn as nn
from transformers import Wav2Vec2Model, AutoModel
from peft import LoraConfig, get_peft_model

AUDIO_MODEL = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
TEXT_BASE   = "vinai/phobert-base"
NUM_LABELS  = 5


class MultimodalSER(nn.Module):
    def __init__(self, num_labels=NUM_LABELS, hidden=256):
        super().__init__()
        self.audio = Wav2Vec2Model.from_pretrained(AUDIO_MODEL)
        self.audio = get_peft_model(self.audio, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"]))
        adim = self.audio.config.hidden_size

        self.text = AutoModel.from_pretrained(TEXT_BASE)
        self.text = get_peft_model(self.text, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["query", "value"]))
        tdim = self.text.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(adim + tdim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, num_labels))

    def forward(self, audio_in, text_in):
        a = self.audio(**audio_in).last_hidden_state.mean(dim=1)
        t = self.text(**text_in).last_hidden_state[:, 0]
        return self.classifier(__import__("torch").cat([a, t], dim=-1))
