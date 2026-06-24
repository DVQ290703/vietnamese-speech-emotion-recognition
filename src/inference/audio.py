import torch
import torchaudio
from transformers import (AutoTokenizer, Wav2Vec2FeatureExtractor,
                          WhisperProcessor, WhisperForConditionalGeneration)
from huggingface_hub import hf_hub_download
from peft import PeftModel

from src.models.ser import MultimodalSER
from src.utils.misc import probs_dict

WHISPER_REPO = "qdovan03/whisper-small-vi-lora-asr"
WHISPER_BASE = "openai/whisper-small"
FUSION_REPO  = "qdovan03/vietspeech-fusion-ser"
AUDIO_MODEL  = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
TEXT_BASE    = "vinai/phobert-base"
SR           = 16000
MAX_AUDIO_S  = 8
MAX_TEXT_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_models = {}


def _get_whisper():
    if "whisper" not in _models:
        proc   = WhisperProcessor.from_pretrained(WHISPER_BASE, use_fast=False)
        base   = WhisperForConditionalGeneration.from_pretrained(WHISPER_BASE)
        model  = PeftModel.from_pretrained(base, WHISPER_REPO).to(DEVICE).eval()
        forced = proc.get_decoder_prompt_ids(language="vi", task="transcribe")
        _models["whisper"] = (proc, model, forced)
    return _models["whisper"]


def _get_fusion():
    if "fusion" not in _models:
        model = MultimodalSER().to(DEVICE)
        path  = hf_hub_download(FUSION_REPO, "best_fusion.pt")
        state = torch.load(path, map_location="cpu", mmap=True)
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval()
        fe  = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_MODEL)
        tok = AutoTokenizer.from_pretrained(TEXT_BASE, use_fast=False)
        _models["fusion"] = (model, fe, tok)
    return _models["fusion"]


def transcribe_audio(audio_path: str):
    wav, sr = torchaudio.load(audio_path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    wav = wav.mean(0)[: MAX_AUDIO_S * SR]
    proc, wmodel, forced = _get_whisper()
    feats = proc(wav.numpy(), sampling_rate=SR, return_tensors="pt").input_features.to(DEVICE)
    with torch.no_grad():
        ids = wmodel.generate(feats, forced_decoder_ids=forced)
    transcript = proc.batch_decode(ids, skip_special_tokens=True)[0]
    return transcript, wav


def classify_audio(wav, transcript: str) -> dict:
    fmodel, fe, ftok = _get_fusion()
    audio_in = fe(wav.numpy(), sampling_rate=SR, return_tensors="pt", padding=True).to(DEVICE)
    text_in  = ftok(transcript, return_tensors="pt", truncation=True,
                    max_length=MAX_TEXT_LEN, padding=True).to(DEVICE)
    with torch.no_grad():
        logits = fmodel(dict(audio_in), dict(text_in))
    return probs_dict(logits)
