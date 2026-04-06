"""
Stage 2a — Transformer Classification
Fine-tuned DistilBERT multitask model predicts department + priority.

Changes from old version:
- Default model ID points to V6 repo
- max_length reduced to 128 (matches V6 training — median ticket ~60 words)
- clean_text extended to catch more salutation variants from Tobi data
"""
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import joblib
from pipeline.logger import get_logger

log = get_logger("stage2a.transformer")


class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_queue_labels, num_priority_labels):
        super().__init__()
        self.encoder             = AutoModel.from_pretrained(model_name)
        h                        = self.encoder.config.hidden_size
        self.queue_classifier    = nn.Linear(h, num_queue_labels)
        self.priority_classifier = nn.Linear(h, num_priority_labels)

    def forward(self, input_ids, attention_mask):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        return self.queue_classifier(pooled), self.priority_classifier(pooled)


def clean_text(text: str) -> str:
    """Normalise raw ticket text before classification."""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = text.replace("\\n", " ")

    # Salutation stripping — anchored to start so it can't eat content words
    text = re.sub(
        r"^(dear|hello|hi|respected|greetings)[^\n,.:!?]{0,50}[,:\-]\s*",
        "", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"\b(customer\s+support|support\s+team|dear\s+customer\s+support"
        r"|dear\s+support\s+team)[,:\-]?\s*",
        "", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"i\s+hope\s+this\s+(message|email)\s+(finds|reaches)\s+you\s+well[.,]?",
        "", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r"i\s+am\s+(writing|contacting|reaching\s+out)\s+to[^.]*\.",
        "", text, flags=re.IGNORECASE
    )
    return re.sub(r"\s+", " ", text).strip()


def load_transformer(hf_model_id: str, hf_token: str, device: torch.device):
    """
    Load V6 multitask model from HuggingFace.

    Default model: Nethra19/multitask-ticket-model-v6
    Files expected in repo:
        config.json, model.safetensors  — DistilBERT encoder backbone
        heads.pt                        — queue + priority classifier heads
        queue_encoder.pkl               — LabelEncoder for 8 dept classes
        priority_encoder.pkl            — LabelEncoder for 3 priority classes
        tokenizer_config.json, vocab.txt etc — tokenizer
    """
    from huggingface_hub import hf_hub_download

    log.info(f"Loading V6 transformer from {hf_model_id}")

    tokenizer      = AutoTokenizer.from_pretrained(hf_model_id, token=hf_token)
    queue_enc_path = hf_hub_download(hf_model_id, "queue_encoder.pkl",    token=hf_token)
    prio_enc_path  = hf_hub_download(hf_model_id, "priority_encoder.pkl", token=hf_token)
    heads_path     = hf_hub_download(hf_model_id, "heads.pt",             token=hf_token)

    queue_encoder    = joblib.load(queue_enc_path)
    priority_encoder = joblib.load(prio_enc_path)

    model = MultiTaskModel(
        hf_model_id,
        len(queue_encoder.classes_),
        len(priority_encoder.classes_)
    )
    heads = torch.load(heads_path, map_location=device, weights_only=False)
    model.queue_classifier.load_state_dict(heads["queue_classifier"])
    model.priority_classifier.load_state_dict(heads["priority_classifier"])
    model.to(device).eval()

    log.info(f"V6 Transformer ready — {len(queue_encoder.classes_)} queue classes: "
             f"{list(queue_encoder.classes_)}")
    log.info(f"Priority classes : {list(priority_encoder.classes_)}")
    return model, tokenizer, queue_encoder, priority_encoder


def transformer_predict(text: str, model, tokenizer, queue_encoder,
                        priority_encoder, device) -> dict:
    text = clean_text(text)
    log.debug(f"Predicting on ({len(text)} chars): {text[:100]!r}...")

    # max_length=128 matches V6 training (median ticket ~60 words)
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()
              if k != "token_type_ids"}

    with torch.no_grad():
        q_logits, p_logits = model(**inputs)
        q_probs = F.softmax(q_logits, dim=1)[0]
        p_probs = F.softmax(p_logits, dim=1)[0]

    q_id       = q_probs.argmax().item()
    p_id       = p_probs.argmax().item()
    top3_idx   = q_probs.topk(3).indices.tolist()
    top3_probs = [
        {"dept": queue_encoder.classes_[i], "prob": round(q_probs[i].item(), 4)}
        for i in top3_idx
    ]

    result = {
        "dept":           queue_encoder.classes_[q_id],
        "dept_conf":      round(q_probs[q_id].item(), 4),
        "top3_dept":      top3_probs,
        "priority":       priority_encoder.classes_[p_id],
        "priority_conf":  round(p_probs[p_id].item(), 4),
        "priority_probs": {
            priority_encoder.classes_[i]: round(p_probs[i].item(), 4)
            for i in range(len(priority_encoder.classes_))
        },
    }

    log.info(f"Stage 2a — dept={result['dept']} ({result['dept_conf']*100:.1f}%)  "
             f"priority={result['priority']} ({result['priority_conf']*100:.1f}%)")
    log.debug(f"Top-3: {[(x['dept'], x['prob']) for x in top3_probs]}")
    return result
