import os
import torch
import joblib
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get path to this file (important for Git safety)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

# -------------------------
# Load LOCAL tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(MODEL_DIR, "tokenizer")
)

# -------------------------
# Load MODELS from HuggingFace
# -------------------------
QUEUE_MODEL_NAME = "Nethra19/queue-ticket-model"
PRIORITY_MODEL_NAME = "Nethra19/priority-ticket-model"

queue_model = AutoModelForSequenceClassification.from_pretrained(
    QUEUE_MODEL_NAME
).to(device)

priority_model = AutoModelForSequenceClassification.from_pretrained(
    PRIORITY_MODEL_NAME
).to(device)

# -------------------------
# Load LOCAL encoders
# -------------------------
queue_encoder = joblib.load(
    os.path.join(MODEL_DIR, "queue_encoder.pkl")
)

priority_encoder = joblib.load(
    os.path.join(MODEL_DIR, "priority_encoder.pkl")
)

queue_model.eval()
priority_model.eval()


def predict_ticket(subject: str, body: str) -> dict:
    text = f"{subject} {body}"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # Queue prediction
        queue_outputs = queue_model(**inputs)
        queue_probs = torch.softmax(queue_outputs.logits, dim=1)
        queue_pred_id = torch.argmax(queue_probs, dim=1).item()

        # Priority prediction
        priority_outputs = priority_model(**inputs)
        priority_probs = torch.softmax(priority_outputs.logits, dim=1)
        priority_pred_id = torch.argmax(priority_probs, dim=1).item()

    return {
        "department": queue_encoder.inverse_transform([queue_pred_id])[0],
        "department_confidence": round(queue_probs.max().item(), 3),
        "priority": priority_encoder.inverse_transform([priority_pred_id])[0],
        "priority_confidence": round(priority_probs.max().item(), 3),
    }