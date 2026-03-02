import torch
import torch.nn as nn
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
from sklearn.metrics.pairwise import cosine_similarity

embedding_index = joblib.load("ticket_embedding_index.pkl")
stored_embeddings = embedding_index["embeddings"]
stored_texts = embedding_index["texts"]
stored_labels = embedding_index["labels"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "Nethra19/multitask-ticket-model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_queue_labels, num_priority_labels):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.queue_classifier = nn.Linear(hidden_size, num_queue_labels)
        self.priority_classifier = nn.Linear(hidden_size, num_priority_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.last_hidden_state[:, 0]

        queue_logits = self.queue_classifier(pooled_output)
        priority_logits = self.priority_classifier(pooled_output)

        return queue_logits, priority_logits


# -------------------------
# Load from HuggingFace
# -------------------------

heads_path = hf_hub_download(MODEL_NAME, "heads.pt")
queue_encoder_path = hf_hub_download(MODEL_NAME, "queue_encoder.pkl")
priority_encoder_path = hf_hub_download(MODEL_NAME, "priority_encoder.pkl")

queue_encoder = joblib.load(queue_encoder_path)
priority_encoder = joblib.load(priority_encoder_path)

model = MultiTaskModel(
    MODEL_NAME,
    len(queue_encoder.classes_),
    len(priority_encoder.classes_)
)

heads = torch.load(heads_path, map_location=device)
model.queue_classifier.load_state_dict(heads["queue_classifier"])
model.priority_classifier.load_state_dict(heads["priority_classifier"])

model.to(device)
model.eval()


# ======================================================
# 🔥 FINAL PREDICTION FUNCTION WITH PROPER SALIENCY
# ======================================================

def predict_ticket(subject: str, body: str):

    text = f"{subject} {body}"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # -------------------------
    # Forward pass
    # -------------------------

    queue_logits, priority_logits = model(**inputs)

    temperature = 1.0
    queue_probs = torch.softmax(queue_logits / temperature, dim=1)
    priority_probs = torch.softmax(priority_logits / temperature, dim=1)

    # Top 2 department classes
    top2 = torch.topk(queue_probs, 2)
    top_class = top2.indices[0][0].item()
    second_class = top2.indices[0][1].item()

    confidence_gap = (
        queue_probs[0][top_class] - queue_probs[0][second_class]
    ).item()

    queue_conf = queue_probs[0][top_class].item()
    priority_class = torch.argmax(priority_probs).item()
    priority_conf = priority_probs[0][priority_class].item()

    # ==================================================
    # 🔥 PROPER GRADIENT SALIENCY
    # ==================================================

    model.zero_grad()

    # Get input embeddings
    embeddings = model.encoder.embeddings.word_embeddings(
        inputs["input_ids"]
    )
    embeddings.retain_grad()

    outputs = model.encoder(
        inputs_embeds=embeddings,
        attention_mask=inputs["attention_mask"]
    )

    pooled_output = outputs.last_hidden_state[:, 0]
    queue_logits = model.queue_classifier(pooled_output)

    model.zero_grad()
    target_logit = queue_logits[0, top_class]
    target_logit.backward(retain_graph=True)
    top_grads = embeddings.grad.clone()

    # Competing class gradient
    model.zero_grad()
    target_competing = queue_logits[0, second_class]
    target_competing.backward()
    competing_grads = embeddings.grad.clone()[0]
    competing_token_importance = torch.norm(
        competing_grads, dim=1
    ).detach().cpu().numpy()

    competing_token_importance = competing_token_importance / (
        competing_token_importance.max() + 1e-8
    )

    grads = top_grads[0] # (seq_len, hidden_dim)
    token_importance = torch.norm(grads, dim=1).detach().cpu().numpy()

    # Normalize token importance
    token_importance = token_importance / (token_importance.max() + 1e-8)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Reconstruct full words from subwords
    words = []
    word_scores = []

    current_word = ""
    current_score = 0
    count = 0

    for token, score in zip(tokens, token_importance):

        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        if token.startswith("##"):
            current_word += token[2:]
            current_score += score
            count += 1
        else:
            if current_word:
                words.append(current_word)
                word_scores.append(current_score / max(count, 1))
            current_word = token
            current_score = score
            count = 1

    if current_word:
        words.append(current_word)
        word_scores.append(current_score / max(count, 1))

    word_scores = np.array(word_scores)
    word_scores = word_scores / (word_scores.max() + 1e-8)

    comp_word_scores = []

    current_word = ""
    current_score = 0
    count = 0

    for token, score in zip(tokens, competing_token_importance):

        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        if token.startswith("##"):
            current_word += token[2:]
            current_score += score
            count += 1
        else:
            if current_word:
                comp_word_scores.append(current_score / max(count, 1))
            current_word = token
            current_score = score
            count = 1

    if current_word:
        comp_word_scores.append(current_score / max(count, 1))

    comp_word_scores = np.array(comp_word_scores)
    comp_word_scores = comp_word_scores / (
        comp_word_scores.max() + 1e-8
    )

    competing_words = sorted(
        list(zip(words, comp_word_scores)),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    # Top 8 influential words
    top_k = 8
    top_indices = np.argsort(word_scores)[-top_k:]

    highlighted_text = ""
    important_words = []

    for i, (word, score) in enumerate(zip(words, word_scores)):

        if i in top_indices and word.isalpha():

            opacity = 0.25 + 0.65 * score

            highlighted_text += (
                f"<span style='"
                f"background: rgba(139,92,246,{opacity});"
                f"color:white;"
                f"padding:3px 6px;"
                f"border-radius:8px;"
                f"font-weight:600'>"
                f"{word}</span> "
            )

            important_words.append((word, round(float(score), 3)))

        else:
            highlighted_text += word + " "

    # Sort important words by strength
    important_words = sorted(
        important_words,
        key=lambda x: x[1],
        reverse=True
    )

    # Confidence explanation
    if queue_conf > 0.85:
        queue_reason = "High confidence — strong class-specific indicators."
    elif queue_conf > 0.6:
        queue_reason = "Moderate confidence — some overlap with other departments."
    else:
        queue_reason = "Low confidence — ambiguous ticket."

    uncertain = confidence_gap < 0.20

    with torch.no_grad():
        embed_output = model.encoder(**inputs)
        query_embedding = embed_output.last_hidden_state[:, 0, :].cpu().numpy()

    similarities = cosine_similarity(query_embedding, stored_embeddings)[0]

    top_sim_indices = similarities.argsort()[-3:][::-1]

    similar_examples = []

    for idx in top_sim_indices:
        similar_examples.append({
            "text": stored_texts[idx][:200] + "...",
            "label": stored_labels[idx],
            "similarity": round(float(similarities[idx]), 3)
        })

    return {
        "department": queue_encoder.inverse_transform([top_class])[0],
        "department_confidence": round(queue_conf, 3),
        "priority": priority_encoder.inverse_transform([priority_class])[0],
        "priority_confidence": round(priority_conf, 3),
        "highlighted_text": highlighted_text,
        "important_words": important_words,
        "queue_reason": queue_reason,
        "competing_department":
            queue_encoder.inverse_transform([second_class])[0],
        "confidence_gap": round(confidence_gap, 3),
        "uncertain": uncertain,
        "queue_distribution": {
            queue_encoder.classes_[i]: round(float(p), 3)
            for i, p in enumerate(queue_probs[0])
        },
        "similar_examples": similar_examples,
        "competing_words": competing_words,
    }