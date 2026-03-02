import torch
import numpy as np
import joblib
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import re

MODEL_NAME = "Nethra19/multitask-ticket-model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading dataset...")

ds = load_dataset("Tobi-Bueck/customer-support-tickets")
df = ds["train"].to_pandas()
df = df[df["language"] == "en"].copy()

# Merge classes exactly like training
merge_classes = ["IT Support", "Technical Support", "Product Support"]
df["queue"] = df["queue"].replace(merge_classes, "Technical & IT Support")

df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

def clean_text(text):
    text = str(text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["text"] = df["text"].apply(clean_text)
df = df[df["text"].str.split().str.len() >= 8]
df = df.drop_duplicates(subset=["text"])

texts = df["text"].tolist()
labels = df["queue"].tolist()

print("Loading encoder...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)
encoder.eval()

embeddings = []

print("Generating embeddings...")

with torch.no_grad():
    for text in tqdm(texts):

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        ).to(device)

        outputs = encoder(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        embeddings.append(cls_embedding.cpu().numpy()[0])

embeddings = np.array(embeddings)

joblib.dump({
    "embeddings": embeddings,
    "texts": texts,
    "labels": labels
}, "ticket_embedding_index.pkl")

print("Embedding index saved successfully.")