# Customer Support Ticket Routing System

An end-to-end NLP system that automatically classifies customer support tickets into:

-  **Department (Queue)**
-  **Priority Level**

Built using a pretrained multilingual transformer and designed for production deployment.

---

## Project Overview

Customer support teams receive large volumes of tickets. Manual routing is slow, error-prone and requires significant manual effort and cost to company.

This project builds an intelligent routing system that:

- Understands complaints in English language
- Predicts the correct department
- Assigns Priority (urgency) level
- Outputs confidence scores
- Is designed for API deployment

---

## Model Architecture

Model used (in-trial):

DistilBert

Why this model?

- Disentangled attention mechanism
- Strong contextual understanding
- Production-ready transformer backbone

### Multi-Task Learning Setup

Input Text  
↓  
Tokenizer  
↓  
mDeBERTa Encoder  
↓  
Shared Representation ([CLS])  
↓  
Department Head & Priority Head

Loss Function:

L_total = L_department + L_priority

---

## Dataset

Dataset: Hugging Face – Customer Support Tickets

Features used:

- subject
- body
- queue → Department label
- priority → Urgency label

### Preprocessing Steps

- Null handling
- Concatenation of subject + body
- Tokenization (max_length = 128)
- Label encoding
- Train/Test split (80/20)

---

## Installation

```bash
pip install torch transformers datasets scikit-learn fastapi uvicorn
pip install gradio
```
## Run the UI

```bash
python ui/app.py
```
