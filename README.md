# Customer Support Ticket Router

An intelligent ticket routing system that classifies incoming customer support tickets into departments and priority levels using a multi-stage pipeline combining a fine-tuned transformer, hybrid RAG retrieval, and an LLM. Built with Gradio and deployed as an interactive web application with full explainability tooling.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline Stages](#pipeline-stages)
- [Confidence-Gated Routing](#confidence-gated-routing)
- [Design Decisions and Justifications](#design-decisions-and-justifications)
- [Explainability UI](#explainability-ui)
- [Data and Models](#data-and-models)
- [Setup and Running](#setup-and-running)
- [File Structure](#file-structure)

---

## Overview

Given a raw customer support ticket, the system outputs:

- **Department** : one of 8 routing queues (Billing and Payments, Technical & IT Support, Returns and Exchanges, etc.)
- **Priority** : high / medium / low
- **Confidence** : high / medium / low
- **Reasoning** : a human-readable justification for the priority decision

The system is designed around a core principle: **use the cheapest component that is accurate enough, and only escalate to more expensive components when genuinely needed**. This is implemented as a confidence gate that sits between the transformer and the RAG + LLM stages.

---

## Architecture

### Full System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OFFLINE (one-time setup)                     │
│                                                                     │
│   Raw Ticket Dataset          Dept Definition Docs   Priority Docs  │
│         │                            │                    │         │
│         ▼                            ▼                    ▼         │
│   Fine-tune DistilBERT        all-MiniLM-L6-v2 encodes both        │
│   (dept head + priority head)         │                    │         │
│         │                     FAISS HNSW Index    FAISS Flat Index  │
│         │                     + BM25 Index        (3 chunks)        │
│         ▼                            │                    │         │
│   HuggingFace Model Repo      HuggingFace Model Repo (Rarry/...)   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        ONLINE (per ticket)                          │
│                                                                     │
│  Raw Ticket                                                         │
│      │                                                              │
│      ▼                                                              │
│  Stage 1: Groq LLM (Llama-3.3-70b)                                │
│  Rewrites ticket → clean structured text                           │
│      │                                                              │
│      ▼                                                              │
│  Stage 2a: Fine-tuned DistilBERT                                   │
│  dept_conf, priority_conf, top-3 dept probabilities               │
│      │                                                              │
│      ▼                                                              │
│  ┌───────────────────────────────────────────┐                     │
│  │           CONFIDENCE GATE                 │                     │
│  │  dept ≥ 85% AND priority ≥ 60%?          │                     │
│  └───────────────┬───────────────────────────┘                     │
│                  │                                                  │
│         YES ◄────┴────► NO                                         │
│          │                │                                         │
│          ▼                ▼                                         │
│     Fast Path        Slow Path (per-signal gating)                 │
│     Transformer      ┌────────────────────────────┐               │
│     direct           │ dept < 85%? → FAISS+BM25+  │               │
│                      │ RRF+CrossEncoder retrieval  │               │
│                      │                             │               │
│                      │ priority < 60%? → retrieve  │               │
│                      │ priority chunk + LLM applies│               │
│                      │ escalation criteria          │               │
│                      └────────────────────────────┘               │
│                                                                     │
│  RoutingResult: dept · priority · confidence · reasoning           │
└─────────────────────────────────────────────────────────────────────┘
```

### Routing Decision Matrix

| Dept confident (≥85%) | Priority confident (≥60%) | Path taken |
|:---:|:---:|---|
| ✓ | ✓ | ⚡ Fast path : transformer only, no RAG, no LLM |
| ✓ | ✗ | 🎯 Priority only : retrieve priority chunk → LLM decides priority |
| ✗ | ✓ | 🔀 Dept only : FAISS+BM25+CrossEncoder decides dept, transformer priority used directly |
| ✗ | ✗ | 🔀 Full slow path : RAG for dept, RAG+LLM for priority |

---

## Pipeline Stages

### Stage 1 : Query Rewriting (`pipeline/stage1_rewriter.py`)

**Model:** Groq Llama-3.3-70b-versatile  
**Input:** Raw customer ticket text  
**Output:** Cleaned, structured ticket text

The raw ticket is sent to the LLM which rewrites it into a structured, professional format. It also extracts urgency signals, technical keywords, and a subject line. This normalisation step ensures the downstream transformer and retriever operate on consistent, noise-free input regardless of how the customer originally phrased their message.

Robust JSON extraction with four fallback levels handles malformed LLM output without crashing.

---

### Stage 2a : Transformer Classification (`pipeline/stage2a_transformer.py`)

**Model:** Fine-tuned DistilBERT (`Nethra19/multitask-ticket-model`)  
**Input:** Cleaned ticket text  
**Output:** `dept_conf`, `priority_conf`, top-3 department probabilities, full priority probability distribution

A custom `MultiTaskModel` wraps DistilBERT with two separate linear classification heads : one for department (8 classes) and one for priority (3 classes). Both heads share the same encoder backbone and run in a single forward pass.

The `[CLS]` token representation is pooled and fed into each head independently. The model was fine-tuned on a labelled dataset of real customer support tickets, giving it strong pattern-matching ability for common ticket types.

```python
class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_queue_labels, num_priority_labels):
        self.encoder             = AutoModel.from_pretrained(model_name)
        self.queue_classifier    = nn.Linear(hidden_size, num_queue_labels)
        self.priority_classifier = nn.Linear(hidden_size, num_priority_labels)
```

**Why two separate heads?** Department and priority are orthogonal groupings. A high-priority billing ticket and a high-priority technical ticket should be neighbours in priority space but far apart in department space. A single shared output head cannot optimise for both simultaneously.

---

### Stage 2b : Hybrid RAG Retrieval (`pipeline/stage2b_retriever.py`)

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (dim=384)  
**Retrieval:** FAISS HNSW + BM25 → RRF fusion → CrossEncoder reranking  
**Input:** Cleaned ticket text  
**Output:** Top-4 dept definition chunks, top-1 priority criteria chunk

Only invoked when at least one confidence threshold is not met. Two completely independent retrieval paths:

**Dept definition retrieval** (only when dept < 85%):
1. `all-MiniLM-L6-v2` encodes the ticket → FAISS HNSW search (top-10 dense candidates)
2. BM25 keyword search (top-10 lexical candidates)
3. Reciprocal Rank Fusion merges both ranked lists
4. CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reranks top-20 candidates
5. Top-4 chunks returned

**Priority criteria retrieval** (only when priority < 60%):
1. Same embedding → FAISS flat exact search over 3 priority chunks (HIGH/MEDIUM/LOW)
2. CrossEncoder reranks the 3 candidates
3. Best match returned

**Why `all-MiniLM-L6-v2` instead of the DistilBERT encoder?**

The DistilBERT encoder was fine-tuned on ticket data, giving it a vector space shaped around ticket language. Dept definition docs and priority escalation criteria are written in policy language : a structurally different domain. Using the ticket encoder to embed policy docs creates a domain mismatch that degrades retrieval quality.

`all-MiniLM-L6-v2` was trained specifically for semantic similarity and retrieval tasks, making it domain-agnostic and far better suited to cross-domain matching between a ticket query and a policy document.

**Why separate FAISS indexes for dept and priority?**

If they shared one index, a priority chunk might rank higher than a dept chunk for a given ticket (e.g. an urgent billing ticket retrieves the HIGH priority criteria chunk instead of the billing department definition). Separating the indexes guarantees the dept retrieval always returns dept definition chunks and the priority retrieval always returns escalation criteria : they cannot crowd each other out.

**Why BM25 alongside FAISS?**

Semantic search (FAISS) handles paraphrase and vocabulary mismatch : a ticket saying "the app keeps crashing" can retrieve a chunk about "software failures" even without exact keyword overlap. BM25 handles precise terminology : department-specific jargon like "SLA breach" or "chargeback" will always score highly regardless of whether the embedding captures the full semantics. RRF fusion gets the best of both. CrossEncoder reranking then applies a more expensive but more accurate relevance model on the fused candidate set.

---

### Stage 3 : LLM Priority Resolution (`pipeline/stage3_generator.py`)

**Model:** Groq Llama-3.3-70b-versatile  
**Input:** Cleaned ticket, transformer priority probabilities, priority criteria chunk  
**Output:** Priority level, confidence, reasoning

Only invoked when priority confidence is below 60%. The LLM's scope is intentionally narrow : it only decides priority, not department.

**Department resolution** is deterministic: the CrossEncoder top-ranked dept definition chunk's department label is used directly. No LLM reasoning needed because the CrossEncoder already applied a relevance model to find the most applicable department definition.

**Priority resolution** requires genuine reasoning: the LLM reads the ticket and the retrieved escalation criteria chunk (HIGH/MEDIUM/LOW section) and decides whether those criteria apply to this specific ticket. This is a criteria-application task : reading a rule and deciding if a situation satisfies it : which is a fundamentally different kind of task than classification or retrieval, and where LLMs excel.

**Why not read the priority chunk's section label directly?**

The CrossEncoder ranks chunks by retrieval relevance, not criteria applicability. A ticket about a slow server might retrieve the HIGH criteria chunk because it mentions servers, even if slow performance over a long period with no business impact is correctly MEDIUM. Reading the section label without reasoning would produce systematic errors in ambiguous cases. The LLM applies judgment; the CrossEncoder provides the relevant context.

**Why priority threshold is 60% not higher:**

The priority head is a 3-class classifier. Empirically, even clearly urgent tickets max out at 0.60-0.65 confidence because the model distributes probability mass across fewer classes and priority boundaries are softer than department boundaries. Setting the threshold at 0.60 triggers the LLM for genuinely uncertain cases (below the model's natural confidence ceiling) while allowing clearly routine or clearly urgent tickets to skip it.

---

## Confidence-Gated Routing

The gate is implemented in `pipeline/router.py` with two independent thresholds:

```python
DEPT_CONF_THRESHOLD     = 0.85   # 8-class head : high threshold justified
PRIORITY_CONF_THRESHOLD = 0.60   # 3-class head : lower ceiling, lower threshold
```

Each signal is evaluated independently. RAG is only invoked for the signal that failed its threshold : a ticket with dept=97% and priority=45% skips FAISS entirely and only fetches the priority criteria chunk for the LLM.

**Why gated RAG rather than always-on RAG?**

The transformer is fast, cheap, and accurate on clear-cut tickets. RAG (FAISS + BM25 + CrossEncoder) and an LLM call are expensive. Running them unconditionally wastes compute on tickets where the transformer is already 97% confident. More importantly, forcing an LLM to arbitrate between a confident transformer prediction and a RAG signal that agrees with it adds latency and non-determinism without improving accuracy.

By gating on confidence, RAG is deployed precisely where it adds value : ambiguous tickets where the transformer's pattern-matching is insufficient and grounding in policy definitions or escalation criteria is needed.

---

## Design Decisions and Justifications

### Why keep the transformer at all if we have RAG?

The transformer and RAG serve different functions. The transformer's classification head was trained on labelled ticket data : it learned the statistical patterns in how customers describe billing issues, technical problems, returns, etc. The RAG retriever finds policy chunks that are semantically related to the ticket. These are different signals: one is trained on outcomes, the other is grounded in policy text.

On the fast path (85%+ confident), the transformer is more reliable than RAG because retrieval quality depends on the embedding model's ability to match ticket language to policy language : an imperfect process. On the slow path (uncertain), the policy grounding from RAG compensates for the transformer's weakness on edge cases.


`all-MiniLM-L6-v2` was trained on a large corpus of sentence pairs for semantic similarity, making it domain-agnostic and far better suited to cross-domain retrieval. The DistilBERT encoder is now used only for classification.

### Why is department routing deterministic (CrossEncoder) but priority requires an LLM?

Department routing from dept definition docs is a matching problem : find which department's definition best describes this ticket. The CrossEncoder already applies a relevance model and the top-ranked result is reliably the right answer. No reasoning required.

Priority assignment from escalation criteria is an application problem : read the criteria and decide if they apply to this specific situation. "Revenue is actively being lost" needs to be matched against a specific ticket's context to decide if it qualifies as HIGH. That is a reasoning task where an LLM's language understanding is the right tool.
### Why 0.60 threshold with Stage 3 for priority is the right design
The priority head is a 3-class classifier (high/medium/low). Unlike the department head which has 8 classes and can achieve very high confidence on clear-cut tickets, a 3-class head naturally distributes probability mass across fewer options. Empirically, the priority head peaks at 0.60-0.65 even on tickets with strong urgency signals : not because the model is wrong, but because priority is genuinely a softer boundary than department. The line between "medium" and "high" is contextual and criteria-driven in a way that "Billing" vs "Technical Support" is not.
Setting the threshold at 0.60 means RAG + Stage 3 activates precisely when the transformer is operating below its natural confidence ceiling : i.e. when it is genuinely uncertain rather than just probabilistically hedged. This is a meaningful gate, not an arbitrary one.
### Why Stage 3 (LLM) is still justified at this threshold rather than reading the priority chunk label directly
The priority escalation criteria document contains rules like "route as HIGH when revenue is actively being lost" or "route as MEDIUM when a single user is affected". These are natural language criteria that require applying judgment to a specific ticket. The CrossEncoder retrieves the most relevant priority section (HIGH/MEDIUM/LOW), but retrieval relevance is not the same as criteria applicability : a ticket about a duplicate charge might retrieve the HIGH section because of keyword overlap with financial loss language, even if the actual charge amount is small and the situation is not urgent.
The LLM's role is to read the ticket and ask: do these criteria actually apply here? That is a reasoning task : reading a rule and deciding if a specific situation satisfies it : which is exactly what LLMs are trained to do well. It is not a classification task (transformer's strength) or a retrieval task (CrossEncoder's strength). The three components are each doing the job they are best suited for
### Why two separate priority and dept FAISS indexes?

Priority chunks (HIGH/MEDIUM/LOW sections of the escalation criteria doc) and dept definition chunks serve completely different retrieval purposes. Mixing them in one index would cause them to compete : for an urgent billing ticket, the HIGH priority criteria chunk might outscore the billing dept definition chunk, leaving the LLM with 3 priority chunks and 1 dept chunk instead of the intended 4 dept chunks and 1 priority chunk. Separating them guarantees each retrieval slot is filled by the right type of document.

---

## Explainability UI

The Gradio interface (`app2.py`) provides five tabs of explainability alongside the routing result:

### Evidence
Shows transformer confidence bars for top-3 department predictions and full priority probability distribution. When RAG is invoked, shows the retrieved dept definition chunks with CrossEncoder scores and the retrieved priority criteria chunk. Shows a RAG vote breakdown aggregating CrossEncoder scores by department.

### Explanation
A pipeline trace table showing every stage that ran and what it decided. Clearly marks which stages were skipped (dept retrieval skipped because transformer was confident, LLM skipped because priority was confident, etc.) and which were invoked and why. Per-signal confidence values with threshold pass/fail indicators.

### Attribution
Token-level attribution heatmaps using Integrated Gradients (requires `captum`). Blue tokens support the prediction, red tokens push against it. Separate heatmaps for department attribution and priority attribution. An occlusion check table shows how much confidence drops when each top-support word is masked out : confirming which words are actually load-bearing for the prediction.

### Radar
Two polar charts. The department radar shows transformer top-3 department probability distribution. When RAG runs, overlays a second trace showing the CrossEncoder score distribution across the same departments, allowing visual comparison of where transformer and RAG agree or diverge. The priority radar shows the transformer's probability distribution across high/medium/low.

### Sensitivity
Shows the predicted department vs its nearest competitor (2nd-highest transformer score), the score margin between them, the top words identified by IG attribution as supporting the current prediction, and (when RAG was invoked) keywords extracted from the competitor department's retrieved chunks : terms that would push the routing toward the alternative department.

---

## Data and Models

| Component | Model / Source |
|---|---|
| Transformer backbone | `Nethra19/multitask-ticket-model` (fine-tuned DistilBERT) |
| Retrieval embedder | `sentence-transformers/all-MiniLM-L6-v2` (dim=384) |
| CrossEncoder reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM (rewrite + priority) | Groq `llama-3.3-70b-versatile` |
| FAISS indexes | `Rarry/Improved_RAG` (HuggingFace) |

**Departments (8 classes):**
Billing and Payments · Customer Service · General Inquiry · Human Resources · Technical & IT Support · Returns and Exchanges · Sales and Pre-Sales · Service Outages and Maintenance

**Priorities (3 classes):** high · medium · low

**Dept definition docs** (`compliance_docs/`):
One `.txt` file per department describing routing scope, representative examples, and decision guidance. Used to build the FAISS dept index.

**Priority escalation criteria** (`compliance_docs/Priority_Escalation_Criteria.txt`):
Universal criteria applying across all departments. Split into HIGH, MEDIUM, and LOW sections. Each section becomes one chunk in the priority FAISS index.

---

## Setup and Running

### Prerequisites

```bash
pip install -r requirements.txt
```

### Environment variables (`secrets.env`)

```
HF_TOKEN=hf_...
GROQ_API_KEY=gsk_...
```

### Build FAISS indices

Run `notebooks/03_rebuild_faiss_index.ipynb` top to bottom. This:
1. Loads `all-MiniLM-L6-v2`
2. Chunks all dept definition docs and the priority doc
3. Builds and saves FAISS indexes locally
4. Uploads to `Rarry/Improved_RAG` on HuggingFace

### Run the application

```bash
python app2.py
```

---

## File Structure

```
customer-support-management/
│
├── app2.py                          # Gradio UI and application entry point
│
├── pipeline/
│   ├── __init__.py
│   ├── logger.py                    # Structured logging
│   ├── router.py                    # Orchestration, confidence gate, RoutingResult
│   ├── stage1_rewriter.py           # LLM query rewriting (Groq)
│   ├── stage2a_transformer.py       # DistilBERT multitask classification
│   ├── stage2b_retriever.py         # Hybrid RAG retrieval (FAISS+BM25+CrossEncoder)
│   ├── stage3_generator.py          # LLM priority resolution (Groq)
│   └── ui_helpers.py                # All HTML rendering and explainability functions
│
├── compliance_docs/                 # Dept definition docs + priority escalation criteria
│   ├── Billing_and_Payments.txt
│   ├── Customer_Service.txt
│   ├── General_Inquiry.txt
│   ├── Human_Resources.txt
│   ├── Priority_Escalation_Criteria.txt
│   ├── Returns_and_Exchanges.txt
│   ├── Sales_and_Pre-Sales.txt
│   ├── Service_Outages_and_Maintenance.txt
│   └── Technical & IT Support.txt
│
├── notebooks/
│   ├── 01_eda_and_data_prep.ipynb   # Exploratory analysis and dataset preparation
│   ├── 02_rag_index_builder.ipynb   # Original index builder (DistilBERT encoder)
│   ├── 03_rebuild_faiss_index.ipynb # Rebuild indexes with all-MiniLM-L6-v2
│   └── 03_pipeline_evaluation.ipynb # End-to-end pipeline evaluation
│
├── requirements.txt
└── README.md
```

---

## Key Architectural Invariants

- The DistilBERT encoder is used **classification** only
- The `all-MiniLM-L6-v2` embedder is **never** used for classification : retrieval only
- Dept definition retrieval and priority criteria retrieval use **separate** FAISS indices and separate retrieval pipelines
- The LLM is **only** invoked for priority reasoning, never for department routing
- Department routing on the slow path is **deterministic** : CrossEncoder top chunk, no LLM
- RAG components are **individually gated** : a confident dept prediction skips FAISS if only priority is uncertain and dept is confident, and vice versa