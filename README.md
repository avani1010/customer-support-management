# Customer Support Ticket Router

An end-to-end ML pipeline that automatically routes customer support tickets to the correct department and assigns a priority level. Built for COMP41860 — ML System Deployment, University College Dublin 2024–25.

---

## What It Does

Given a raw support ticket, the system:

1. Rewrites it into clean structured text using an LLM
2. Classifies it using a fine-tuned DistilBERT transformer (department + priority)
3. Retrieves supporting evidence from custom department definition documents using hybrid RAG
4. Makes a final routing decision using an LLM that reads all three sources together

**Output:** Department (one of 8), Priority (HIGH / MEDIUM / LOW), and a natural language reasoning explanation.

---

## Results

| Metric | Transformer alone | Full pipeline |
|---|---|---|
| Department accuracy | 58.9% | **81.8%** |
| Priority accuracy | 39.1% | **85.4%** |
| RAG Recall@1 | — | 72.9% |
| RAG Recall@4 | — | 94.8% |
| RAG MRR | — | 0.822 |
| Stage 1 rewrite quality (AI judge) | — | 3.00 / 3 |
| Stage 3 reasoning quality (AI judge) | — | 2.27 / 3 |
| E2E median latency | — | 3.31s |
| TTFT | — | 2.16s |
| TPOT | — | 0.0178s/token |

---

## Pipeline Architecture

```
Raw ticket
    │
    ▼
Stage 1 — LLM rewrite (OpenAI gpt-4o)
    Cleans and structures the raw ticket.
    Anti-injection constraint: never adds department vocabulary
    not present in the original ticket.
    │
    ▼
Stage 2a — DistilBERT V6 classifier  [always runs, no API]
    Fine-tuned multitask model with two heads:
      Head 1: Linear(768 → 8)  — department
      Head 2: Linear(768 → 3)  — priority
    Outputs confidence scores used by the confidence gate.
    │
    ▼
Confidence gate: dept_conf ≥ 0.90?
    ├── YES → dept RAG skipped (fast path)
    └── NO  → Stage 2b runs (slow path)
    │
    ▼
Stage 2b — Hybrid RAG retrieval  [no API]
    Dept retrieval (gated at 90% confidence):
      1. all-MiniLM-L6-v2 embeds query → FAISS HNSW search (top-10)
      2. BM25Okapi keyword search (top-10)
      3. RRF fusion: score = Σ 1/(60 + rank) → top-20 candidates
      4. ms-marco CrossEncoder reranks → top-4 dept chunks returned

    Priority retrieval (always runs, never gated):
      1. all-MiniLM-L6-v2 embeds query → FAISS flat exact search
      2. CrossEncoder scores all 3 priority sections (HIGH/MEDIUM/LOW)
      3. Best section returned
    │
    ▼
Stage 3 — LLM final decision (OpenAI gpt-4o)
    Reads: transformer predictions + dept chunks + priority chunk
    Decides: department + priority + reasoning
    Falls back to transformer prediction on JSON parse failure.
    │
    ▼
RoutingResult
    department, priority, confidence, reasoning,
    transformer_dept, transformer_conf, rag_chunks,
    rag_gap, priority_chunk, dept_rag_skipped
```

---

## Departments

| Department | Description |
|---|---|
| Billing and Payments | Invoices, charges, refunds, subscription issues |
| Technical & IT Support | Software, hardware, network, security incidents |
| Customer Service | Complaints, escalations, relationship management |
| Service Outages and Maintenance | Platform-wide outages, infrastructure disruptions |
| Returns and Exchanges | Product returns, investment return calculations |
| Sales and Pre-Sales | Pricing, demos, upgrades, brand growth |
| Human Resources | Onboarding, payroll, HR portal, employee training |
| General Inquiry | Informational questions, documentation requests |

---

## Project Structure

```
customer-support-management/
│
├── pipeline/
│   ├── router.py              # Orchestrates all stages, owns confidence gate
│   ├── stage1_rewriter.py     # LLM ticket rewriter (OpenAI gpt-4o)
│   ├── stage2a_transformer.py # DistilBERT V6 multitask classifier
│   ├── stage2b_retriever.py   # Hybrid RAG: FAISS + BM25 + RRF + CrossEncoder
│   ├── stage3_generator.py    # LLM routing decision (OpenAI gpt-4o)
│   ├── ui_helpers.py          # All Gradio render functions
│   └── logger.py              # Structured logging
│
├── compliance_docs/           # 9 custom department definition documents
│   ├── Billing_and_Payments.txt
│   ├── Customer_Service.txt
│   ├── General_Inquiry.txt
│   ├── Human_Resources.txt
│   ├── Returns_and_Exchanges.txt
│   ├── Sales_and_Pre-Sales.txt
│   ├── Service_Outages_and_Maintenance.txt
│   ├── Technical & IT Support.txt
│   └── Priority_Escalation_Criteria.txt
│
├── notebooks/
│   ├── 01_eda_and_data_prep.ipynb       # Dataset loading, cleaning, train/test split
│   ├── 02_rag_index_builder.ipynb       # FAISS + BM25 index construction
│   ├── 03_pipeline_evaluation.ipynb     # Early evaluation notebook
│   ├── 03_rebuild_faiss_index.ipynb     # Index rebuild after doc improvements
│   ├── 04_evaluation_pipeline.ipynb     # Final 4-section evaluation pipeline
│   └── 05_encoder_comparison.ipynb      # DistilBERT vs MiniLM encoder comparison
│
├── app2.py                    # Gradio UI entry point
├── requirements.txt
└── .env                       # API keys (not committed)
```

---

## Models

| Component | Model                                    | Hosted at |
|---|------------------------------------------|---|
| Transformer classifier | `Nethra19/multitask-ticket-model-v6`     | HuggingFace |
| RAG artifacts | `Nethra19/rag-index-v6`                  | HuggingFace |
| Query embedder | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace |
| CrossEncoder reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2`   | HuggingFace |
| Stage 1 + Stage 3 LLM | `gpt-4o`                                 | OpenAI API |
| AI-as-a-Judge | `gpt-5.2`                                | OpenAI API |

**Why MiniLM for retrieval instead of DistilBERT:** DistilBERT was fine-tuned on ticket text and learned ticket-specific representations. The compliance docs are written in policy language — a different domain. MiniLM was trained for semantic similarity across domains and produces better retrieval quality for query-to-policy-document matching.

---

## RAG Knowledge Base

The 9 compliance documents were written from scratch using TF-IDF differential analysis on the training set — identifying vocabulary that appears significantly more in each department's real tickets than in all other departments. Each document contains:

- **Overview** — department scope and distinction from similar departments
- **Key Vocabulary** — exact terms extracted from real training tickets
- **Representative Examples** — 8 real tickets sampled from train.csv, balanced across HIGH/MEDIUM/LOW priority
- **Routing Guidance** — when to route here and explicit disambiguation rules

**Chunking strategy:** Each document produces two types of chunks:
- Section-based: each named section becomes one chunk (preserves semantic coherence)
- Sliding window: 400-character windows, 200-character step (fine-grained coverage)

**Two separate FAISS indexes:**
- Dept index: `IndexHNSWFlat` (approximate nearest neighbour, fast for ~80 chunks)
- Priority index: `IndexFlatIP` (exact search — only 3 chunks, brute force is instant)

**BM25 is built only over dept chunks**, not priority. Priority retrieval skips BM25 and RRF entirely — there are only 3 possible answers (HIGH/MEDIUM/LOW) so FAISS returns all 3 candidates and the CrossEncoder picks the best one directly.

---

## Confidence Gate

The router checks the transformer's department confidence against a threshold of 0.90:

- **Fast path (≥ 0.90):** FAISS, BM25, RRF, and CrossEncoder are all skipped for department. Stage 3 LLM still always runs for priority.
- **Slow path (< 0.90):** Full hybrid RAG runs. The RAG gap (CE score difference between rank-1 and rank-2 chunks) is computed. If gap < 0.15, both transformer and RAG are uncertain — the LLM is told explicitly and sets confidence to low.

---

## Evaluation Pipeline

`notebooks/04_evaluation_pipeline.ipynb` runs four sections:

| Section | What | API calls |
|---|---|---|
| 1 — Transformer standalone | Accuracy, Macro F1, ECE, calibration curves, confusion matrix | 0 |
| 2 — RAG retrieval | Recall@1, Recall@4, MRR per department | 0 |
| 3 — Full pipeline | Dept + priority accuracy, LLM override analysis, TTFT, TPOT, E2E latency | 2 per ticket |
| 4 — AI-as-a-Judge | Rewrite quality, reasoning quality, routing quality (gpt-4o-mini) | 3 per ticket |

All sections cache results to CSV. Re-running any cell resumes from where it left off.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/avani1010/customer-support-management.git
cd customer-support-management
pip install -r requirements.txt
```

### 2. Environment variables

Create a `.env` file in the project root:

```
HF_TOKEN=hf_...
OPENAI_API_KEY=sk-...
```

### 3. Run the UI

```bash
python app2.py
```

Opens at `http://localhost:7860`

---

## UI Features

The Gradio interface is designed for incident managers. After submitting a ticket:

- **Routing card** — department, priority, pipeline flow (which stages ran / were skipped), LLM reasoning
- **Evidence tab** — transformer confidence bars, retrieved RAG chunks with CrossEncoder scores, RAG vote breakdown
- **Trace tab** — full pipeline trace table showing every stage decision and confidence value
- **Attribution tab** — Integrated Gradients token heatmap showing which words drove the transformer's prediction (requires `captum`)
- **Radar tab** — transformer vs RAG signal comparison charts
- **Sensitivity tab** — which words support the current prediction, which words would push toward the runner-up department

---

## Dataset

**Source:** `Tobi-Bueck/customer-support-tickets` on HuggingFace — 61,800 total, 28,261 English tickets.

**Issues identified and fixed:**
- Merged `Technical Support`, `Product Support`, and `IT Support` into a single `Technical & IT Support` queue
- Removed tickets where label contradicted content (HR-labelled IT incidents, Returns-labelled investment calculations)
- Stratified 80/20 train/test split on combined `queue + priority` key
- Evaluation fixed at 30 tickets per department to prevent high-volume classes dominating metrics

**Label noise (known limitation):** Minority class labels are semantically inconsistent with ticket content in the original dataset. Returns & Exchanges tickets are investment return calculations; HR tickets are often IT security incidents. The LLM routes logically by department definitions but is penalised against noisy ground-truth labels — the AI-as-a-judge score is a more reliable measure of true routing quality than accuracy alone on this dataset.

---

## References

- Sanh et al. (2019). DistilBERT, a distilled version of BERT. arXiv:1910.01108
- Robertson & Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond
- Cormack et al. (2009). Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods
- Wang et al. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression. NeurIPS
- Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS