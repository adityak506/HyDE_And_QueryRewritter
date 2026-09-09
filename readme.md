he goal of this mini-project is to help learners build a **modern, scalable, LLM-agnostic Retrieval-Augmented Generation (RAG) pipeline** that supports:

✅ **OpenAI & Gemini models (switchable via config)**
✅ **Provider-specific FAISS vectorstore (auto-created when missing)**
✅ **Baseline RAG retrieval**
✅ **Query Rewriting (Pre-Retrieval Optimization)**
✅ **HyDE – Hypothetical Document Embeddings**
✅ **Strict context-based answering (no hallucination)**
✅ **Reproducible, real-world project structure**

This lab serves as a foundational building block for later advanced topics like Hybrid Search, Reranking, Multi-Source Retrieval, Context Distillation, and Agentic RAG.

---

# 🧠 **What You Will Learn**

### ✅ Multi-LLM RAG Architecture

Switch seamlessly between providers:

* **OpenAI (GPT-4o-mini, GPT-4.1 family)**
* **Gemini (2.0 / 2.5 Flash, Pro, etc.)**

### ✅ Provider-Aware Vector DB Design

Each provider has its own vectorDB due to different embedding sizes:

* OpenAI → 1536-dim embeddings
* Gemini → 768-dim embeddings

To avoid FAISS dimension mismatch errors, we maintain:

```
faiss_openai/
faiss_gemini/
```

### ✅ Query Rewriting

Improve query clarity, specificity, and retrievability using:

* LLM rewriting heuristics
* Intent expansion
* Vocabulary alignment

### ✅ HyDE (Hypothetical Document Embeddings)

Generate synthetic “ideal answers” and embed them to retrieve more relevant chunks.

### ✅ Modular RAG Pipeline

Reusable utilities:

* loader.py → load & chunk documents
* retriever.py → provider-aware embedding & vectorDB
* query_rewriter.py → rewrite user queries
* hyde_generator.py → synthetic answer generation
* app.py → final RAG comparison (baseline, rewrite, HyDE)

---

# 📂 **Project Structure**

```
project/
│
├── app.py                             # Main RAG pipeline (Baseline + Rewrite + HyDE)
├── config.yaml                         # LLM provider, embedding models, FAISS paths
│
├── data/
│   └── raw/
│       └── insurance_docs/             # Your PDFs / TXT documents
│
│   └── embeddings/
│       ├── faiss_openai/               # FAISS index for OpenAI embeddings
│       └── faiss_gemini/               # FAISS index for Gemini embeddings
│
├── utils/
│   ├── loader.py                       # Document loading & chunking
│   ├── retriever.py                    # Provider-aware FAISS retriever
│   ├── query_rewriter.py               # Query rewriting LLM utility
│   └── hyde_generator.py               # HyDE synthetic answer + embeddings
│
├── requirements.txt                    # All dependencies (LangChain ≥0.3.x compatible)
└── README.md                           # This file
```

---

# ⚙️ **Setup Instructions**

### ✅ 1. Create a Virtual Environment

```bash
python -m venv myenv
source myenv/bin/activate       # macOS / Linux
myenv\Scripts\activate          # Windows
```

### ✅ 2. Install Requirements

(These versions are **stable for LangChain ≥0.3.x**)

```bash
pip install -r requirements.txt
```

### ✅ 3. Create a `.env` File

Add your keys:

```
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

### ✅ 4. Configure Provider in `config.yaml`

```yaml
llm:
  provider: "openai"     # or "gemini"
```

### ✅ 5. Put Documents Inside:

```
/data/raw/insurance_docs/
```

Supported formats: `.txt` and `.pdf`

### ✅ 6. Run the Application

```bash
python app.py
```

Enter a query such as:

```
cashless hospitalization?
deductible?
policy waiting periods?
```

---

# 🚀 **Pipeline Overview**

When you run the app:

## ✅ 1. Baseline RAG

* Embeds user query
* Searches provider-specific FAISS DB
* LLM answers strictly using retrieved context

Good for clear queries.

---

## ✅ 2. Query Rewriting RAG

* Improves vague/short queries
* Adds specificity & domain language
* Retrieves better chunks
* LLM answers using rewritten query

Excellent for:

* Users with poor phrasing
* Domain-misaligned questions
* Missing keywords

---

## ✅ 3. HyDE RAG

* LLM generates synthetic “ideal answer”
* Embedding of that synthetic answer → used for retrieval
* Often retrieves more relevant content

HyDE is powerful for:

✅ Vague questions
✅ One-word queries
✅ Queries not phrased like documents
✅ Poor quality user input

---

# 🏆 **Why This Project Matters**

This mini-project teaches **production-grade RAG design principles**, including:

### ✅ Provider isolation

Avoiding FAISS dimensional conflicts with separate vectorDBs.

### ✅ Pre-retrieval intelligence

Using LLM rewriting + HyDE for higher recall.

### ✅ Modular system architecture

Real-world structure used in enterprise GenAI apps.

### ✅ Strict context enforcement

Avoiding hallucinations through grounded answers.

### ✅ Multi-LLM orchestration

A core skill for any generative AI engineer.

---

# ✅ **Future Labs (Part of Advanced RAG Series)**

This project lays the foundation for upcoming labs:

### ✅ Hybrid Search (BM25 + FAISS)

### ✅ Re-ranking using cross-encoders

### ✅ Multi-source Retrieval (PDF + web + database)

### ✅ Context Distillation

### ✅ Hallucination & Citation enforcement

### ✅ Latency + Cost Optimization

### ✅ Capstone: Advanced Research Assistant

Each new module will extend this project structure.

---

# 🧑‍🏫 **Who Is This For?**

This project is ideal for:

* Developers new to RAG
* Engineers building production GenAI systems
* Learners transitioning from "prompting" to "orchestration"
* Anyone needing a clean, modern RAG template

---

# ✅ **Final Notes**

This repo is intentionally kept:

* **Simple**
* **Provider-agnostic**
* **Notebook-friendly**
* **Reproducible**
* **Production-aligned**

The goal is to teach **the right mental models** before scaling up into LangChain Agents, Hybrid Retrieval, LangGraph, and agentic RAG.

---