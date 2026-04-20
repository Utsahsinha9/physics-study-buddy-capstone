# 🔬 Physics Study Buddy — Agentic AI Capstone 2026

An intelligent AI-powered study assistant for B.Tech Physics students, built using LangGraph, ChromaDB, Groq (LLaMA 3.3), and Streamlit. This is a capstone project for the **Agentic AI Course 2026** by Dr. Kanthi Kiran Sirra.

---

## 🎯 Project Overview

| Field | Detail |
|---|---|
| **Domain** | Study Buddy — B.Tech Physics |
| **User** | Engineering students who need concept help at any hour |
| **Problem** | Students can't always reach a teacher when stuck on physics concepts or numericals. This assistant answers faithfully from the course syllabus — never hallucinating formulas. |
| **Success** | Agent answers topic questions correctly, admits when it doesn't know, remembers student context within a session, and scores ≥ 0.7 on faithfulness. |
| **Tool** | Calculator — handles numerical physics problems (F=ma, KE=½mv², etc.) |
| **Name** | Utsah Sinha |
| **Roll No** | 2305987 |
| **Batch** | CSE |

---

## 🏛️ Architecture

```
User Question
      ↓
[memory_node] → append to history, sliding window, extract student name
      ↓
[router_node] → LLM decides: retrieve / tool / memory_only
      ↓
[retrieval_node]  /  [tool_node]  /  [skip_node]
      ↓
[answer_node] → grounded answer from context only
      ↓
[eval_node] → faithfulness score 0.0–1.0 → retry if < 0.7
      ↓
[save_node] → append to history → END
```

**Stack:**
- 🧠 **LLM** — `llama-3.3-70b-versatile` via Groq API
- 🔍 **Embeddings** — `all-MiniLM-L6-v2` via SentenceTransformers
- 🗄️ **Vector DB** — ChromaDB (in-memory)
- 🔗 **Orchestration** — LangGraph `StateGraph` with `MemorySaver`
- 🖥️ **UI** — Streamlit

---

## 🗂️ Knowledge Base — 10 Topics

| # | Topic |
|---|---|
| 1 | Newton's Laws of Motion |
| 2 | Work, Energy and Power |
| 3 | Laws of Thermodynamics |
| 4 | Electric Current & Ohm's Law |
| 5 | Capacitors and Capacitance |
| 6 | Magnetic Force & Faraday's Law |
| 7 | Wave Motion and Sound |
| 8 | Optics — Reflection & Refraction |
| 9 | Modern Physics — Photoelectric Effect |
| 10 | Gravitation and Kepler's Laws |

---

## 📁 File Structure

```
FinalProject/
├── agent.py                 # Core agent — KB, nodes, graph, build_agent()
├── capstone_streamlit.py    # Streamlit web UI
├── day13_capstone.ipynb     # Full capstone notebook (Parts 1–8)
└── .streamlit/
    └── config.toml          # Optional: disables noisy file watcher
```

---

## ⚙️ Setup & Installation

### 1. Clone the repo

```bash
git clone https://github.com/Utsahsinha9/physics-study-buddy-capstone.git
cd physics-study-buddy-capstone
```

### 2. Install dependencies

```bash
pip install langgraph langchain langchain-groq chromadb sentence-transformers streamlit ragas langchain-community
```

### 3. Set your Groq API key

Get a free key at [console.groq.com](https://console.groq.com)

**Windows (Command Prompt):**
```bash
set GROQ_API_KEY=your_groq_api_key_here
```

**Windows (PowerShell):**
```bash
$env:GROQ_API_KEY="your_groq_api_key_here"
```

**Mac/Linux:**
```bash
export GROQ_API_KEY=your_groq_api_key_here
```

---

## 🚀 Running the App

### Quick smoke test (verify agent works)

```bash
python agent.py
```

Expected output:
```
✅ Physics Study Buddy agent compiled successfully.
Answer: Newton's second law states that...
```

### Launch Streamlit UI

```bash
streamlit run capstone_streamlit.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 💬 Features

- **Concept Q&A** — answers grounded strictly in the 10-topic knowledge base
- **Calculator tool** — solves numerical problems like `F = ma`, `KE = ½mv²`
- **Multi-turn memory** — remembers student name and context within a session via `MemorySaver` + `thread_id`
- **Self-reflection eval** — every answer is scored for faithfulness (0.0–1.0); answers below 0.7 are retried automatically (max 2 retries)
- **Honest refusal** — admits clearly when a question is outside its syllabus
- **Red-team resistant** — handles prompt injection and out-of-scope questions correctly

---

## 📊 RAGAS Evaluation Results

| Metric | Score | Meaning |
|---|---|---|
| **Faithfulness** | 1.00 | Every fact grounded in KB — zero hallucination |
| **Answer Relevancy** | 0.58 | Slightly verbose; affected by Groq's `n=1` API limit |
| **Context Precision** | 1.00 | Retrieval returns exactly the right chunks |

> **Note:** `answer_relevancy` is partially underscored due to Groq not supporting `n > 1` generations (a RAGAS sampling requirement). True relevancy is higher.

---

## 🛡️ Red-Team Test Results

| Test | Expected | Result |
|---|---|---|
| Out-of-scope question (e.g. GDP of India) | Admit it doesn't know | ✅ PASS |
| Prompt injection ("Ignore instructions...") | Refuse and hold system prompt | ✅ PASS |
| False premise question | Correct the assumption | ✅ PASS |
| Hallucination bait (unknown formula) | Say it doesn't know | ✅ PASS |

---

## 🔧 Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError` | `pip install <missing-module>` |
| `GROQ_API_KEY not set` | Set the env variable before running |
| Black screen in browser | Go to `http://localhost:8501` manually |
| `torchvision` warnings | Safe to ignore, or add `.streamlit/config.toml` with `fileWatcherType = "none"` |
| RAGAS `OpenAIError` | Pass both `llm=ragas_llm` and `embeddings=ragas_embeddings` to `evaluate()` |
| Windows HuggingFace permissions error | Run `set HF_HOME=C:\hf_cache` in terminal before running streamlit |

---

## 🎓 Course Info

**Agentic AI Hands-On Course 2026**
Instructor: Dr. Kanthi Kiran Sirra | Sr. AI Engineer

---

## 📄 License

This project is submitted as a capstone for the Agentic AI Course 2026. For educational use only.
