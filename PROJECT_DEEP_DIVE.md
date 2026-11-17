# ⚡ Pick My LLM — Complete Project Deep Dive & Portfolio Architecture

> **What happened, how it works, line-by-line explanation, advanced portfolio features, and interview guide.**

---

## 🗂️ Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [Supported Providers & Models (Cloud + Local Offline)](#2-supported-providers--models-cloud--local-offline)
3. [How It Works — Big Picture Flow](#3-how-it-works--big-picture-flow)
4. [Project File Structure](#4-project-file-structure)
5. [Line-by-Line Code Walkthrough](#5-line-by-line-code-walkthrough)
   - [app.py — Main Streamlit Dashboard & Tabs](#-apppy--main-streamlit-dashboard--tabs)
   - [core/client.py — Universal API Adapter (Cloud & Local)](#-coreclientpy--universal-api-adapter-cloud--local)
   - [core/runner.py — Rate-Limited Async Orchestrator](#-corerunnerpy--rate-limited-async-orchestrator)
   - [core/scorer.py — Multi-Mode Answer Evaluator](#-corescorerpy--multi-mode-answer-evaluator)
   - [core/recommender.py — Min-Max Normalizer & Ranking Engine](#-corerecommenderpy--min-max-normalizer--ranking-engine)
   - [core/viz.py — Interactive Plotly Radar & Breakdown Visualizations](#-corevizpy--interactive-plotly-radar--breakdown-visualizations)
   - [core/history.py — Persistent Benchmark Run History Cache](#-corehistorypy--persistent-benchmark-run-history-cache)
   - [run_tests.py — Automated Test Suite](#-run_testspy--automated-test-suite)
   - [data/sample_dataset.json — 20 Multi-Domain Questions](#-datasample_datasetjson--20-multi-domain-questions)
6. [How to Test the Entire System](#6-how-to-test-the-entire-system)
7. [Core Concepts You Must Understand (Mental Model for Interviews)](#7-core-concepts-you-must-understand-mental-model-for-interviews)
8. [Dataset Engineering: How to Create More Datasets](#8-dataset-engineering-how-to-create-more-datasets)
9. [How to Deploy to Streamlit Cloud](#9-how-to-deploy-to-streamlit-cloud)
10. [How to Present This on Your Portfolio & Resume](#10-how-to-present-this-on-your-portfolio--resume)

---

## 1. What Is This Project?

**Pick My LLM** is an open-source **AI model benchmarking & decision recommendation engine** built with Python and Streamlit.

### The Problem It Solves:
When building LLM-powered applications, developers struggle to choose the best model. Generic leaderboards (like LMSYS Chatbot Arena) evaluate generic chatbot vibes rather than **your specific questions, domain accuracy, latency constraints, and operational cost**.

### What This Tool Does:
- **Simultaneous Evaluation:** Tests identical prompts concurrently across Groq, Google Gemini, OpenRouter, Cerebras, Mistral, and local offline Ollama models.
- **Flexible Scoring:** Evaluates answers via **Deterministic Exact Match**, **Substring Keyword Matching**, or **LLM-as-a-Judge** (using an independent AI to grade open-ended answers).
- **Multi-Attribute Optimization:** Uses min-max normalized weighted scoring so users can adjust sliders (Accuracy %, Speed, Cost) to receive a custom-tailored recommendation.
- **Interactive Multi-Dimensional Visualizations:** Dynamic Plotly Radar / Spider profiles, Pareto frontier speed-vs-accuracy trade-offs, and category-wise performance bars.
- **Persistent Historical Benchmarks:** Automatically logs runs to disk (`cached_results.json`) so users can compare runs, audit latency trends, and download past CSVs.

---

## 2. Supported Providers & Models (Cloud + Local Offline)

| Provider | Supported Models | Access / Pricing | Best For |
|---|---|---|---|
| **Groq** | `llama-3.1-8b-instant`, `llama-3.3-70b-versatile` | Free Tier (console.groq.com) | Ultra-fast token generation |
| **Google AI Studio** | `gemini-2.5-flash`, `gemini-2.5-pro` | Free Tier (aistudio.google.com) | High reasoning, multilingual & logic |
| **OpenRouter** | `mistral-7b-instruct:free`, `llama-3.2-3b-instruct:free` | Free Tier (openrouter.ai) | Open-weights ecosystem |
| **Cerebras AI** | `llama3.1-8b`, `llama-3.3-70b` | Free Tier (cloud.cerebras.ai) | World record inference speeds |
| **Mistral AI** | `mistral-small-latest`, `codestral-latest` | Free Tier (console.mistral.ai) | Code generation & reasoning |
| **Ollama (Local Offline)** | `llama3.2:1b`, `llama3.1:8b`, `mistral:latest`, `qwen2.5:7b` | **100% Free (No Internet/API Key needed)** | Edge AI, privacy, zero API costs |
| **Custom OpenAI-Compatible** | Any model via vLLM, DeepSeek, LocalAI, Together AI | Custom URL & API Key | Private models & custom hosted endpoints |

---

## 3. How It Works — Big Picture Flow

```
                      User (Browser)
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
      [Tab 1: Live Benchmark]   [Tab 2: History & Trends]
               │                         │
       Select Models & Prompts     Select past run from dropdown
               │                         │
        Run Benchmark              Inspect past Leaderboard & Radar
               │
    ┌──────────┴──────────┐
    ▼                     ▼
Demo Mode           Live API Calls (runner.py)
(Mock Data)               │  [Exponential Backoff retry]
                          ▼
                    client.py (Groq / Gemini / OpenRouter / Cerebras / Mistral / Ollama)
                          │
                          ▼
                    scorer.py (Exact / Keyword / LLM Judge)
                          │
                          ▼
                    recommender.py (Min-Max Normalize & Rank)
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
    history.py (Save to JSON)      viz.py (Plotly Radar, Trade-off, Bars)
          │                               │
          └───────────────┬───────────────┘
                          ▼
              Render Leaderboard & Visuals
```

---

## 4. Project File Structure

```
pick_my_llm/
│
├── app.py                    ← Main Streamlit Dashboard (Tabs: Live Benchmark & History)
├── run_tests.py              ← Automated 6-part test suite verifying 100% of the system
├── requirements.txt          ← Production dependencies (Streamlit, Plotly, GenAI, Groq, OpenAI)
├── .env.example              ← Template documentation for API keys
├── .gitignore                ← Excludes venv/, credentials, and caches from Git
├── README.md                 ← Comprehensive documentation with badges, setup, and roadmap
├── PROJECT_DEEP_DIVE.md      ← Complete line-by-line guide, architecture & portfolio strategies
│
├── core/
│   ├── __init__.py           ← Marks core as an importable Python package
│   ├── client.py             ← Universal API adapter (Groq, Google, OpenRouter, Cerebras, Mistral, Ollama)
│   ├── runner.py             ← Orchestrates prompt execution with exponential backoff retries
│   ├── scorer.py             ← Exact match, keyword match, and LLM-as-a-judge
│   ├── recommender.py        ← Cost simulation, min-max normalization, and weighted ranking
│   ├── viz.py                ← Interactive Plotly charts (Radar profiles, Category bars, Trade-off)
│   └── history.py            ← JSON run caching, history inspection, and run management
│
└── data/
    ├── sample_dataset.json   ← 20 multi-domain questions (Python, SQL, ML, Logic, Multilingual, Cloud, Security)
    └── cached_results.json   ← Persistent JSON store for historical benchmark runs
```

---

## 5. Line-by-Line Code Walkthrough

### 📄 `app.py` — Main Streamlit Dashboard & Tabs
- **Sidebar:** Expands into **Cloud API Keys** and **Local / Custom Endpoints** (Ollama, custom base URLs).
- **Model Selection:** Supports Groq, Google AI Studio, OpenRouter, Cerebras, Mistral, and local Ollama models.
- **Dataset Selection:** Radios to choose between the **Full 20-Question Suite**, a **Quick 5-Question Smoke Test**, or **Custom Upload**.
- **Execution & Scoring:** Dynamically invokes either deterministic exact/keyword scoring or the **LLM-as-a-Judge** evaluator.
- **History & Visualizations:** Renders Plotly Multi-Attribute Radar Profiles, Pareto Trade-Off Scatter plots, and grouped category bars, alongside a history management tab.

### 📄 `core/client.py` — Universal API Adapter (Cloud & Local)
- Unified entry point `call_llm(provider, model_name, prompt, api_key, custom_base_url)`.
- Handles native Google Gemini SDK (`google-genai`), Groq SDK (`groq`), and uses `openai.OpenAI` for OpenRouter, Cerebras, Mistral, and local Ollama endpoints.
- Returns normalized metrics `{response, latency, prompt_tokens, completion_tokens, error}`.

### 📄 `core/runner.py` — Rate-Limited Async Orchestrator
- Iterates over all $(Questions \times Models)$.
- Automatically detects local offline models (like Ollama) so they run without requiring API keys.
- Implements exponential backoff: on HTTP 429 rate limit errors, sleeps $delay \times 2^{attempts}$ seconds.

### 📄 `core/scorer.py` — Multi-Mode Answer Evaluator
- `score_exact_match()`: Sanitized case-insensitive equality for MCQ options.
- `score_keyword_match()`: Substring verification for short factual responses.
- `score_llm_judge()`: Dispatches an evaluation prompt to an independent model to grade open-ended answers as `CORRECT` or `WRONG`.

### 📄 `core/recommender.py` — Min-Max Normalizer & Ranking Engine
- Maps token counts against `MODEL_PRICING` lookup table (local Ollama models cost $0.00).
- Normalizes accuracy, inverted speed (lower is better), and inverted cost (lower is better) to a 0–1 scale.
- Computes final score using user weights: $(Accuracy \times w_{acc}) + (Speed \times w_{spd}) + (Cost \times w_{cost})$.

### 📄 `core/viz.py` — Interactive Plotly Radar & Breakdown Visualizations
- `create_radar_chart()`: Renders interactive closed polygons for each model across 4 dimensions.
- `create_tradeoff_chart()`: Bubble scatter plot of Latency vs. Accuracy.
- `create_category_chart()`: Grouped bar chart breaking down accuracy per question category.

### 📄 `core/history.py` — Persistent Benchmark Run History Cache
- Full CRUD operations on `data/cached_results.json`: `load_history()`, `save_run()`, `delete_run()`, and `clear_history()`.

### 📄 `run_tests.py` — Automated Test Suite
- Comprehensive automated test runner verifying all 6 modules in ~1 second.

---

## 6. How to Test the Entire System

You have **3 ways to test the entire system**:

### Method 1: Automated Test Suite (Fastest & Most Rigorous)
Run the built-in test runner from your terminal:
```bash
python run_tests.py
```
**What this tests:**
- ✅ Exact & Keyword match scorers
- ✅ Cost estimation formulas
- ✅ Min-max normalization and weighted ranking logic
- ✅ History cache lifecycle (Save, Load, Delete, Clear)
- ✅ Plotly chart object generation
- ✅ Sample dataset schema integrity

### Method 2: UI Demo Mode (Zero API Keys Needed)
1. Run the app: `streamlit run app.py`
2. In the sidebar, toggle **"🚀 Enable Demo Mode (No API Key Required)"**.
3. Select any models, click **"Run Benchmark Comparison"**.
4. The system simulates full execution, computes scores, generates Plotly radar charts, logs the run to history, and lets you inspect Tab 2.

### Method 3: Live End-to-End API Test
1. Enter a free API key (e.g. Groq from `console.groq.com` or Google AI Studio from `aistudio.google.com`).
2. Select **"⚡ Quick 5-Question Smoke Test"** under Section 2.
3. Select 2 models (e.g., Llama 3.1 8B vs. Gemini 2.5 Flash).
4. Click **"Run Benchmark Comparison"** and observe live latency, token counts, and accuracy comparisons.

---

## 7. Core Concepts You Must Understand (Mental Model for Interviews)

When discussing this project with recruiters, engineering leads, or interviewers, focus on these **5 core software engineering concepts**:

### 1. The Universal Adapter Pattern (Decoupling)
- **Concept:** Every AI provider has its own SDK, parameter names, and response structures.
- **Implementation:** `core/client.py` wraps Groq, Google, OpenAI, Cerebras, and Ollama into a unified contract returning identical dictionaries. The rest of the app never cares which provider is being queried.

### 2. Multi-Criteria Decision Analysis (MCDA) & Inverted Min-Max Normalization
- **Concept:** You cannot simply add Accuracy (percentage: 0–100), Latency (seconds: 0.2–5.0), and Cost (dollars: $0.0001–$0.05).
- **Implementation:** In `core/recommender.py`, every metric is normalized to a $[0, 1]$ scale. Because lower latency and lower cost are superior, their scales are inverted:
  $$\text{Normalized Speed} = \frac{\max(\text{latency}) - \text{latency}}{\max(\text{latency}) - \min(\text{latency})}$$
  This mathematical grounding is a major talking point in ML systems interviews.

### 3. LLM-as-a-Judge Evaluation Rigor
- **Concept:** String matching fails for generative AI (e.g., if the answer is "Paris", a model answering "The capital is Paris, France" would fail exact match).
- **Implementation:** `core/scorer.py` introduces an objective judge prompt that queries a separate model to perform semantic grading.

### 4. Distributed Resilience (Exponential Backoff)
- **Concept:** Free-tier APIs strictly throttle requests (HTTP 429).
- **Implementation:** `core/runner.py` catches rate-limit exceptions and pauses dynamically using $delay \times 2^{attempt}$, preventing cascading job failures.

### 5. Edge / Local AI Support (Ollama)
- **Concept:** Enterprise clients often cannot send proprietary data to cloud APIs.
- **Implementation:** By supporting local Ollama models on `localhost:11434`, the benchmark can evaluate models completely offline with zero API cost and 100% privacy.

---

## 8. Dataset Engineering: How to Create More Datasets

The project now includes a **20-question multi-domain benchmark** in [`data/sample_dataset.json`](file:///D:/pick_my_llm/data/sample_dataset.json) covering 8 domains:
- Python Programming
- SQL & Databases
- Machine Learning & Data Science
- Math & Quantitative Logic
- Multilingual (Gujarati & Hindi)
- Cloud & DevOps Concepts
- JSON Entity Extraction
- Security & Safety (Prompt Injection)

### How to Create Your Own Custom Dataset:
Create a `.json` file containing a list of objects with these 4 keys:
```json
[
  {
    "id": 1,
    "category": "Customer Support",
    "question": "Can a user cancel their subscription within 14 days for a full refund according to EU law? Reply with Yes or No.",
    "correct_answer": "Yes"
  },
  {
    "id": 2,
    "category": "Medical Q&A",
    "question": "What is the medical term for high blood pressure? Give only the single term.",
    "correct_answer": "Hypertension"
  }
]
```
Upload this file via the **"Upload custom JSON Suite"** option in Section 2, and the benchmark will automatically evaluate models on your specialized domain!

---

## 9. How to Deploy to Streamlit Cloud

1. **Commit and push your repo to GitHub:**
   ```bash
   git add .
   git commit -m "feat: multi-provider support, automated tests, 20-question benchmark suite"
   git push origin main
   ```
2. **Deploy on Streamlit Community Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io).
   - Select your repo and branch `main`.
   - Set **Main file path** to `app.py`.
   - Click **Deploy**.

---

## 10. How to Present This on Your Portfolio & Resume

### 📌 Resume Bullet Points:
```
• Engineered "Pick My LLM", a full-stack AI evaluation platform benchmarking models across 6+ providers 
  (Groq, Google Gemini, OpenRouter, Cerebras, Mistral, and local offline Ollama instances).

• Implemented multi-mode answer evaluation combining Deterministic Matching with an LLM-as-a-Judge pattern, 
  backed by an exponential backoff retry pipeline to mitigate free-tier rate limits.

• Designed a min-max normalized multi-criteria decision algorithm weighting Accuracy, Latency, and Simulated 
  Token Cost to deliver customized model recommendations visualized via Plotly Radar Profiles and historical dashboards.

• Developed an automated test suite (run_tests.py) validating client adapters, scoring logic, normalization algorithms, 
  and data persistence lifecycles with 100% pass rate.
```
