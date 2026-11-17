# ⚡ Pick My LLM

**A free, open-source benchmarking tool to compare AI models and pick the best one for YOUR use case.**

> Compare Llama, Gemini, Mistral and more — side by side — on accuracy, speed, and cost. No paid APIs required.

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

> _Replace the link above after deploying to [Streamlit Cloud](https://share.streamlit.io)._

---

## 📸 What It Looks Like

| Leaderboard | Charts |
|---|---|
| Model ranked by weighted score | Accuracy & Latency bar charts side by side |

---

## 🔥 Features

| Feature | Details |
|---|---|
| **Multi-Provider** | Benchmarks Groq, Google AI Studio (Gemini), and OpenRouter simultaneously |
| **LLM-as-a-Judge** | An independent AI model evaluates open-ended answers — not just MCQ |
| **Weighted Ranking** | Tune sliders for Accuracy, Speed, Cost — get a personalized recommendation |
| **Demo Mode** | Try the full app instantly with no API key required |
| **Custom Questions** | Upload your own JSON question set |
| **Category Breakdown** | Questions tagged by topic (Python, ML, Logic, etc.) |
| **CSV Export** | Download all raw benchmark results |
| **Exponential Backoff** | Handles free-tier rate limits automatically with retry logic |
| **Token-Cost Estimation** | Simulates real dollar cost from token usage and published pricing |

---

## 🏗️ How It Works

```
User configures API keys + scoring weights
        ↓
Selects models + loads test questions
        ↓
runner.py loops: each question × each model
        ↓
client.py makes the API call → response + latency + tokens
        ↓
scorer.py checks correctness (exact / keyword / LLM-judge)
        ↓
recommender.py normalizes + ranks models by weighted score
        ↓
app.py displays leaderboard, charts, and recommendation
```

---

## 📁 Project Structure

```
pick_my_llm/
│
├── app.py                    ← Streamlit UI (entry point)
├── requirements.txt          ← All Python dependencies
├── .env.example              ← API key reference template
├── .gitignore
│
├── core/
│   ├── client.py             ← Unified API caller (Groq / Google / OpenRouter)
│   ├── runner.py             ← Benchmark loop with retry + backoff
│   ├── scorer.py             ← Exact match, keyword match, LLM-as-a-judge
│   └── recommender.py        ← Min-max normalization + weighted ranking
│
└── data/
    └── sample_dataset.json   ← 5 built-in test questions (Python, GK, ML, Logic, Gujarati)
```

---

## 🛠️ Local Setup

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/pick_my_llm.git
cd pick_my_llm
```

**2. Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app.py
```

**5. Enter your free API keys in the sidebar**

| Provider | Get Free Key |
|---|---|
| Groq | [console.groq.com](https://console.groq.com) |
| Google AI Studio | [aistudio.google.com](https://aistudio.google.com) |
| OpenRouter | [openrouter.ai](https://openrouter.ai) |

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub account → select this repo.
4. Set **Main file path** to `app.py`.
5. Click **Deploy**. Done! 🎉

---

## 📊 Scoring Formula

Models are ranked using **min-max normalized weighted scoring**:

```
final_score = (norm_accuracy × w_acc) + (norm_speed × w_spd) + (norm_cost × w_cost)
```

- **Accuracy** (higher = better): `(acc - min_acc) / (max_acc - min_acc)`
- **Speed** (lower latency = better): `(max_lat - lat) / (max_lat - min_lat)` _(inverted)_
- **Cost** (lower = better): `(max_cost - cost) / (max_cost - min_cost)` _(inverted)_

Default weights: **60% Accuracy / 30% Speed / 10% Cost** — adjustable via sidebar sliders.

---

## 📝 Custom Question Format

Upload a `.json` file with this structure:

```json
[
  {
    "id": 1,
    "category": "Python",
    "question": "What does len([1,2,3]) return? Give only the number.",
    "correct_answer": "3"
  }
]
```

---

## 🗺️ Roadmap

- [x] Multi-provider benchmarking (Groq, Gemini, OpenRouter)
- [x] LLM-as-a-Judge scoring (live!)
- [x] Weighted ranking with user sliders
- [x] Demo mode
- [ ] Persistent run history dashboard
- [ ] Radar/spider chart visualization
- [ ] Category-wise performance breakdown
- [ ] PDF/HTML report export
- [ ] System prompt testing mode
- [ ] Token efficiency scoring metric

---

## 🤝 Contributing

Pull requests welcome! Please open an issue first to discuss major changes.

---

## 📄 License

MIT License — free to use, fork, and build upon.

---

> Built with ❤️ using Python, Streamlit, and free-tier LLM APIs.
