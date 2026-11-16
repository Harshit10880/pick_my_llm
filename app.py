import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt

from core.runner import run_benchmark
from core.scorer import score_exact_match, score_keyword_match, score_llm_judge
from core.recommender import aggregate_results, rank_models, generate_recommendation_summary

# Page Configuration
st.set_page_config(
    page_title="Pick My LLM - AI Benchmarking Tool",
    page_icon="⚡",
    layout="wide"
)

# Custom Styling & Dark Mode Vibe
st.markdown("""
    <style>
    .main-title { font-size: 2.5rem; font-weight: 700; color: #38bdf8; margin-bottom: 0px; }
    .sub-title { font-size: 1.1rem; color: #94a3b8; margin-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">⚡ Pick My LLM</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">A Free Benchmarking and Recommendation Tool for Choosing the Right AI Model</p>', unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR: Configuration & API Keys
# -------------------------------------------------------------
st.sidebar.header("🔐 API Configuration & Keys")
st.sidebar.info("Enter your free API keys below. Keys are kept strictly in memory and never saved.")

groq_key = st.sidebar.text_input("Groq API Key", type="password")
google_key = st.sidebar.text_input("Google AI Studio API Key", type="password")
openrouter_key = st.sidebar.text_input("OpenRouter API Key", type="password")

api_keys = {
    "Groq": groq_key,
    "Google AI Studio": google_key,
    "OpenRouter": openrouter_key
}

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Recommendation Weights")
w_acc = st.sidebar.slider("Accuracy Weight", 0.0, 1.0, 0.6, 0.1)
w_spd = st.sidebar.slider("Speed (Latency) Weight", 0.0, 1.0, 0.3, 0.1)
w_cst = st.sidebar.slider("Estimated Cost Weight", 0.0, 1.0, 0.1, 0.1)

user_weights = {"accuracy": w_acc, "speed": w_spd, "cost": w_cst}

st.sidebar.markdown("---")
demo_mode = st.sidebar.checkbox("🚀 Enable Demo Mode (No API Key Required)", value=False)

# -------------------------------------------------------------
# MODEL SELECTION
# -------------------------------------------------------------
st.subheader("1️⃣ Select Models to Compare")
col1, col2, col3 = st.columns(3)

selected_models = []

with col1:
    st.markdown("**Groq (Fast & Free Tier)**")
    if st.checkbox("Llama 3.1 8B Instant", value=True):
        selected_models.append({"provider": "Groq", "model": "llama-3.1-8b-instant"})
    if st.checkbox("Llama 3.3 70B Versatile", value=False):
        selected_models.append({"provider": "Groq", "model": "llama-3.3-70b-versatile"})

with col2:
    st.markdown("**Google AI Studio (Gemini)**")
    if st.checkbox("Gemini 2.5 Flash", value=True):
        selected_models.append({"provider": "Google AI Studio", "model": "gemini-2.5-flash"})

with col3:
    st.markdown("**OpenRouter (Free Models)**")
    if st.checkbox("Mistral 7B Instruct (Free)", value=False):
        selected_models.append({"provider": "OpenRouter", "model": "mistralai/mistral-7b-instruct:free"})

# -------------------------------------------------------------
# TEST DATASET SETUP
# -------------------------------------------------------------
st.subheader("2️⃣ Load Test Questions")

questions_data = []

# Load default sample dataset if available
default_path = "data/sample_dataset.json"
if os.path.exists(default_path):
    with open(default_path, "r", encoding="utf-8") as f:
        default_questions = json.load(f)
else:
    default_questions = [
        {"id": 1, "category": "Python", "question": "What is print(type([]))? Reply with option letter only.\nA) dict\nB) list", "correct_answer": "B"},
        {"id": 2, "category": "General", "question": "Capital of Gujarat? Just city name.", "correct_answer": "Gandhinagar"}
    ]

uploaded_file = st.file_uploader("Upload custom JSON question file (optional)", type=["json"])
if uploaded_file is not None:
    try:
        questions_data = json.load(uploaded_file)
        st.success(f"Successfully loaded {len(questions_data)} questions from uploaded file!")
    except Exception as e:
        st.error(f"Error reading JSON file: {e}")
        questions_data = default_questions
else:
    questions_data = default_questions
    st.info(f"Using default sample test suite ({len(questions_data)} questions loaded).")

# -------------------------------------------------------------
# RUN BENCHMARK BUTTON
# -------------------------------------------------------------
st.subheader("3️⃣ Execute & Evaluate")

if st.button("🚀 Run Benchmark Comparison", type="primary"):
    if not selected_models:
        st.warning("Please select at least one model to test.")
    elif not demo_mode and not any(api_keys.values()):
        st.error("Please enter at least one provider API key in the sidebar, or enable Demo Mode.")
    else:
        if demo_mode:
            st.info("Running in Demo Mode using simulated benchmark responses...")
            # Simulated demo data for instant presentation
            raw_results = []
            for m in selected_models:
                m_name = f"{m['provider']} / {m['model']}"
                for q in questions_data:
                    raw_results.append({
                        "model": m_name,
                        "question_id": q["id"],
                        "category": q.get("category", "General"),
                        "question": q["question"],
                        "correct_answer": q["correct_answer"],
                        "answer": q["correct_answer"] if "8b" in m_name or "flash" in m_name else "Wrong Answer Example",
                        "latency": 0.35 if "8b" in m_name else 0.85,
                        "prompt_tokens": 45,
                        "completion_tokens": 12,
                        "error": None
                    })
            results_df = pd.DataFrame(raw_results)
        else:
            with st.spinner("Running live benchmark across selected models... Please wait."):
                results_df = run_benchmark(questions_data, selected_models, api_keys, delay_seconds=1)

        # Score the answers automatically
        scores = []
        for _, row in results_df.iterrows():
            # Use exact match or keyword match depending on question type
            is_match = score_exact_match(row["answer"], row["correct_answer"])
            if not is_match:
                is_match = score_keyword_match(row["answer"], row["correct_answer"])
            scores.append(is_match)

        # Aggregate and Rank
        agg_df = aggregate_results(results_df, scores)
        ranked_df = rank_models(agg_df, weights=user_weights)

        # Save into session state so results persist on UI re-runs
        st.session_state["results_df"] = results_df
        st.session_state["ranked_df"] = ranked_df
        st.success("Benchmark completed successfully!")

# -------------------------------------------------------------
# DISPLAY RESULTS & DASHBOARD CHARTS
# -------------------------------------------------------------
if "ranked_df" in st.session_state and not st.session_state["ranked_df"].empty:
    st.markdown("---")
    st.header("📊 Benchmark Results & Recommendation")

    ranked_df = st.session_state["ranked_df"]
    results_df = st.session_state["results_df"]

    # Recommendation Summary Box
    summary_text = generate_recommendation_summary(ranked_df)
    st.success(summary_text)

    # Leaderboard Table
    st.subheader("🏆 Model Leaderboard")
    st.dataframe(ranked_df[["model", "accuracy_pct", "avg_latency", "total_cost", "final_score"]], use_container_width=True)

    # Visualizations using Matplotlib
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Accuracy Comparison (%)")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(ranked_df["model"], ranked_df["accuracy_pct"], color="#38bdf8")
        ax.set_xlabel("Accuracy (%)")
        ax.set_ylabel("Model")
        ax.set_xlim(0, 100)
        st.pyplot(fig)

    with col_chart2:
        st.subheader("Average Latency (Seconds)")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(ranked_df["model"], ranked_df["avg_latency"], color="#f43f5e")
        ax.set_xlabel("Latency (s)")
        ax.set_ylabel("Model")
        st.pyplot(fig)

    # Detailed Raw Results & Wrong Answers Explorer
    st.subheader("🔍 Detailed Question-by-Question Responses")
    st.dataframe(results_df, use_container_width=True)

    # CSV Download Button
    csv_data = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Benchmark Results (CSV)",
        data=csv_data,
        file_name="pick_my_llm_results.csv",
        mime="text/csv"
    )