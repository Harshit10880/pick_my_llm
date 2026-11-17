import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt

from core.runner import run_benchmark
from core.scorer import score_exact_match, score_keyword_match, score_llm_judge
from core.recommender import aggregate_results, rank_models, generate_recommendation_summary
from core.history import load_history, save_run, clear_history, delete_run
from core.viz import create_radar_chart, create_category_chart, create_tradeoff_chart

# Page Configuration
st.set_page_config(
    page_title="Pick My LLM - AI Benchmarking Tool",
    page_icon="⚡",
    layout="wide"
)

# Custom Styling & Modern Dark Accent
st.markdown("""
    <style>
    .main-title { font-size: 2.5rem; font-weight: 700; color: #38bdf8; margin-bottom: 0px; }
    .sub-title { font-size: 1.1rem; color: #94a3b8; margin-bottom: 1.5rem; }
    .metric-card { background-color: #1e293b; border-radius: 8px; padding: 1rem; border: 1px solid #334155; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">⚡ Pick My LLM</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Free Benchmarking & Recommendation Engine to Pick the Optimal AI Model for Your Task</p>', unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR: Configuration & API Keys
# -------------------------------------------------------------
st.sidebar.header("🔐 API Configuration & Keys")
st.sidebar.info("Enter free API keys below. Keys are kept strictly in memory and never saved to disk.")

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
st.sidebar.header("🧠 Scoring Method")
scoring_mode = st.sidebar.radio(
    "Choose how answers are evaluated:",
    ["Auto (Exact → Keyword)", "LLM-as-a-Judge (AI evaluates open-ended answers)"],
    index=0,
    help="Auto works for MCQ/short answers. LLM Judge is better for paragraph or open-ended responses."
)

if scoring_mode == "LLM-as-a-Judge (AI evaluates open-ended answers)":
    judge_provider_options = ["Groq", "Google AI Studio", "OpenRouter"]
    judge_provider = st.sidebar.selectbox("Judge Model Provider", judge_provider_options)
    judge_model_defaults = {
        "Groq": "llama-3.1-8b-instant",
        "Google AI Studio": "gemini-2.5-flash",
        "OpenRouter": "mistralai/mistral-7b-instruct:free"
    }
    judge_model = st.sidebar.text_input("Judge Model Name", value=judge_model_defaults[judge_provider])
    st.sidebar.info("The judge model uses your already-entered API key for the selected provider.")
else:
    judge_provider = None
    judge_model = None

st.sidebar.markdown("---")
demo_mode = st.sidebar.checkbox("🚀 Enable Demo Mode (No API Key Required)", value=False)

# -------------------------------------------------------------
# MAIN APP TABS: Live Benchmark vs. History
# -------------------------------------------------------------
tab_benchmark, tab_history = st.tabs(["🚀 Live Benchmark", "📜 Benchmark History & Trends"])

# =============================================================
# TAB 1: LIVE BENCHMARK
# =============================================================
with tab_benchmark:
    # MODEL SELECTION
    st.subheader("1️⃣ Select Models to Compare")
    col1, col2, col3 = st.columns(3)

    selected_models = []

    with col1:
        st.markdown("**Groq (Ultra-Fast Free Tier)**")
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

    # TEST DATASET SETUP
    st.subheader("2️⃣ Load Test Questions")

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
        st.info(f"Using default sample test suite ({len(questions_data)} questions across multiple categories).")

    # RUN BENCHMARK BUTTON
    st.subheader("3️⃣ Execute & Evaluate")

    if st.button("🚀 Run Benchmark Comparison", type="primary"):
        if not selected_models:
            st.warning("Please select at least one model to test.")
        elif not demo_mode and not any(api_keys.values()):
            st.error("Please enter at least one provider API key in the sidebar, or enable Demo Mode.")
        else:
            if demo_mode:
                st.info("Running in Demo Mode using simulated benchmark responses...")
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

            # Score answers
            scores = []
            judge_verdicts = []
            scoring_progress = st.progress(0, text="Scoring model responses...")
            total_rows = len(results_df)

            for i, (_, row) in enumerate(results_df.iterrows()):
                scoring_progress.progress((i + 1) / total_rows, text=f"Evaluating response {i+1}/{total_rows}...")

                if scoring_mode == "LLM-as-a-Judge (AI evaluates open-ended answers)":
                    judge_key = api_keys.get(judge_provider, "")
                    if demo_mode or not judge_key:
                        verdict_bool = score_exact_match(row["answer"], row["correct_answer"]) or \
                                       score_keyword_match(row["answer"], row["correct_answer"])
                        verdict_str = "✅ CORRECT" if verdict_bool else "❌ WRONG"
                    else:
                        verdict_str_raw = score_llm_judge(
                            question=row["question"],
                            correct_answer=row["correct_answer"],
                            model_answer=row["answer"],
                            judge_provider=judge_provider,
                            judge_model=judge_model,
                            judge_api_key=judge_key
                        )
                        verdict_bool = (verdict_str_raw == "CORRECT")
                        verdict_str = "✅ CORRECT" if verdict_bool else ("⚠️ ERROR" if verdict_str_raw == "ERROR" else "❌ WRONG")
                else:
                    verdict_bool = score_exact_match(row["answer"], row["correct_answer"])
                    if not verdict_bool:
                        verdict_bool = score_keyword_match(row["answer"], row["correct_answer"])
                    verdict_str = "✅ CORRECT" if verdict_bool else "❌ WRONG"

                scores.append(verdict_bool)
                judge_verdicts.append(verdict_str)

            scoring_progress.empty()
            results_df["verdict"] = judge_verdicts

            # Aggregate and Rank
            agg_df = aggregate_results(results_df, scores)
            ranked_df = rank_models(agg_df, weights=user_weights)

            # Persist to session state
            st.session_state["results_df"] = results_df
            st.session_state["ranked_df"] = ranked_df

            # Automatically record run in history
            try:
                saved_entry = save_run(ranked_df, results_df, user_weights, scoring_mode)
                st.success(f"Benchmark completed! Run recorded to history (ID: {saved_entry['run_id']}).")
            except Exception as e:
                st.warning(f"Benchmark completed, but failed to cache history: {e}")

    # DISPLAY RESULTS & VISUALIZATIONS
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
        st.dataframe(
            ranked_df[["model", "accuracy_pct", "avg_latency", "total_cost", "final_score"]],
            use_container_width=True
        )

        # Interactive Visualizations
        st.subheader("📈 Multi-Dimensional Visualizations")
        col_viz1, col_viz2 = st.columns(2)

        with col_viz1:
            st.markdown("**🕸️ Multi-Attribute Radar Profile**")
            radar_fig = create_radar_chart(ranked_df)
            st.plotly_chart(radar_fig, use_container_width=True, key="live_radar_chart")

        with col_viz2:
            st.markdown("**⚡ Latency vs. Accuracy Trade-Off Matrix**")
            tradeoff_fig = create_tradeoff_chart(ranked_df)
            st.plotly_chart(tradeoff_fig, use_container_width=True, key="live_tradeoff_chart")

        # Category-wise Breakdown
        if "category" in results_df.columns and len(results_df["category"].unique()) > 1:
            st.markdown("**🎯 Category-Wise Performance Breakdown**")
            cat_fig = create_category_chart(results_df)
            if cat_fig:
                st.plotly_chart(cat_fig, use_container_width=True, key="live_category_chart")

        # Detailed Question Responses Explorer
        st.subheader("🔍 Detailed Question-by-Question Responses")
        display_cols = ["model", "category", "question", "correct_answer", "answer", "verdict", "latency", "prompt_tokens", "completion_tokens", "error"]
        available_cols = [c for c in display_cols if c in results_df.columns]
        st.dataframe(results_df[available_cols], use_container_width=True)

        # CSV Download Button
        csv_data = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Benchmark Results (CSV)",
            data=csv_data,
            file_name="pick_my_llm_results.csv",
            mime="text/csv",
            key="download_live_csv"
        )

# =============================================================
# TAB 2: BENCHMARK HISTORY & TRENDS
# =============================================================
with tab_history:
    st.subheader("📜 Historical Benchmark Runs")
    history_runs = load_history()

    if not history_runs:
        st.info("No saved benchmark runs yet. Run a benchmark from the first tab to begin tracking history!")
    else:
        # History Summary Metrics
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Total Runs Cached", len(history_runs))
        with col_m2:
            latest_run = history_runs[0]
            st.metric("Latest Run Date", latest_run["timestamp"].split()[0])
        with col_m3:
            st.metric("Latest Winner", latest_run.get("winner", "N/A"))

        st.markdown("---")
        
        # Run Selector Dropdown
        run_options = {
            r["run_id"]: f"🕒 {r['timestamp']} — 🏆 Winner: {r.get('winner', 'N/A')} ({r.get('num_models', 0)} models, {r.get('num_questions', 0)} questions)"
            for r in history_runs
        }
        
        selected_run_id = st.selectbox(
            "Select a past run to inspect:",
            options=list(run_options.keys()),
            format_func=lambda x: run_options[x]
        )

        selected_run = next((r for r in history_runs if r["run_id"] == selected_run_id), None)

        if selected_run:
            st.markdown(f"### Run Details: `{selected_run['run_id']}`")
            st.markdown(
                f"- **Timestamp:** {selected_run['timestamp']}  \n"
                f"- **Scoring Mode:** {selected_run.get('scoring_mode', 'N/A')}  \n"
                f"- **Top Performer:** **{selected_run.get('winner', 'N/A')}** ({selected_run.get('winner_accuracy', 0)}% accuracy)"
            )

            hist_ranked_df = pd.DataFrame(selected_run.get("ranked_df", []))
            hist_results_df = pd.DataFrame(selected_run.get("results_df", []))

            if not hist_ranked_df.empty:
                st.subheader("🏆 Leaderboard")
                st.dataframe(
                    hist_ranked_df[["model", "accuracy_pct", "avg_latency", "total_cost", "final_score"]],
                    use_container_width=True
                )

                # Charts for this historical run
                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    st.markdown("**🕸️ Radar Performance Profile**")
                    h_radar = create_radar_chart(hist_ranked_df)
                    st.plotly_chart(h_radar, use_container_width=True, key=f"hist_radar_{selected_run_id}")

                with col_h2:
                    st.markdown("**⚡ Latency vs. Accuracy**")
                    h_tradeoff = create_tradeoff_chart(hist_ranked_df)
                    st.plotly_chart(h_tradeoff, use_container_width=True, key=f"hist_tradeoff_{selected_run_id}")

                if not hist_results_df.empty and "category" in hist_results_df.columns:
                    st.markdown("**🎯 Category Breakdown**")
                    h_cat = create_category_chart(hist_results_df)
                    if h_cat:
                        st.plotly_chart(h_cat, use_container_width=True, key=f"hist_cat_{selected_run_id}")

            if not hist_results_df.empty:
                st.subheader("🔍 Detailed Responses for this Run")
                display_cols = ["model", "category", "question", "correct_answer", "answer", "verdict", "latency", "prompt_tokens", "completion_tokens", "error"]
                avail_cols = [c for c in display_cols if c in hist_results_df.columns]
                st.dataframe(hist_results_df[avail_cols], use_container_width=True)

                # Download button for past run
                past_csv = hist_results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"📥 Download CSV for {selected_run_id}",
                    data=past_csv,
                    file_name=f"{selected_run_id}_results.csv",
                    mime="text/csv",
                    key=f"download_hist_csv_{selected_run_id}"
                )

            st.markdown("---")
            # History actions: delete single run or clear all
            act_col1, act_col2 = st.columns([1, 1])
            with act_col1:
                if st.button("🗑️ Delete Selected Run", key="delete_single_run"):
                    delete_run(selected_run_id)
                    st.success(f"Run {selected_run_id} deleted!")
                    st.rerun()

            with act_col2:
                if st.button("⚠️ Clear All Run History", key="clear_all_runs"):
                    clear_history()
                    st.success("All run history has been cleared.")
                    st.rerun()
