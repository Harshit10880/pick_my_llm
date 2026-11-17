import os
import json
from datetime import datetime
import pandas as pd

HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cached_results.json")

def load_history() -> list:
    """
    Loads all previous benchmark run records from cached_results.json.
    """
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except Exception as e:
        print(f"Error loading benchmark history: {e}")
        return []

def save_run(ranked_df: pd.DataFrame, results_df: pd.DataFrame, weights: dict, scoring_mode: str) -> dict:
    """
    Saves a completed benchmark run into cached_results.json.
    """
    history = load_history()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    winner = ranked_df.iloc[0]["model"] if not ranked_df.empty else "N/A"
    winner_acc = float(ranked_df.iloc[0]["accuracy_pct"]) if not ranked_df.empty else 0.0
    
    # Extract distinct categories from results
    categories = list(results_df["category"].unique()) if "category" in results_df.columns else []

    run_entry = {
        "run_id": run_id,
        "timestamp": timestamp,
        "winner": winner,
        "winner_accuracy": winner_acc,
        "num_models": len(ranked_df),
        "num_questions": len(results_df["question_id"].unique()) if "question_id" in results_df.columns else len(results_df),
        "categories": categories,
        "scoring_mode": scoring_mode,
        "weights": weights,
        "ranked_df": ranked_df.to_dict(orient="records"),
        "results_df": results_df.to_dict(orient="records")
    }
    
    # Prepend new run so newest appears first
    history.insert(0, run_entry)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
        
    return run_entry

def clear_history():
    """
    Clears all saved benchmark runs.
    """
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2)

def delete_run(run_id: str):
    """
    Deletes a specific run by its run_id.
    """
    history = load_history()
    updated = [r for r in history if r.get("run_id") != run_id]
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
