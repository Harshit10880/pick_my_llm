import pandas as pd
import numpy as np

# Estimated public pricing per 1 million tokens (USD) for common free-tier/open models
# Used for cost estimation simulations
MODEL_PRICING = {
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "default": {"input": 0.20, "output": 0.20}
}

def calculate_costs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates simulated operational cost based on token consumption and published pricing.
    """
    costs = []
    for _, row in df.iterrows():
        model_full_name = row["model"]
        # Extract model name string after provider slash
        model_key = model_full_name.split("/")[-1].strip()
        
        pricing = MODEL_PRICING.get(model_key, MODEL_PRICING["default"])
        
        prompt_toks = row.get("prompt_tokens", 0)
        comp_toks = row.get("completion_tokens", 0)
        
        cost = (prompt_toks / 1_000_000 * pricing["input"]) + (comp_toks / 1_000_000 * pricing["output"])
        costs.append(cost)
        
    df["estimated_cost"] = costs
    return df

def aggregate_results(results_df: pd.DataFrame, scores_list: list) -> pd.DataFrame:
    """
    Aggregates individual question scores into model-level performance metrics:
    Accuracy %, Average Latency, Total Cost, and Error Count.
    """
    if results_df.empty:
        return pd.DataFrame()

    results_df["is_correct"] = scores_list
    results_df = calculate_costs(results_df)

    # Group by model name
    agg = results_df.groupby("model").agg(
        total_questions=("question_id", "count"),
        correct_count=("is_correct", "sum"),
        avg_latency=("latency", "mean"),
        total_cost=("estimated_cost", "sum"),
        error_count=("error", lambda x: x.notnull().sum())
    ).reset_index()

    # Calculate Accuracy Percentage
    agg["accuracy_pct"] = (agg["correct_count"] / agg["total_questions"]) * 100
    agg["avg_latency"] = agg["avg_latency"].round(3)
    agg["accuracy_pct"] = agg["accuracy_pct"].round(2)
    agg["total_cost"] = agg["total_cost"].round(6)

    return agg

def rank_models(agg_df: pd.DataFrame, weights: dict = {"accuracy": 0.6, "speed": 0.3, "cost": 0.1}) -> pd.DataFrame:
    """
    Ranks models using normalized weights for accuracy, speed, and cost.
    Weights default to: Accuracy (60%), Speed (30%), Cost (10%).
    """
    if agg_df.empty:
        return agg_df

    df = agg_df.copy()

    # Normalize Accuracy (higher is better, range 0 to 1)
    max_acc = df["accuracy_pct"].max()
    min_acc = df["accuracy_pct"].min()
    if max_acc == min_acc:
        df["norm_accuracy"] = 1.0
    else:
        df["norm_accuracy"] = (df["accuracy_pct"] - min_acc) / (max_acc - min_acc)

    # Normalize Speed / Latency (lower latency is better, invert scale)
    max_lat = df["avg_latency"].max()
    min_lat = df["avg_latency"].min()
    if max_lat == min_lat:
        df["norm_speed"] = 1.0
    else:
        # Inverted so lower time yields a higher score
        df["norm_speed"] = (max_lat - df["avg_latency"]) / (max_lat - min_lat)

    # Normalize Cost (lower cost is better, invert scale)
    max_cost = df["total_cost"].max()
    min_cost = df["total_cost"].min()
    if max_cost == min_cost:
        df["norm_cost"] = 1.0
    else:
        df["norm_cost"] = (max_cost - df["total_cost"]) / (max_cost - min_cost)

    # Calculate Weighted Final Score
    w_acc = weights.get("accuracy", 0.6)
    w_spd = weights.get("speed", 0.3)
    w_cst = weights.get("cost", 0.1)

    df["final_score"] = (
        (df["norm_accuracy"] * w_acc) + 
        (df["norm_speed"] * w_spd) + 
        (df["norm_cost"] * w_cst)
    ) * 100

    # Sort models by highest final score
    df = df.sort_values(by="final_score", ascending=False).reset_index(drop=True)
    return df

def generate_recommendation_summary(ranked_df: pd.DataFrame) -> str:
    """
    Generates a plain-language summary explaining the winning model.
    """
    if ranked_df.empty:
        return "No benchmark data available yet to generate a recommendation."

    winner = ranked_df.iloc[0]
    model_name = winner["model"]
    acc = winner["accuracy_pct"]
    latency = winner["avg_latency"]

    summary = f"🏆 **Top Recommendation:** **{model_name}** is the best choice for your task.\n\n"
    summary += f"- **Accuracy:** Achieved **{acc}%** accuracy across your test dataset.\n"
    summary += f"- **Speed:** Averaged **{latency} seconds** per response.\n"

    if len(ranked_df) > 1:
        runner_up = ranked_df.iloc[1]
        summary += f"\n*Why not others?* While runner-up **{runner_up['model']}** scored {runner_up['accuracy_pct']}% accuracy, **{model_name}** offers the optimal balance of speed and reliability based on your custom preferences."

    return summary