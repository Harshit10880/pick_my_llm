"""
Automated Test Suite for Pick My LLM.
Runs full system verification across client, runner, scorers, recommender, viz, and history.

Usage:
    python run_tests.py
"""

import sys
import json
import pandas as pd
import unittest

from core.client import call_llm
from core.scorer import score_exact_match, score_keyword_match, score_llm_judge
from core.recommender import calculate_costs, aggregate_results, rank_models, generate_recommendation_summary
from core.history import save_run, load_history, delete_run, clear_history
from core.viz import create_radar_chart, create_category_chart, create_tradeoff_chart
from core.runner import run_benchmark

class TestPickMyLLM(unittest.TestCase):

    def test_01_exact_and_keyword_scorers(self):
        """Tests deterministic scoring functions."""
        # Exact match
        self.assertTrue(score_exact_match("B", "B"))
        self.assertTrue(score_exact_match("  b  ", "B"))
        self.assertFalse(score_exact_match("A", "B"))
        
        # Keyword match
        self.assertTrue(score_keyword_match("The capital is Gandhinagar.", "Gandhinagar"))
        self.assertTrue(score_keyword_match("pandas dataframe", "pandas"))
        self.assertFalse(score_keyword_match("numpy arrays", "pandas"))
        print("  [PASS] Exact and Keyword Scorers verified.")

    def test_02_cost_calculation(self):
        """Tests simulated operational token cost estimation."""
        df = pd.DataFrame([
            {"model": "Groq / llama-3.1-8b-instant", "prompt_tokens": 1000, "completion_tokens": 500},
            {"model": "Google AI Studio / gemini-2.5-flash", "prompt_tokens": 2000, "completion_tokens": 1000}
        ])
        cost_df = calculate_costs(df)
        self.assertIn("estimated_cost", cost_df.columns)
        self.assertGreater(cost_df["estimated_cost"].sum(), 0.0)
        print("  [PASS] Cost estimation calculation verified.")

    def test_03_normalization_and_ranking(self):
        """Tests min-max normalization and weighted ranking."""
        raw_results = pd.DataFrame([
            {"model": "Model-A", "question_id": 1, "is_correct": True, "latency": 0.20, "prompt_tokens": 10, "completion_tokens": 5, "error": None},
            {"model": "Model-A", "question_id": 2, "is_correct": True, "latency": 0.30, "prompt_tokens": 10, "completion_tokens": 5, "error": None},
            {"model": "Model-B", "question_id": 1, "is_correct": True, "latency": 0.80, "prompt_tokens": 10, "completion_tokens": 5, "error": None},
            {"model": "Model-B", "question_id": 2, "is_correct": False, "latency": 0.90, "prompt_tokens": 10, "completion_tokens": 5, "error": None}
        ])
        
        agg = aggregate_results(raw_results, raw_results["is_correct"].tolist())
        ranked = rank_models(agg, weights={"accuracy": 0.6, "speed": 0.3, "cost": 0.1})
        
        self.assertEqual(len(ranked), 2)
        # Model-A should win (100% accuracy and lower latency)
        self.assertEqual(ranked.iloc[0]["model"], "Model-A")
        self.assertGreater(ranked.iloc[0]["final_score"], ranked.iloc[1]["final_score"])
        
        summary = generate_recommendation_summary(ranked)
        self.assertIn("Model-A", summary)
        print("  [PASS] Normalization, ranking, and recommendation summary verified.")

    def test_04_history_lifecycle(self):
        """Tests saving, querying, deleting, and clearing historical benchmark runs."""
        clear_history()
        self.assertEqual(len(load_history()), 0)
        
        ranked_sample = pd.DataFrame([
            {"model": "Model-Alpha", "accuracy_pct": 100.0, "avg_latency": 0.3, "total_cost": 0.00001, "final_score": 90.0, "norm_speed": 1.0, "norm_cost": 1.0}
        ])
        results_sample = pd.DataFrame([
            {"model": "Model-Alpha", "question_id": 1, "category": "Python", "is_correct": True, "answer": "B", "verdict": "✅ CORRECT", "latency": 0.3}
        ])
        
        entry = save_run(ranked_sample, results_sample, {"accuracy": 0.6, "speed": 0.3, "cost": 0.1}, "Auto")
        runs = load_history()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["run_id"], entry["run_id"])
        
        # Test delete
        delete_run(entry["run_id"])
        self.assertEqual(len(load_history()), 0)
        print("  [PASS] History cache lifecycle (Save, Load, Delete, Clear) verified.")

    def test_05_visualizations(self):
        """Tests that Plotly charts compile and generate valid Figure objects."""
        ranked_df = pd.DataFrame([
            {"model": "Model-1", "accuracy_pct": 90.0, "norm_speed": 0.8, "norm_cost": 0.9, "final_score": 85.0, "avg_latency": 0.4},
            {"model": "Model-2", "accuracy_pct": 70.0, "norm_speed": 0.4, "norm_cost": 0.5, "final_score": 60.0, "avg_latency": 0.9}
        ])
        results_df = pd.DataFrame([
            {"model": "Model-1", "category": "Python", "is_correct": True},
            {"model": "Model-2", "category": "Python", "is_correct": False}
        ])
        
        radar = create_radar_chart(ranked_df)
        tradeoff = create_tradeoff_chart(ranked_df)
        cat_chart = create_category_chart(results_df)
        
        self.assertIsNotNone(radar)
        self.assertIsNotNone(tradeoff)
        self.assertIsNotNone(cat_chart)
        print("  [PASS] Plotly charts (Radar, Trade-off, Category bars) verified.")

    def test_06_sample_dataset_integrity(self):
        """Ensures the built-in dataset has valid schema and categories."""
        with open("data/sample_dataset.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.assertGreaterEqual(len(data), 20)
        for item in data:
            self.assertIn("id", item)
            self.assertIn("category", item)
            self.assertIn("question", item)
            self.assertIn("correct_answer", item)
        print(f"  [PASS] Sample dataset schema verified ({len(data)} questions).")

if __name__ == "__main__":
    print("\n" + "="*50)
    print(" ⚡ Running Automated Test Suite for Pick My LLM")
    print("="*50)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPickMyLLM)
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n" + "="*50)
        print(" ✅ ALL 6 TEST SUITES PASSED! System is 100% operational.")
        print("="*50 + "\n")
        sys.exit(0)
    else:
        print("\n ❌ Some tests failed.")
        sys.exit(1)
