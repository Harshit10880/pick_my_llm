import time
import pandas as pd
from core.client import call_llm

def run_benchmark(questions: list, selected_providers: list, api_keys: dict, delay_seconds: int = 2):
    """
    Runs benchmark questions across multiple providers/models with rate-limit handling.
    
    Parameters:
        questions (list of dict): List containing test questions.
        selected_providers (list of dict): List of models to test, e.g., 
          [{"provider": "Groq", "model": "llama-3.1-8b-instant"}, ...]
        api_keys (dict): Dictionary of API keys {"Groq": "...", "Google AI Studio": "...", ...}
        delay_seconds (int): Pause between calls to respect free-tier rate limits.
        
    Returns:
        pd.DataFrame: Comprehensive results table.
    """
    results = []

    for item in questions:
        q_id = item.get("id")
        question_text = item.get("question")
        correct_answer = item.get("correct_answer")
        category = item.get("category", "General")

        for target in selected_providers:
            provider = target["provider"]
            model_name = target["model"]
            custom_base_url = target.get("custom_base_url") or api_keys.get(f"{provider}_base_url")
            
            api_key = api_keys.get(provider, "")

            # Ollama is local, so no API key is required
            is_local = (provider == "Ollama (Local Offline)")
            if not api_key and not is_local:
                results.append({
                    "model": f"{provider} / {model_name}",
                    "question_id": q_id,
                    "category": category,
                    "question": question_text,
                    "correct_answer": correct_answer,
                    "answer": "",
                    "latency": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "error": "API Key missing"
                })
                continue

            # Attempt call with exponential backoff retry loop for rate limits
            success = False
            attempts = 0
            max_retries = 3
            res_data = {}

            while not success and attempts < max_retries:
                attempts += 1
                res_data = call_llm(
                    provider=provider,
                    model_name=model_name,
                    prompt=question_text,
                    api_key=api_key or "ollama",
                    custom_base_url=custom_base_url
                )
                
                if res_data["error"] and ("rate_limit" in res_data["error"].lower() or "429" in res_data["error"]):
                    sleep_time = delay_seconds * (2 ** attempts)
                    time.sleep(sleep_time)
                else:
                    success = True

            # Record results for this question & model pair
            results.append({
                "model": f"{provider} / {model_name}",
                "question_id": q_id,
                "category": category,
                "question": question_text,
                "correct_answer": correct_answer,
                "answer": res_data["response"],
                "latency": res_data["latency"],
                "prompt_tokens": res_data["prompt_tokens"],
                "completion_tokens": res_data["completion_tokens"],
                "error": res_data["error"]
            })

            # Polite pause between requests to stay safe on free limits
            time.sleep(delay_seconds)

    return pd.DataFrame(results)
