import re
from core.client import call_llm

def score_exact_match(model_answer: str, correct_answer: str) -> bool:
    """
    Checks if the model answer matches the correct answer exactly (case-insensitive, stripped).
    Best for multiple-choice questions.
    """
    if not model_answer:
        return False
    # Clean text: lowercase and remove extra spaces or punctuation
    cleaned_model = model_answer.strip().lower()
    cleaned_correct = correct_answer.strip().lower()
    return cleaned_model == cleaned_correct

def score_keyword_match(model_answer: str, correct_answer: str) -> bool:
    """
    Checks if the correct answer substring is present inside the model's response.
    Best for short factual answers.
    """
    if not model_answer:
        return False
    return correct_answer.strip().lower() in model_answer.strip().lower()

def score_llm_judge(question: str, correct_answer: str, model_answer: str, judge_provider: str, judge_model: str, judge_api_key: str) -> str:
    """
    Uses an independent free LLM to judge if an open-ended answer is CORRECT or WRONG.
    """
    if not model_answer:
        return "WRONG"

    judge_prompt = f"""
You are an objective AI evaluation judge. 
Compare the Model's Answer against the Correct Answer for the given Question.

Question: {question}
Correct Answer: {correct_answer}
Model's Answer: {model_answer}

Is the model's answer substantially correct and accurate based on the correct answer? 
Reply with ONLY ONE WORD: "CORRECT" or "WRONG". Do not add any extra punctuation or explanation.
"""
    
    # Call the judge model using our unified client
    res = call_llm(provider=judge_provider, model_name=judge_model, prompt=judge_prompt, api_key=judge_api_key)
    
    if res["error"]:
        return "ERROR"
    
    reply = res["response"].strip().upper()
    if "CORRECT" in reply:
        return "CORRECT"
    else:
        return "WRONG"