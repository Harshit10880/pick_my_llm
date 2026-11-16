import time
import os
from google import genai
from google.genai import types
from groq import Groq
import openai

def call_llm(provider: str, model_name: str, prompt: str, api_key: str):
    """
    Unified client to call Groq, Google AI Studio, or OpenRouter.
    Returns:
        dict: {
            "response": str,
            "latency": float (in seconds),
            "prompt_tokens": int,
            "completion_tokens": int,
            "error": str or None
        }
    """
    start_time = time.time()
    result = {
        "response": "",
        "latency": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "error": None
    }

    try:
        if provider == "Groq":
            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            result["response"] = completion.choices[0].message.content
            if hasattr(completion, 'usage') and completion.usage:
                result["prompt_tokens"] = completion.usage.prompt_tokens
                result["completion_tokens"] = completion.usage.completion_tokens

        elif provider == "Google AI Studio":
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            result["response"] = response.text
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                result["prompt_tokens"] = response.usage_metadata.prompt_token_count
                result["completion_tokens"] = response.usage_metadata.candidates_token_count

        elif provider == "OpenRouter":
            client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            result["response"] = completion.choices[0].message.content
            if hasattr(completion, 'usage') and completion.usage:
                result["prompt_tokens"] = completion.usage.prompt_tokens
                result["completion_tokens"] = completion.usage.completion_tokens

        else:
            result["error"] = f"Unknown provider: {provider}"

    except Exception as e:
        result["error"] = str(e)

    result["latency"] = round(time.time() - start_time, 3)
    return result