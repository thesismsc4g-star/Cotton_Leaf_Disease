import os
from typing import List, Dict

from groq import Groq


def build_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")
    return Groq(api_key=api_key)


def generate_response(model: str, messages: List[Dict[str, str]]) -> str:
    client = build_groq_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=600,
    )
    return response.choices[0].message.content
