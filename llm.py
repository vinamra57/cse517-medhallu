import os
import openai
from dotenv import load_dotenv
from prompt import *

load_dotenv()


def _get_client():
    """Auto-detect API provider from available environment variables.
    Prefers GROQ_API_KEY (free) over OPENAI_API_KEY if both are set.
    """
    groq_key = os.environ.get("GROQ_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if groq_key:
        return openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key
        )
    elif openai_key:
        return openai.OpenAI(api_key=openai_key)
    else:
        raise RuntimeError("No API key found. Set GROQ_API_KEY or OPENAI_API_KEY in .env")


class LLM():
    def __init__(self, model: str, system_prompt: str):
        self.client = _get_client()
        self.model = model
        self.system_prompt = system_prompt

    def get_response(self, user_prompt: str) -> str:
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_prompt}]

        try:
            response = self.client.chat.completions.create(
                model = self.model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            return None
