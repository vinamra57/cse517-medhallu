import os
import openai
from dotenv import load_dotenv
from prompt import *

load_dotenv()

class LLM():
    def __init__(self, model: str):
        self.client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY")
        )
        self.model = model
    
    def get_hallucinated_response(self, user_prompt: str) -> str:
        messages = [{"role": "system", "content": hallucination_prompt}, {"role": "user", "content": user_prompt}]

        try:
            response = self.client.chat.completions.create(
                model = self.model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            return None
