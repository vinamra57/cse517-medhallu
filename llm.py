import os
from typing import Optional

import openai
import torch
from dotenv import load_dotenv

load_dotenv()

# OpenAI model prefixes - route these to OpenAI API
OPENAI_PREFIXES = ("gpt-", "o1-", "o3-")

# Models available on Groq (checked against their catalog)
GROQ_MODELS = {
    "qwen/qwen3-32b", "llama-3.1-8b-instant", "llama-3.3-70b-versatile",
    "moonshotai/kimi-k2-instruct", "openai/gpt-oss-20b", "openai/gpt-oss-120b",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
}

# Cache for local HuggingFace pipelines (avoid reloading per call)
_local_pipelines = {}

# Cache for HuggingFace InferenceClient instances
_hf_clients = {}


def _get_device():
    """Detect best available device for local inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_local_pipeline(model_id: str):
    """Load a HuggingFace model locally via transformers pipeline."""
    if model_id not in _local_pipelines:
        from transformers import pipeline as hf_pipeline
        device = _get_device()
        print(f"Loading {model_id} locally on {device}...")
        _local_pipelines[model_id] = hf_pipeline(
            "text-generation",
            model=model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        print(f"Loaded {model_id}")
    return _local_pipelines[model_id]


def _get_hf_client(model_id: str):
    """Get or create a HuggingFace InferenceClient for the given model."""
    if model_id not in _hf_clients:
        from huggingface_hub import InferenceClient
        token = os.environ.get("HF_TOKEN")
        _hf_clients[model_id] = InferenceClient(model_id, token=token)
    return _hf_clients[model_id]


class LLM():
    def __init__(self, model: str, system_prompt: str):
        self.model = model
        self.system_prompt = system_prompt
        self.backend = None  # "openai", "groq", "hf_api", "local"

        if model.startswith(OPENAI_PREFIXES):
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(f"OPENAI_API_KEY required for model {model}")
            self.client = openai.OpenAI(api_key=api_key)
            self.backend = "openai"
        elif model in GROQ_MODELS:
            groq_key = os.environ.get("GROQ_API_KEY")
            if not groq_key:
                raise RuntimeError(f"GROQ_API_KEY required for model {model}")
            self.client = openai.OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=groq_key
            )
            self.backend = "groq"
        elif os.environ.get("HF_TOKEN"):
            # Use HuggingFace Inference API for non-OpenAI/non-Groq models
            self.hf_client = _get_hf_client(model)
            self.backend = "hf_api"
        else:
            # Fall back to local HuggingFace inference
            self.pipeline = _get_local_pipeline(model)
            self.backend = "local"

    def get_response(self, user_prompt: str) -> Optional[str]:
        if self.backend == "local":
            return self._get_local_response(user_prompt)
        elif self.backend == "hf_api":
            return self._get_hf_api_response(user_prompt)
        return self._get_api_response(user_prompt)

    def _get_api_response(self, user_prompt: str) -> Optional[str]:
        messages: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_prompt}]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages  # type: ignore[arg-type]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error ({self.backend}): {e}")
            return None

    def _get_hf_api_response(self, user_prompt: str) -> Optional[str]:
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_prompt}]
        try:
            response = self.hf_client.chat_completion(
                messages=messages,
                max_tokens=512,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error (hf_api): {e}")
            return None

    def _get_local_response(self, user_prompt: str) -> Optional[str]:
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_prompt}]
        try:
            outputs = self.pipeline(
                messages,
                max_new_tokens=512,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
            )
            return outputs[0]["generated_text"][-1]["content"]
        except Exception as e:
            print(f"Error (local): {e}")
            return None
