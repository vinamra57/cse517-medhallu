from typing import List
import os
from prompt import *
from llm import LLM
from datasets import load_dataset

BATCH_SIZE = 9000
MODELS = ["openai/gpt-oss-20b", "llama-3.1-8b-instant", "moonshotai/kimi-k2-instruct"]
DIFFICULTIES = {1: "Easy", 2: "Medium", 3: "Hard"}

#Can update the inputs and outputs of these methods
def evaluate_response_quality(hallu_response: str, ground_truth: str, question: str) -> List[bool]:
    responses = []
    answer, justification = find_hallu_parts(hallu_response)
    option_1 = ground_truth
    option_2 = answer
    for model in MODELS:
        llm = LLM(model=model, system_prompt=detection_prompt)
        user_prompt = f"Question: {question} Option 1: {option_1} Option 2: {option_2} + {justification}"
        response = llm.get_response(user_prompt)
        responses.append(False if response == "Option 1" else True)

    return responses

def optimize_with_textgrad():
    pass

def find_difficulty(results):
    num_llm_fooled = sum(results)
    #Difficulty score is the number of LLMs that were fooled here
    return DIFFICULTIES[num_llm_fooled]

def get_entailment(hallu_response: str, ground_truth: str):
    return 0

def get_min_similarity(hallucinations):
    pass

def find_hallu_parts(hallu_response):
    # Find Hallucinated Answer
    start_idx = hallu_response.find("#Hallucinated Answer#: ")
    if start_idx != -1:
        start_idx += len("#Hallucinated Answer#: ")
        end_idx = hallu_response.find("#Justification", start_idx)
        hallucinated_answer = hallu_response[start_idx:end_idx].strip() if end_idx != -1 else hallu_response[start_idx:].strip()
    else:
        hallucinated_answer = ""
    
    # Find Justification
    start_idx = hallu_response.find("#Justification of Hallucinated answer#: ")
    if start_idx != -1:
        start_idx += len("#Justification of Hallucinated answer#: ")
        justification = hallu_response[start_idx:].strip()
    else:
        justification = ""
    
    return hallucinated_answer, justification

def download_file_if_not_exists():
    if not os.path.exists("medqa_data_artificial.csv") or not os.path.exists("medqa_data_labeled.csv"):
        dataset_artificial = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split=f"train[:{BATCH_SIZE}]").to_pandas()
        dataset_labeled = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train").to_pandas()

        dataset_artificial.to_csv("medqa_data_artificial.csv", index=False)
        dataset_labeled.to_csv("medqa_data_labeled.csv", index=False)