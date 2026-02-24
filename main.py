import pandas as pd

from llm import LLM
from prompt import *
from typing import Tuple
import os
from utils import *

#Hyperparams to tune

# Paper uses Qwen/Qwen2.5-14B-Instruct; override via MODEL_NAME env var for local testing
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct")
NUM_ATTEMPTS = 4
ENTAILMENT_THRESH = 0.75

def generate_hallucinated_answer(question: str, knowledge: str, ground_truth: str) -> Tuple[str, int]:
    hallucinations = []
    llm_model = LLM(model = MODEL_NAME, system_prompt= hallucination_prompt)

    for attempt in range(NUM_ATTEMPTS):
        print(f"Attempt {attempt + 1}")
        prompt = f"#Question#: {question}\n\n#Knowledge#: {knowledge}\n\n#Ground truth answer#: {ground_truth}\n\n#Hallucinated Answer#:"
        
        #For now, we only get the response and print it. Code for quality control is required.
        hallu_response = llm_model.get_response(prompt)

        if hallu_response is None:
            print(f"Attempt {attempt + 1}: LLM returned None, skipping")
            continue

        results = evaluate_response_quality(hallu_response, ground_truth, question)
        #Must calculate entailment
        entailment_score = get_entailment(hallu_response, ground_truth)
        if any(results) and entailment_score < ENTAILMENT_THRESH:
            difficulty = find_difficulty(results)
            return hallu_response, difficulty
        
        optimized_hallu_response = optimize_with_textgrad(hallu_response, ground_truth, question, knowledge)
        
        results = evaluate_response_quality(optimized_hallu_response, ground_truth, question)
        entailment_score = get_entailment(optimized_hallu_response, ground_truth)
        if any(results) and entailment_score < ENTAILMENT_THRESH:
            difficulty = find_difficulty(results)
            return optimized_hallu_response, difficulty
        
        hallucinations.append(optimized_hallu_response)

    min_hallu = get_min_similarity(hallucinations, ground_truth)
    return min_hallu, 1

def main():
    #Load the data locally, or if it isn't there, from hugging face
    download_file_if_not_exists()

    df_artificial = pd.read_csv("medqa_data_artificial.csv")
    df_labeled = pd.read_csv("medqa_data_labeled.csv")
    print("Loaded data from local machine")
    
    #Only one response for now.
    question = df_artificial['question'][4]
    knowledge = df_artificial['context'][4]
    ground_truth = df_artificial['long_answer'][4]

    result, difficulty = generate_hallucinated_answer(question, knowledge, ground_truth)
    print(f"Question: {question}")
    print(f"True Answer: {ground_truth}")
    print(f"Hallucinated Answer: {result}")
    print(f"Difficulty: {difficulty}")

if __name__ == "__main__":
    main()
