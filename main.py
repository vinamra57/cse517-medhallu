import os
from typing import Tuple
import math
import pandas as pd
import time
from llm import LLM
from prompt import hallucination_prompt
from utils import (
    download_file_if_not_exists,
    evaluate_response_quality,
    find_difficulty,
    get_entailment,
    get_min_similarity,
    optimize_with_textgrad,
    find_hallu_parts, 
    parse_knowledge
)
from tqdm import tqdm

#Hyperparams to tune

# Paper uses Qwen/Qwen2.5-14B-Instruct; override via MODEL_NAME env var for local testing
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct")
NUM_ATTEMPTS = 4
ENTAILMENT_THRESH = 0.75
GENERATOR_TEMPERATURE = 0.8
DISCRIMINATOR_TEMPERATURE = 0.3
TOP_P_THRESHOLD = 0.95

def generate_hallucinated_answer(question: str, knowledge: str, ground_truth: str) -> Tuple[str, int]:
    hallucinations = []
    llm_model = LLM(model=MODEL_NAME, system_prompt=hallucination_prompt)

    for attempt in range(NUM_ATTEMPTS):
        print(f"Attempt {attempt + 1}")
        prompt = f"#Question#: {question}\n\n#Knowledge#: {knowledge}\n\n#Ground truth answer#: {ground_truth}\n\n#Hallucinated Answer#:"
        
        #For now, we only get the response and print it. Code for quality control is required.
        print(f"Getting initial response attempt {attempt + 1}")
        hallu_response = llm_model.get_response(prompt, temp = GENERATOR_TEMPERATURE, top_p=TOP_P_THRESHOLD)

        if hallu_response is None or hallu_response.strip() == "":
            print(f"Attempt {attempt + 1}: LLM returned None, skipping")
            continue
        
        hallu, justification, type = find_hallu_parts(hallu_response)
        print(f"Evaluating initial response attempt {attempt + 1}")
        results = evaluate_response_quality(hallu, justification, ground_truth, question, temp = DISCRIMINATOR_TEMPERATURE)
        #Must calculate entailment
        entailment_score = get_entailment(hallu_response, ground_truth)
        if any(results) and entailment_score < ENTAILMENT_THRESH:
            difficulty = find_difficulty(results)
            return hallu, justification, type, difficulty
        
        print(f"Optimizing attempt {attempt + 1}")
        try:
            optimized_hallu_response = optimize_with_textgrad(hallu_response, ground_truth, question, knowledge)
        except Exception as e:
            print(f"TextGrad failed: {e}, skipping optimization")
            optimized_hallu_response = None
        
        if optimized_hallu_response is None or optimized_hallu_response.strip() == "":
            print(f"Attempt {attempt + 1}: LLM returned None, skipping")
            continue
        
        hallu, justification, type = find_hallu_parts(optimized_hallu_response)
        print(f"Evaluating optimized response attempt {attempt + 1}")
        results = evaluate_response_quality(hallu, justification, ground_truth, question, temp = DISCRIMINATOR_TEMPERATURE)
        entailment_score = get_entailment(hallu, ground_truth)
        if any(results) and entailment_score < ENTAILMENT_THRESH:
            difficulty = find_difficulty(results)
            return hallu, justification, type, difficulty
        
        hallucinations.append(hallu_response)

    min_hallu = get_min_similarity(hallucinations, ground_truth)
    hallu, justification, type = find_hallu_parts(min_hallu)
    return hallu, justification, type, "Easy"

def main():
    download_file_if_not_exists()

    df_artificial = pd.read_csv("medqa_data_artificial.csv")
    df_labeled = pd.read_csv("medqa_data_labeled.csv")
    original_df_artificial = pd.read_csv("medhallu_dataset_artificial.csv")
    original_df_labeled = pd.read_csv("medhallu_dataset_labeled.csv")
    print("Loaded data from local machine")

    if os.path.exists("checkpoint.csv"):
        existing_df = pd.read_csv("checkpoint.csv")
        results = existing_df.to_dict('records')
        completed_questions = set(existing_df["Question"].tolist())
        print(f"Resuming: {len(results)} samples already completed")
    else:
        results = []
        completed_questions = set()
    accurate = 0

    for i in tqdm(range(500)):
        question = df_artificial['question'][i]
        # Skip already completed
        if question in completed_questions:
            continue

        knowledge = df_artificial['context'][i]
        ground_truth = df_artificial['long_answer'][i]

        hallu, justification, type, difficulty = generate_hallucinated_answer(question, knowledge, ground_truth)
        print(f"Difficulty: {difficulty}")

        results.append({
            'Question': question,
            'Knowledge': parse_knowledge(knowledge),
            'Ground Truth': ground_truth,
            'Difficulty Level': difficulty,
            'Hallucinated Answer': hallu,
            'Category of Hallucination': type
        })

        if (difficulty == original_df_artificial["Difficulty Level"][i]):
            accurate += 1
        
        print(f"Current accuracy = {accurate}/{(i + 1)} = {accurate/(i + 1)}")
        pd.DataFrame(results).to_csv("checkpoint.csv", index=False)
        #To avoid rate limits
        time.sleep(10) 

    df_out = pd.DataFrame(results)
    df_out.to_csv("medqa_hallucinated.csv", index=False)
    print(f"Saved {len(df_out)} rows to medqa_hallucinated.csv")

if __name__ == "__main__":
    main()
