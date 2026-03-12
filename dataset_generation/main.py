import os
from typing import Tuple
import math
import pandas as pd
import time
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

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


# Paper uses Qwen/Qwen2.5-14B-Instruct; override via MODEL_NAME env var for local testing
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
#Hyperparameters
NUM_ATTEMPTS = 4
ENTAILMENT_THRESH = 0.75
GENERATOR_TEMPERATURE = 0.8
DISCRIMINATOR_TEMPERATURE = 0.3
TOP_P_THRESHOLD = 0.95

def generate_hallucinated_answer(question: str, knowledge: str, ground_truth: str) -> Tuple[str, int]:
    """
    Generate a plausible but incorrect (hallucinated) answer for a question using an LLM.

    The function attempts multiple generations, evaluates each hallucination using
    discriminator models and an entailment check against the ground truth, and
    optionally improves weak outputs using TextGrad optimization. If a response
    successfully fools at least one evaluator and has low entailment with the
    ground truth, it is returned with a difficulty label. If all attempts fail,
    the hallucination most semantically similar to the ground truth is returned
    with "Easy" difficulty.

    Args:
        question (str): Input question.
        knowledge (str): Contextual knowledge provided to the generator.
        ground_truth (str): Correct answer used for evaluation.

    Returns:
        Tuple[str, str, str, str]: (hallucinated_answer, justification, hallucination_type, difficulty)
    """

    hallucinations = []
    #Initialize LLM Model
    llm_model = LLM(model=MODEL_NAME, system_prompt=hallucination_prompt)

    #NUM_ATTEMPTS attempts are provided for an high quality hallucination
    for attempt in range(NUM_ATTEMPTS):
        print(f"Attempt {attempt + 1}")
        prompt = f"#Question#: {question}\n\n#Knowledge#: {knowledge}\n\n#Ground truth answer#: {ground_truth}\n\n#Hallucinated Answer#:"
        
        #Get the hallucinated response from the LLM
        print(f"Getting initial response attempt {attempt + 1}")
        hallu_response = llm_model.get_response(prompt, temp = GENERATOR_TEMPERATURE, top_p=TOP_P_THRESHOLD)

        if hallu_response is None or hallu_response.strip() == "":
            print(f"Attempt {attempt + 1}: LLM returned None, skipping")
            continue
        
        hallu, justification, type = find_hallu_parts(hallu_response)
        print(f"Evaluating initial response attempt {attempt + 1}")
        #Evaluate the quality of the response
        results = evaluate_response_quality(hallu, justification, ground_truth, question, temp = DISCRIMINATOR_TEMPERATURE)
        #Calculate entailment score to ensure that the responses aren't implications of each other.
        entailment_score = get_entailment(hallu_response, ground_truth)
        #Response is high quality if any LLM was fooled and if the entailment score was low enough.
        if any(results) and entailment_score < ENTAILMENT_THRESH:
            difficulty = find_difficulty(results)
            return hallu, justification, type, difficulty
        
        print(f"Optimizing attempt {attempt + 1}")
        #If the response is low quality, we choose to optimize it using textgrad.
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

        #Check the quality of the optimized response and calculate the entailment
        results = evaluate_response_quality(hallu, justification, ground_truth, question, temp = DISCRIMINATOR_TEMPERATURE)
        entailment_score = get_entailment(hallu, ground_truth)
        if any(results) and entailment_score < ENTAILMENT_THRESH:
            difficulty = find_difficulty(results)
            return hallu, justification, type, difficulty
        
        #If the optimized response is not good enough, we store the attempt and repeat
        hallucinations.append(hallu_response)

    #If the model fails, we pick the closest response semantically to the truth, and and return Easy difficulty
    min_hallu = get_min_similarity(hallucinations, ground_truth)
    hallu, justification, type = find_hallu_parts(min_hallu)
    return hallu, justification, type, "Easy"

def main():
    #Download file from huggingFace onto local machine.
    download_file_if_not_exists()

    df_artificial = pd.read_csv("dataset_generation/medqa_data_artificial.csv")
    original_df_artificial = pd.read_csv("dataset_generation/medhallu_dataset_artificial.csv")
    print("Loaded data from local machine")

    #Since this code takes a while to run, we load any previous results before starting.
    if os.path.exists("checkpoint.csv"):
        existing_df = pd.read_csv("checkpoint.csv")
        results = existing_df.to_dict('records')
        completed_questions = set(existing_df["Question"].tolist())
        print(f"Resuming: {len(results)} samples already completed")
    else:
        results = []
        completed_questions = set()
    accurate = 0

    #We only generate 500 samples due to computational requirements.
    for i in tqdm(range(500)):
        question = df_artificial['question'][i]
        # Skip already completed
        if question in completed_questions:
            continue

        knowledge = df_artificial['context'][i]
        ground_truth = df_artificial['long_answer'][i]

        #Generate the hallucinated answer for this question
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

        #Check whether difficulty labels match with the original dataset
        if (difficulty.lower() == original_df_artificial["Difficulty Level"][i].lower()):
            accurate += 1
        
        print(f"Current accuracy = {accurate}/{(i + 1)} = {accurate/(i + 1)}")
        
        #Store the results temporailily every question
        pd.DataFrame(results).to_csv("checkpoint.csv", index=False)
        #To avoid any rate limits if on lower plans.
        time.sleep(2) 

    #Generate the dataset and save it.
    df_out = pd.DataFrame(results)
    df_out.to_csv("medqa_hallucinated.csv", index=False)
    print(f"Saved {len(df_out)} rows to medqa_hallucinated.csv")

if __name__ == "__main__":
    main()
