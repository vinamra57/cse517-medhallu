import pandas as pd
from datasets import load_dataset
from llm import LLM
from typing import List, Tuple
import os

#Hyperparams to tune
BATCH_SIZE = 9000
#Different model used for now
MODEL_NAME = "openai/gpt-oss-20b"
NUM_ATTEMPTS = 4
DIFFICULTIES = {1: "Easy", 2: "Medium", 3: "Hard"}
ENTAILMENT_THRESH = 0.5

#Can update the inputs and outputs of these methods
def evaluate_response_quality(hallu_response: str, ground_truth: str) -> List[bool]:
    return [True]

def optimize_with_textgrad():
    pass

def find_difficulty(results):
    pass

def get_entailment(hallu_response: str, ground_truth: str):
    return 0

def get_min_similarity(hallucinations):
    pass

def generate_hallucinated_answer(question: str, knowledge: str, ground_truth: str, llm_model: LLM) -> Tuple[str, int]:
    hallucinations = []
    for attempt in range(NUM_ATTEMPTS):
        print(f"Attempt {attempt + 1}")
        prompt = f"#Question#: {question}\n\n#Knowledge#: {knowledge}\n\n#Ground truth answer#: {ground_truth}\n\n#Hallucinated Answer#:"
        
        #For now, we only get the response and print it. Code for quality control is required.
        hallu_response = llm_model.get_hallucinated_response(prompt)

        results = evaluate_response_quality(hallu_response, ground_truth)
        #Must calculate entailment
        entailment_score = get_entailment(hallu_response, ground_truth)
        if any(results) and entailment_score < ENTAILMENT_THRESH:
            difficulty = find_difficulty(results)
            return hallu_response, difficulty
        
        optimized_hallu_response = optimize_with_textgrad()
        
        results = evaluate_response_quality(optimized_hallu_response, ground_truth)
        entailment_score = get_entailment(hallu_response, ground_truth)
        if any(results) and entailment_score < ENTAILMENT_THRESH:
            difficulty = find_difficulty(results)
            return optimized_hallu_response, difficulty
        
        hallucinations.append(optimized_hallu_response)

    min_hallu = get_min_similarity(hallucinations)
    return min_hallu, 1

def main():
    #Load the data locally, or if it isn't there, from hugging face
    if os.path.exists("medqa_data_artificial.csv") and os.path.exists("medqa_data_labeled.csv"):
        df_artificial = pd.read_csv("medqa_data_artificial.csv")
        df_labeled = pd.read_csv("medqa_data_labeled.csv")
        print("Loaded data from local machine")
    else:
        dataset_artificial = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split=f"train[:{BATCH_SIZE}]")
        df_artificial = dataset_artificial.to_pandas()

        dataset_labeled = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        df_labeled = dataset_labeled.to_pandas()

        print("Loaded data from web")

        df_artificial.to_csv("medqa_data_artificial.csv", index=False)
        df_labeled.to_csv("medqa_data_labeled.csv", index=False)
        print("Saved data locally")

    #Only one response for now.
    question = df_artificial['question'][0]
    knowledge = df_artificial['context'][0]
    ground_truth = df_artificial['long_answer'][0]
    model = LLM(MODEL_NAME)

    result = generate_hallucinated_answer(question, knowledge, ground_truth, model)
    print(f"Question: {question}")
    print(f"True Answer: {ground_truth}")
    print(f"Hallucinated Answer: {result}")

if __name__ == "__main__":
    main()
