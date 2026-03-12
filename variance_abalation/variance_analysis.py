import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
from typing import Tuple
from tqdm import tqdm
import os
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

# Now import from parent directly
from utils import (
    evaluate_response_quality,
    find_difficulty,
    get_entailment,
    get_min_similarity,
    optimize_with_textgrad,
    find_hallu_parts
)
from llm import LLM
from prompt import hallucination_prompt


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

        # #Check the quality of the optimized response and calculate the entailment
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
    df = pd.read_csv("dataset_generation/medqa_hallucinated.csv")
    original_df = pd.read_csv("dataset_generation/medhallu_dataset_artificial.csv")
    for i in tqdm(range(20)):
        results = []
        accurate = 0

        filename = Path(__file__).parent / f"runs/variance_run_{i + 1}.csv"

        if filename.exists():
            continue

        for j in tqdm(range(20)):
            question = df['Question'][j]
            knowledge = df['Knowledge'][j]
            ground_truth = df['Ground Truth'][j]

            #Generate the hallucinated answer for this question
            hallu, justification, type, difficulty = generate_hallucinated_answer(question, knowledge, ground_truth)
            print(f"Difficulty: {difficulty}")
            results.append({
                'Question': question,
                'Knowledge': knowledge,
                'Ground Truth': ground_truth,
                'Difficulty Level': difficulty,
                'Hallucinated Answer': hallu,
                'Category of Hallucination': type
            })
             #Check whether difficulty labels match with the original dataset
            if (difficulty.lower() == original_df["Difficulty Level"][i].lower()):
                accurate += 1
            
            print(f"Current accuracy = {accurate}/{(j + 1)} = {accurate/(j + 1)}")
        
        pd.DataFrame(results).to_csv(f"runs/variance_run_{i + 1}.csv", index=False)
        print("Saved")

def eval():
    original_df = pd.read_csv("dataset_generation/medhallu_dataset_artificial.csv")
    
    all_runs = []
    accuracies = []

    for i in range(20):
        filename = Path(__file__).parent / f"runs/variance_run_{i + 1}.csv"
        df = pd.read_csv(filename)
        df["run"] = i + 1
        all_runs.append(df)
        accuracies.append(get_accuracy(df, original_df))

    combined = pd.concat(all_runs, ignore_index=True)

    # ── 1. Accuracy variance across runs ──────────────────────────────────────
    print("=== Accuracy Across Runs ===")
    accuracy_df = pd.DataFrame({
        "run": range(1, 21),
        "accuracy": accuracies
    })
    accuracy_df.loc[len(accuracy_df)] = ["summary_mean", np.mean(accuracies)]
    accuracy_df.loc[len(accuracy_df)] = ["summary_std", np.std(accuracies)]
    accuracy_df.loc[len(accuracy_df)] = ["summary_min", np.min(accuracies)]
    accuracy_df.loc[len(accuracy_df)] = ["summary_max", np.max(accuracies)]
    accuracy_df.loc[len(accuracy_df)] = ["summary_cv", np.std(accuracies) / np.mean(accuracies)]
    print(accuracy_df.to_string())
    accuracy_df.to_csv("variance_abalation/variance_accuracy.csv", index=False)

    # ── 2. Difficulty distribution per run ────────────────────────────────────
    print("\n=== Difficulty Distribution Per Run ===")
    diff_dist = combined.groupby("run")["Difficulty Level"].value_counts(normalize=True).unstack(fill_value=0)
    diff_std = diff_dist.std().rename("std_across_runs")
    print(diff_dist.to_string())
    print(diff_std.to_string())
    diff_dist.to_csv("variance_abalation/variance_difficulty_distribution.csv")
    diff_std.to_csv("variance_abalation/variance_difficulty_std.csv")

    # ── 3. Category distribution per run ─────────────────────────────────────
    print("\n=== Category Distribution Per Run ===")
    cat_dist = combined.groupby("run")["Category of Hallucination"].value_counts(normalize=True).unstack(fill_value=0)
    cat_std = cat_dist.std().rename("std_across_runs")
    print(cat_dist.to_string())
    print(cat_std.to_string())
    cat_dist.to_csv("variance_abalation/variance_category_distribution.csv")
    cat_std.to_csv("variance_abalation/variance_category_std.csv")

    # ── 4. Per-question difficulty stability ──────────────────────────────────
    print("\n=== Per-Question Difficulty Stability ===")
    question_stability = combined.groupby("Question")["Difficulty Level"].apply(
        lambda x: x.value_counts(normalize=True).max()
    ).reset_index()
    question_stability.columns = ["Question", "stability_score"]
    print(question_stability.describe().to_string())
    question_stability.to_csv("variance_abalation/variance_question_stability.csv", index=False)

    # ── 5. Plots ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(range(1, 21), accuracies, marker='o')
    axes[0, 0].axhline(np.mean(accuracies), color='r', linestyle='--', label=f'Mean={np.mean(accuracies):.2f}')
    axes[0, 0].set_title("Difficulty Match Accuracy Across Runs")
    axes[0, 0].set_xlabel("Run")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend()

    diff_dist.plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title("Difficulty Distribution Per Run")
    axes[0, 1].set_xlabel("Run")
    axes[0, 1].set_ylabel("Proportion")
    axes[0, 1].tick_params(axis='x', rotation=45)

    cat_dist.plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title("Category Distribution Per Run")
    axes[1, 0].set_xlabel("Run")
    axes[1, 0].set_ylabel("Proportion")
    axes[1, 0].tick_params(axis='x', rotation=45)

    axes[1, 1].hist(question_stability["stability_score"], bins=10, edgecolor='black')
    axes[1, 1].set_title("Per-Question Difficulty Stability")
    axes[1, 1].set_xlabel("Stability Score (1.0 = always same label)")
    axes[1, 1].set_ylabel("Number of Questions")

    plt.tight_layout()
    plt.savefig("variance_abalation/variance_analysis.png", dpi=150)
    plt.show()

    print("\n=== Saved Files ===")

def get_accuracy(df, df_actual):
    accurate = 0
    for i in range(len(df)):
        diff_1 = df["Difficulty Level"][i]
        diff_2 = df_actual["Difficulty Level"][i]

        if diff_1.lower() == diff_2.lower():
            accurate += 1
    return accurate/len(df)

if __name__ == "__main__":
    main()
    eval()
