import pandas as pd
import random
from sklearn.metrics import f1_score, precision_score, accuracy_score
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from prompt import eval_prompt
from llm import LLM

models_to_test = [
    "google/gemma-2-2b-it",
    "BioMistral/BioMistral-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "TsinghuaC3I/Llama-3.1-8B-UltraMedical",
    "meta-llama/Llama-3.1-8B-Instruct",
    "gpt-4o-mini",
]

CATEGORIES = [
    'Misinterpretation of Question',
    'Incomplete Information', 
    'Mechanism and Pathway Misattribution',
    'Methodological and Evidence Fabrication',
    'Unknown'
]

#Get the correct user prompt based on the settings provided.
def get_user_prompt(knowledge, question, answer, use_context, use_not_sure):
    context_block = f"World Knowledge: {knowledge}\n        " if use_context else ""
    options = "Answer '0' if factual, '1' if hallucinated" + (", or '2' if unsure." if use_not_sure else ".")
    not_sure_hint = "\n        If you are unsure, choose `2` instead of guessing." if use_not_sure else ""

    return f"""
        {context_block}Question: {question}
        Answer: {answer}

        Return just the answer. {options} Don't return anything else, don't be verbose.{not_sure_hint}
        Your Judgement:
        """

#Extract the correct response from the LLM.
def extract_response(response):
    if "0" in response:
        return 0
    elif "1" in response: 
        return 1
    else:
        return 2
    
#Get the accuracy measures needed.
def compute(preds, labels):
    return {
        "f1": f1_score(labels, preds, average="binary", zero_division=0),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "accuracy": accuracy_score(labels, preds),
    }

def get_eval_metrics(llm_results, correct_answers, difficulties, categories):
    # Normalise unknown categories
    categories = [c if c in CATEGORIES else "Unknown" for c in categories]

    # Filter out 2s
    filtered = [(p, l, d, c) for p, l, d, c in zip(llm_results, correct_answers, difficulties, categories) if p != 2]
    preds, labels, diffs, cats = zip(*filtered) if filtered else ([], [], [], [])

    #Get overall metrics
    overall = compute(preds, labels)

    metrics = {
        "overall_f1": overall["f1"],
        "overall_precision": overall["precision"],
        "overall_accuracy": overall["accuracy"],
        "abstain_rate": llm_results.count(2) / len(llm_results),
    }

    #Metrics by difficulty
    for level in ["Easy", "Medium", "Hard"]:
        subset = [(p, l) for p, l, d, c in zip(preds, labels, diffs, cats) if d == level]
        if subset:
            p, l = zip(*subset)
            metrics[f"f1_{level.lower()}"] = compute(p, l)["f1"]
            metrics[f"accuracy_{level.lower()}"] = compute(p, l)["accuracy"]
        else:
            metrics[f"f1_{level.lower()}"] = None
            metrics[f"accuracy_{level.lower()}"] = None

    #Metrics by category
    for cat in CATEGORIES:
        subset = [(p, l) for p, l, d, c in zip(preds, labels, diffs, cats) if c == cat]
        if subset:
            p, l = zip(*subset)
            key = cat.lower().replace(" ", "_")
            metrics[f"accuracy_{key}"] = compute(p, l)["accuracy"]
            metrics[f"f1_{key}"] = compute(p, l)["f1"]
        else:
            key = cat.lower().replace(" ", "_")
            metrics[f"accuracy_{key}"] = None
            metrics[f"f1_{key}"] = None

    return metrics


def get_llm_results(llm, df, system_prompt, use_context = False, use_not_sure = False):
    llm_results = []
    correct_answers = []
    diffs = []
    categories = []
    #Iterate through the entire dataframe
    for i in tqdm(range(len(df))):
        ground_truth = df["Ground Truth"][i]
        hallu = df["Hallucinated Answer"][i]
        context = df["Knowledge"][i]
        question = df["Question"][i]
        difficulty = df["Difficulty Level"][i]
        category = df['Category of Hallucination'][i]

        #To avoid bias, we randomly choose one of the truth or the hallucinated answer as the question.
        answers = [ground_truth, hallu]
        random_val = random.randint(0, 1)
        chosen_ans = answers[random_val]

        correct_answers.append(random_val)
        diffs.append(difficulty)
        categories.append(category)

        user_prompt = get_user_prompt(context, question, chosen_ans, use_context, use_not_sure)
        #Handle models without system prompt allowed.
        if "gemma" in llm.model or "BioMistral" in llm.model:
            response = llm.get_response(f"System Prompt: {system_prompt} \n\n User Prompt: {user_prompt}")
        else:
            response = llm.get_response(user_prompt)
        llm_results.append(extract_response(response))
    
    #Evaluate the responses and return these.
    return get_eval_metrics(llm_results, correct_answers, diffs, categories)


def main():
    df = pd.read_csv("dataset_generation/medqa_hallucinated_no_opt.csv")
    all_results = []

    for model in models_to_test:
        print(f"Testing model: {model}")
        #Special rules for these models that dont allow system prompts.
        if "gemma" in model or "BioMistral" in model:
            llm = LLM(model = model, system_prompt = None)
        else:
            llm = LLM(model = model, system_prompt = eval_prompt)
        
        #For the classic performance test
        results_default = get_llm_results(llm, df, eval_prompt)
        #Performance test with context
        results_context = get_llm_results(llm, df, eval_prompt, use_context = True)
        #Performance test with not sure option given
        results_not_sure = get_llm_results(llm, df, eval_prompt, use_not_sure = True)
    
        all_results.append({
                "model": model,
                "default": results_default,
                "context": results_context,
                "not_sure": results_not_sure
            })
        
        #Save the checkpoint to avoid issues.
        pd.DataFrame(all_results).to_csv("checkpoint_results.csv", index=False)

    # Convert list of dicts to DataFrame and save as CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("eval_results_no_opt.csv", index=False)
    print("Results saved to eval_results.csv")
        

if __name__ == "__main__":
    main()

