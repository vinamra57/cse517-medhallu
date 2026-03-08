import pandas as pd
from prompt import eval_prompt
from llm import LLM
import random
from sklearn.metrics import f1_score, precision_score, accuracy_score
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

models_to_test = [
    "google/gemma-2-2b-it",
    "TsinghuaC3I/Llama-3.1-8B-UltraMedical",
    "BioMistral/BioMistral-7B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "gpt-4o-mini",
]

MODEL_SHORT_NAMES = {
    "TsinghuaC3I/Llama-3.1-8B-UltraMedical": "UltraMedical-8B",
    "BioMistral/BioMistral-7B": "BioMistral-7B",
    "google/gemma-2-2b-it": "Gemma-2-2B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "gpt-4o-mini": "GPT-4o Mini",
}

CATEGORIES = [
    'Misinterpretation of Question',
    'Incomplete Information', 
    'Mechanism and Pathway Misattribution',
    'Methodological and Evidence Fabrication',
    'Unknown'
]
CATEGORY_LABELS = [
    'Misinterpretation\nof Question',
    'Incomplete\nInformation',
    'Mechanism &\nPathway Misattribution',
    'Methodological &\nEvidence Fabrication',
    'Unknown'
]

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

def extract_response(response):
    if "0" in response:
        return 0
    elif "1" in response: 
        return 1
    else:
        return 2
    

def compute(preds, labels):
    return {
        "f1": f1_score(labels, preds, average="binary", zero_division=0),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "accuracy": accuracy_score(labels, preds),
    }

def plot_accuracy_by_category(all_results: list, output_path: str = "accuracy_by_category.png"):
    models = [r["model"] for r in all_results]
    short_names = [MODEL_SHORT_NAMES.get(m, m) for m in models]
    n_models = len(models)
    n_cats = len(CATEGORIES)

    bar_width = 0.13
    x = np.arange(n_cats)
    colors = plt.cm.tab10(np.linspace(0, 0.6, n_models))

    fig, ax = plt.subplots(figsize=(14, 6.5))
    fig.patch.set_facecolor("#F7F8FA")
    ax.set_facecolor("#F7F8FA")

    for i, (row, name, color) in enumerate(zip(all_results, short_names, colors)):
        offset = (i - n_models / 2 + 0.5) * bar_width
        values = [row["default"].get(f"accuracy_{k}") or 0 for k in CATEGORIES]
        bars = ax.bar(x + offset, values, width=bar_width - 0.01, label=name, color=color, zorder=3)

        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=6.5, color="#444", fontweight="bold"
                )
    
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORY_LABELS, fontsize=10, color="#333")
    ax.set_ylabel("Accuracy", fontsize=11, color="#444", labelpad=10)
    ax.set_ylim(0, 1.15)
    ax.set_title("Model Accuracy by Hallucination Category (Default Settings)", fontsize=13, fontweight="bold", color="#222", pad=16)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#ddd", zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#ccc")
    ax.tick_params(axis="both", colors="#555")
    ax.legend(fontsize=9, framealpha=0.9, edgecolor="#ccc", loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Chart saved → {output_path}")
    plt.show()

def get_eval_metrics(llm_results, correct_answers, difficulties, categories):
    # Normalise unknown categories
    categories = [c if c in CATEGORIES else "Unknown" for c in categories]

    # Filter out 2s
    filtered = [(p, l, d, c) for p, l, d, c in zip(llm_results, correct_answers, difficulties, categories) if p != 2]
    preds, labels, diffs, cats = zip(*filtered) if filtered else ([], [], [], [])

    overall = compute(preds, labels)

    metrics = {
        "overall_f1": overall["f1"],
        "overall_precision": overall["precision"],
        "overall_accuracy": overall["accuracy"],
        "abstain_rate": llm_results.count(2) / len(llm_results),
    }

    for level in ["Easy", "Medium", "Hard"]:
        subset = [(p, l) for p, l, d, c in zip(preds, labels, diffs, cats) if d == level]
        if subset:
            p, l = zip(*subset)
            metrics[f"f1_{level.lower()}"] = compute(p, l)["f1"]
            metrics[f"accuracy_{level.lower()}"] = compute(p, l)["accuracy"]
        else:
            metrics[f"f1_{level.lower()}"] = None
            metrics[f"accuracy_{level.lower()}"] = None

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
    for i in tqdm(range(len(df))):
        ground_truth = df["Ground Truth"][i]
        hallu = df["Hallucinated Answer"][i]
        context = df["Knowledge"][i]
        question = df["Question"][i]
        difficulty = df["Difficulty Level"][i]
        category = df['Category of Hallucination'][i]

        answers = [ground_truth, hallu]
        random_val = random.randint(0, 1)
        chosen_ans = answers[random_val]
        correct_answers.append(random_val)
        diffs.append(difficulty)
        categories.append(category)

        user_prompt = get_user_prompt(context, question, chosen_ans, use_context, use_not_sure)
        #gemma model doesnt have system prompt
        if "gemma" in llm.model or "BioMistral" in llm.model:
            response = llm.get_response(f"System Prompt: {system_prompt} \n\n User Prompt: {user_prompt}")
        else:
            response = llm.get_response(user_prompt)
        llm_results.append(extract_response(response))
    
    return get_eval_metrics(llm_results, correct_answers, diffs, categories)


def main():
    df = pd.read_csv("medqa_hallucinated.csv")
    print(len(df))
    all_results = []
    for model in models_to_test:
        print(f"Testing model: {model}")
        if "gemma" in model or "BioMistral" in model:
            llm = LLM(model = model, system_prompt = None)
        else:
            llm = LLM(model = model, system_prompt = eval_prompt)
        results_default = get_llm_results(llm, df, eval_prompt)
        results_context = get_llm_results(llm, df, eval_prompt, use_context = True)
        results_not_sure = get_llm_results(llm, df, eval_prompt, use_not_sure = True)
    
        all_results.append({
                "model": model,
                "default": results_default,
                "context": results_context,
                "not_sure": results_not_sure
            })
        
        pd.DataFrame(all_results).to_csv("checkpoint_results.csv", index=False)

    plot_accuracy_by_category(all_results)
    # Convert list of dicts to DataFrame and save as CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("eval_results.csv", index=False)
    print("Results saved to eval_results.csv")
        

if __name__ == "__main__":
    main()

