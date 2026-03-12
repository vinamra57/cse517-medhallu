import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util as st_util
from rouge_score import rouge_scorer
from tqdm import tqdm
import torch
import os
from scipy import stats

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from llm import LLM
from prompt import hallucination_prompt
from utils import find_hallu_parts, evaluate_response_quality, _compute_nli_entailment_score

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
NUM_SAMPLES = 15
NLI_THRESHOLD = 0.75
_embed_model = None

# ── Helpers ───────────────────────────────────────────────────────────────────

def difficulty_label(avg_fooled: float) -> str:
    """Convert average number of LLMs fooled to Easy/Medium/Hard."""
    if avg_fooled <= 1:
        return "Easy"
    elif avg_fooled > 1 and avg_fooled <= 2:
        return "Medium"
    else:
        return "Hard"

def _load_embedding_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model

# ── Generation ────────────────────────────────────────────────────────────────

def generate_hallucinations(question, knowledge, ground_truth):
    """Generate NUM_SAMPLES hallucinations and track how many LLMs each fools."""
    hallucinations = []
    difficulties = []
    llm_model = LLM(model=MODEL_NAME, system_prompt=hallucination_prompt)

    for _ in tqdm(range(NUM_SAMPLES), desc="Generating hallucinations", leave=False):
        prompt = f"#Question#: {question}\n\n#Knowledge#: {knowledge}\n\n#Ground truth answer#: {ground_truth}\n\n#Hallucinated Answer#:"
        hallu_response = llm_model.get_response(prompt, temp=0.8, top_p=0.95)
        hallu, justification, hallu_type = find_hallu_parts(hallu_response)

        if not hallu or not isinstance(hallu, str):
            continue

        results = evaluate_response_quality(hallu, justification, ground_truth, question, 0.3)
        difficulties.append(sum(results))  # num LLMs fooled for this hallucination
        hallucinations.append(hallu)

    return hallucinations, difficulties

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(hallucination: str, ground_truth: str):
    model = _load_embedding_model()

    #Embed the hallucination and truth
    hallu_emb = model.encode(hallucination, convert_to_tensor=True)
    truth_emb = model.encode(ground_truth, convert_to_tensor=True)

    #Calculate cosine_sim, distance and ROUGE score
    cosine_sim = float(st_util.cos_sim(hallu_emb, truth_emb).item())
    euclidean_dist = float(torch.dist(hallu_emb, truth_emb).item())

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = scorer.score(ground_truth, hallucination)['rougeL'].fmeasure

    return cosine_sim, euclidean_dist, rouge_l

# ── Clustering ────────────────────────────────────────────────────────────────

def bidirectional_entailment(text1: str, text2: str, threshold: float = NLI_THRESHOLD) -> bool:
    """Returns True if text1 and text2 mutually entail each other."""
    return (
        _compute_nli_entailment_score(text1, text2) >= threshold and
        _compute_nli_entailment_score(text2, text1) >= threshold
    )

def cluster_hallucinations(hallucinations: list, ground_truth: str, threshold: float = NLI_THRESHOLD):
    """
    Cluster hallucinations + ground truth using bidirectional entailment.
    Returns clusters (list of index lists) and texts (hallucinations + ground truth).
    Ground truth is always the last element in texts.
    """
    texts = hallucinations + [ground_truth]
    n = len(texts)

    clusters = []
    assigned = set()

    #Clusters are created greedily here. Two elements can be in a same cluster only if their entailment is
    #the threshold
    for i in range(n):
        if i in assigned:
            continue
        cluster = [i]
        assigned.add(i)
        for j in range(i + 1, n):
            if j in assigned:
                continue
            if bidirectional_entailment(texts[i], texts[j], threshold):
                cluster.append(j)
                assigned.add(j)
        clusters.append(cluster)

    return clusters, texts

def compute_cluster_metrics(clusters: list, texts: list, difficulties: list, ground_truth: str):
    """
    For each cluster:
      - Compute centroid embedding and its distance to ground truth
      - Compute average difficulty of hallucinations within the cluster
      - ground truth is texts[-1], difficulties aligns with texts[:-1]
    """
    model = _load_embedding_model()
    truth_emb = model.encode(ground_truth, convert_to_tensor=True)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    ground_truth_idx = len(texts) - 1  # ground truth is always last

    cluster_metrics = []
    for cluster in clusters:
        cluster_texts = [texts[i] for i in cluster]

        # ── Difficulty: average of hallucinations in cluster (exclude ground truth) ──
        cluster_difficulties = [
            difficulties[i] for i in cluster
            if i != ground_truth_idx and i < len(difficulties)
        ]
        avg_cluster_difficulty = np.mean(cluster_difficulties) if cluster_difficulties else 0.0
        difficulty = difficulty_label(avg_cluster_difficulty)

        # ── Centroid distance to ground truth ──
        embeddings = model.encode(cluster_texts, convert_to_tensor=True)
        centroid = embeddings.mean(dim=0)

        cosine_sim = float(st_util.cos_sim(centroid, truth_emb).item())
        euclidean_dist = float(torch.dist(centroid, truth_emb).item())

        # ── ROUGE-L: average across cluster members ──
        rouge_l = np.mean([
            scorer.score(ground_truth, t)['rougeL'].fmeasure
            for t in cluster_texts
        ])

        cluster_metrics.append({
            "cluster_size": len(cluster),
            "avg_llms_fooled": avg_cluster_difficulty,
            "difficulty": difficulty,
            "cosine_similarity": cosine_sim,
            "euclidean_distance": euclidean_dist,
            "rouge_l": rouge_l
        })

    return cluster_metrics

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv("dataset_generation/medqa_hallucinated.csv")

    df_easy = df[df["Difficulty Level"] == "Easy"].sample(n=15, random_state=42)
    df_medium = df[df["Difficulty Level"] == "Medium"].sample(n=15, random_state=42)
    df_hard = df[df["Difficulty Level"] == "Hard"].sample(n=15, random_state=42)

    df_sample = pd.concat([df_easy, df_medium, df_hard]).reset_index(drop=True)
    total_clusters = 0
    results = []

    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        truth = row["Ground Truth"]
        knowledge = row["Knowledge"]
        question = row["Question"]

        # ── Generate 15 hallucinations + per-hallucination difficulty ──
        hallucinations, difficulties = generate_hallucinations(
            question, knowledge=knowledge, ground_truth=truth
        )

        if not hallucinations:
            print(f"Skipping question — no valid hallucinations generated.")
            continue

        # ── Cluster ──
        clusters, texts = cluster_hallucinations(hallucinations, truth)

        # ── Compute cluster-level metrics ──
        cluster_metrics = compute_cluster_metrics(clusters, texts, difficulties, truth)
        total_clusters += len(cluster_metrics)
        # ── One row per cluster ──
        for cm in cluster_metrics:
            results.append({
                "question": question,
                "difficulty": cm["difficulty"],
                "avg_llms_fooled": cm["avg_llms_fooled"],
                "cluster_size": cm["cluster_size"],
                "cosine_similarity": cm["cosine_similarity"],
                "euclidean_distance": cm["euclidean_distance"],
                "rouge_l": cm["rouge_l"]
            })

    results_df = pd.DataFrame(results)

    # ── Summary by difficulty ──
    summary = results_df.groupby("difficulty")[
        ["cosine_similarity", "euclidean_distance", "rouge_l", "cluster_size"]
    ].mean()

    print("\n=== Cluster Proximity Results by Difficulty ===")
    print(summary)
    print(f"Total clusters: {total_clusters}")

    # ── Statistical significance: Hard vs Easy ──
    hard = results_df[results_df["difficulty"] == "Hard"]
    easy = results_df[results_df["difficulty"] == "Easy"]

    print("\n=== Statistical Significance (Hard vs Easy) ===")
    for metric in ["cosine_similarity", "euclidean_distance", "rouge_l"]:
        if len(hard) == 0 or len(easy) == 0:
            print(f"{metric}: not enough data in one group")
            continue
        stat, p = stats.mannwhitneyu(hard[metric], easy[metric], alternative="two-sided")
        print(f"{metric}: p={p:.4f} {'✅ significant' if p < 0.05 else '❌ not significant'}")

    # ── Save ──
    results_df.to_csv("cluster_proximity_results.csv", index=False)
    summary.to_csv("cluster_proximity_summary.csv", index=False)
    print("\nSaved to cluster_proximity_results.csv and cluster_proximity_summary.csv")


if __name__ == "__main__":
    main()