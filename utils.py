from typing import List
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util as st_util
import textgrad as tg
from prompt import *
from llm import LLM
from datasets import load_dataset

BATCH_SIZE = 9000
MODELS = ["openai/gpt-oss-20b", "llama-3.1-8b-instant", "moonshotai/kimi-k2-instruct"]
DIFFICULTIES = {1: "Easy", 2: "Medium", 3: "Hard"}

# Lazy-loaded singletons for models
_nli_tokenizer = None
_nli_model = None
_embed_model = None

NLI_MODEL_NAME = "microsoft/deberta-large-mnli"
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def _load_nli_model():
    """Lazy singleton loader for the DeBERTa-large-MNLI model."""
    global _nli_tokenizer, _nli_model
    if _nli_tokenizer is None:
        _nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
        _nli_model.eval()
    return _nli_tokenizer, _nli_model


def _load_embedding_model():
    """Lazy singleton loader for the sentence embedding model."""
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _compute_nli_entailment_score(premise: str, hypothesis: str) -> float:
    """
    Compute P(entailment | premise, hypothesis) using DeBERTa-large-MNLI.
    DeBERTa label mapping: {0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT}
    """
    tokenizer, model = _load_nli_model()

    inputs = tokenizer(
        premise, hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=-1)[0]

    # DeBERTa-large-MNLI: index 2 = ENTAILMENT
    entailment_idx = 2
    id2label = model.config.id2label
    for idx, label in id2label.items():
        if label.upper() == "ENTAILMENT":
            entailment_idx = int(idx)
            break

    return probs[entailment_idx].item()


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


def optimize_with_textgrad(
    hallu_response: str,
    ground_truth: str,
    question: str,
    knowledge: str,
    max_iterations: int = 5
) -> str:
    """
    Use TextGrad to iteratively refine a hallucinated answer that failed
    quality control checks. Uses GPT-4o-mini as the backward engine.

    Args:
        hallu_response: The current hallucinated answer to optimize
        ground_truth: The ground truth answer (for context, not for copying)
        question: The medical question
        knowledge: The knowledge/context for the question
        max_iterations: Maximum TextGrad refinement iterations (default 5)

    Returns:
        The optimized hallucinated answer string
    """
    tg.set_backward_engine("gpt-4o-mini", override=True)

    hallu_var = tg.Variable(
        value=hallu_response,
        requires_grad=True,
        role_description=(
            "A hallucinated medical answer that should sound plausible "
            "and natural but be factually incorrect. It must fool medical "
            "expert LLMs into thinking it is correct. The output must use "
            "the format: #Hallucinated Answer#: <answer> and "
            "#Justification of Hallucinated answer#: <justification>"
        )
    )

    loss_fn = tg.TextLoss(
        f"You are evaluating a hallucinated medical answer for quality. "
        f"The answer is for this question: '{question}'. "
        f"The related knowledge is: '{knowledge}'. "
        f"Evaluate the hallucinated answer on these criteria:\n"
        f"1. Does it contain linguistic patterns that obviously signal "
        f"artificial or fabricated content? If so, identify them.\n"
        f"2. Are there structural elements that could be refined to enhance "
        f"naturalness and plausibility?\n"
        f"3. Does it sound like a real, authoritative medical answer that "
        f"could reasonably fool an expert?\n"
        f"4. Is it subtly incorrect rather than blatantly wrong?\n"
        f"5. Does the justification cite realistic-sounding references?\n"
        f"Be critical and concise. Identify specific weaknesses."
    )

    optimizer = tg.TGD(
        parameters=[hallu_var],
        constraints=[
            "The output MUST maintain the format: "
            "#Hallucinated Answer#: <answer> followed by "
            "#Justification of Hallucinated answer#: <justification>. "
            "The hallucinated answer should have about 5 more words than "
            "the ground truth. Never mention it is hallucinated or incorrect."
        ]
    )

    for _ in range(max_iterations):
        loss = loss_fn(hallu_var)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return hallu_var.value


def find_difficulty(results):
    num_llm_fooled = sum(results)
    #Difficulty score is the number of LLMs that were fooled here
    return DIFFICULTIES[num_llm_fooled]


def get_entailment(hallu_response: str, ground_truth: str) -> float:
    """
    Compute bidirectional entailment score between hallucinated response
    and ground truth using DeBERTa-large-MNLI.

    Formula: E(H, GT) = min(NLI(H -> GT), NLI(GT -> H))

    Returns:
        Float 0.0-1.0. Lower = more different (desired for hallucinations).
        The sample should be retained when score < threshold (0.75).
    """
    score_h_to_gt = _compute_nli_entailment_score(
        premise=hallu_response,
        hypothesis=ground_truth
    )

    score_gt_to_h = _compute_nli_entailment_score(
        premise=ground_truth,
        hypothesis=hallu_response
    )

    return min(score_h_to_gt, score_gt_to_h)


def get_min_similarity(hallucinations: List[str], ground_truth: str) -> str:
    """
    Select the hallucination candidate most semantically similar to ground truth.

    Per MedHallu: H* = argmax_{H in candidates} CosineSimilarity(embed(H), embed(GT))

    The most similar candidate is the most plausible fallback choice.
    """
    # Filter out None values
    valid = [h for h in hallucinations if h is not None]

    if not valid:
        return ""

    if len(valid) == 1:
        return valid[0]

    model = _load_embedding_model()

    gt_embedding = model.encode(ground_truth, convert_to_tensor=True)
    hallu_embeddings = model.encode(valid, convert_to_tensor=True)

    cosine_scores = st_util.cos_sim(hallu_embeddings, gt_embedding)  # shape: [N, 1]
    best_idx = cosine_scores.argmax().item()

    return valid[best_idx]


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
