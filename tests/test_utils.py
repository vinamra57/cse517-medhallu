"""Unit tests for utils.py with mocked models (no API keys or model downloads needed)."""
import pytest
from unittest.mock import patch, MagicMock
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Tests for find_hallu_parts ──

from utils import find_hallu_parts


def test_find_hallu_parts_normal():
    response = (
        "#Hallucinated Answer#: The sky is green. "
        "#Justification of Hallucinated answer#: According to a 2019 study."
    )
    answer, justification = find_hallu_parts(response)
    assert answer == "The sky is green."
    assert justification == "According to a 2019 study."


def test_find_hallu_parts_missing_answer():
    response = "#Justification of Hallucinated answer#: Some justification."
    answer, justification = find_hallu_parts(response)
    assert answer == ""
    assert justification == "Some justification."


def test_find_hallu_parts_missing_justification():
    response = "#Hallucinated Answer#: The sky is green."
    answer, justification = find_hallu_parts(response)
    assert answer == "The sky is green."
    assert justification == ""


def test_find_hallu_parts_empty_string():
    answer, justification = find_hallu_parts("")
    assert answer == ""
    assert justification == ""


# ── Tests for find_difficulty ──

from utils import find_difficulty


def test_difficulty_easy():
    assert find_difficulty([True, False, False]) == "Easy"


def test_difficulty_medium():
    assert find_difficulty([True, True, False]) == "Medium"


def test_difficulty_hard():
    assert find_difficulty([True, True, True]) == "Hard"


# ── Tests for evaluate_response_quality ──

from utils import evaluate_response_quality


@patch("utils.LLM")
def test_evaluate_all_fooled(mock_llm_class):
    mock_instance = MagicMock()
    mock_instance.get_response.return_value = "Option 2"
    mock_llm_class.return_value = mock_instance

    response = (
        "#Hallucinated Answer#: Wrong answer. "
        "#Justification of Hallucinated answer#: Because reasons."
    )
    results = evaluate_response_quality(response, "Correct answer", "What is X?")
    assert results == [True, True, True]


@patch("utils.LLM")
def test_evaluate_none_fooled(mock_llm_class):
    mock_instance = MagicMock()
    mock_instance.get_response.return_value = "Option 1"
    mock_llm_class.return_value = mock_instance

    response = (
        "#Hallucinated Answer#: Wrong answer. "
        "#Justification of Hallucinated answer#: Because reasons."
    )
    results = evaluate_response_quality(response, "Correct answer", "What is X?")
    assert results == [False, False, False]


@patch("utils.LLM")
def test_evaluate_partial_fooled(mock_llm_class):
    mock_instance = MagicMock()
    mock_instance.get_response.side_effect = ["Option 1", "Option 2", "Option 1"]
    mock_llm_class.return_value = mock_instance

    response = (
        "#Hallucinated Answer#: Wrong answer. "
        "#Justification of Hallucinated answer#: Because reasons."
    )
    results = evaluate_response_quality(response, "Correct answer", "What is X?")
    assert results == [False, True, False]


# ── Tests for get_entailment (mocked) ──


@patch("utils._load_nli_model")
def test_entailment_identical_texts(mock_load):
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[0, 1, 2]]), "attention_mask": torch.tensor([[1, 1, 1]])}
    mock_model = MagicMock()
    mock_model.config.id2label = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
    # High entailment score for identical texts
    mock_model.return_value.logits = torch.tensor([[-5.0, -5.0, 5.0]])
    mock_load.return_value = (mock_tokenizer, mock_model)

    from utils import get_entailment
    score = get_entailment("The cat sat on the mat.", "The cat sat on the mat.")
    assert score > 0.9


@patch("utils._load_nli_model")
def test_entailment_contradictory_texts(mock_load):
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[0, 1, 2]]), "attention_mask": torch.tensor([[1, 1, 1]])}
    mock_model = MagicMock()
    mock_model.config.id2label = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
    # Low entailment score for contradictory texts
    mock_model.return_value.logits = torch.tensor([[5.0, 0.0, -5.0]])
    mock_load.return_value = (mock_tokenizer, mock_model)

    from utils import get_entailment
    score = get_entailment("It is raining.", "It is sunny and dry.")
    assert score < 0.1


@patch("utils._load_nli_model")
def test_entailment_bidirectional_takes_min(mock_load):
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[0, 1, 2]]), "attention_mask": torch.tensor([[1, 1, 1]])}
    mock_model = MagicMock()
    mock_model.config.id2label = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}

    # First call: high entailment (0.95), second call: low entailment (0.05)
    mock_model.return_value.logits = torch.tensor([[0.0, 0.0, 3.0]])

    call_count = [0]
    original_return = mock_model.return_value

    def side_effect(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            mock_model.return_value.logits = torch.tensor([[-5.0, -5.0, 5.0]])  # high entailment
        else:
            mock_model.return_value.logits = torch.tensor([[5.0, 0.0, -5.0]])  # low entailment
        return original_return

    mock_model.side_effect = side_effect
    mock_load.return_value = (mock_tokenizer, mock_model)

    from utils import get_entailment
    score = get_entailment("Premise text", "Hypothesis text")
    # Should take the minimum of both directions
    assert score < 0.5


@patch("utils._load_nli_model")
def test_entailment_returns_float_in_range(mock_load):
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[0, 1, 2]]), "attention_mask": torch.tensor([[1, 1, 1]])}
    mock_model = MagicMock()
    mock_model.config.id2label = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
    mock_model.return_value.logits = torch.tensor([[1.0, 1.0, 1.0]])
    mock_load.return_value = (mock_tokenizer, mock_model)

    from utils import get_entailment
    score = get_entailment("Some text.", "Other text.")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


# ── Tests for get_min_similarity (mocked) ──


@patch("utils._load_embedding_model")
def test_min_similarity_single_item(mock_load):
    from utils import get_min_similarity
    result = get_min_similarity(["Only one candidate"], "ground truth")
    assert result == "Only one candidate"
    mock_load.assert_not_called()


def test_min_similarity_empty_list():
    from utils import get_min_similarity
    result = get_min_similarity([], "ground truth")
    assert result == ""


@patch("utils._load_embedding_model")
def test_min_similarity_selects_most_similar(mock_load):
    mock_model = MagicMock()
    # Simulate embeddings where candidate 1 is more similar to ground truth
    mock_model.encode.side_effect = [
        torch.tensor([1.0, 0.0, 0.0]),  # ground truth
        torch.tensor([[0.9, 0.1, 0.0], [0.0, 1.0, 0.0]]),  # candidates
    ]
    mock_load.return_value = mock_model

    from utils import get_min_similarity
    result = get_min_similarity(["similar candidate", "different candidate"], "ground truth")
    assert result == "similar candidate"


@patch("utils._load_embedding_model")
def test_min_similarity_filters_none(mock_load):
    from utils import get_min_similarity
    # Only one valid candidate after filtering None
    result = get_min_similarity([None, "valid candidate", None], "ground truth")
    assert result == "valid candidate"
    mock_load.assert_not_called()


@patch("utils._load_embedding_model")
def test_min_similarity_identical_candidate(mock_load):
    mock_model = MagicMock()
    mock_model.encode.side_effect = [
        torch.tensor([1.0, 0.0, 0.0]),  # ground truth
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),  # candidates (first is identical)
    ]
    mock_load.return_value = mock_model

    from utils import get_min_similarity
    result = get_min_similarity(["identical to gt", "very different"], "ground truth")
    assert result == "identical to gt"


# ── Tests for optimize_with_textgrad (mocked) ──


@patch("utils.tg")
def test_textgrad_returns_string(mock_tg):
    mock_var = MagicMock()
    mock_var.value = (
        "#Hallucinated Answer#: Optimized wrong answer. "
        "#Justification of Hallucinated answer#: Improved reasoning."
    )
    mock_tg.Variable.return_value = mock_var
    mock_tg.TextLoss.return_value = MagicMock(return_value=MagicMock())
    mock_tg.TGD.return_value = MagicMock()

    from utils import optimize_with_textgrad
    result = optimize_with_textgrad(
        "initial response", "ground truth", "question?", "knowledge", max_iterations=1
    )
    assert isinstance(result, str)
    assert len(result) > 0


@patch("utils.tg")
def test_textgrad_preserves_format(mock_tg):
    mock_var = MagicMock()
    mock_var.value = (
        "#Hallucinated Answer#: Optimized answer. "
        "#Justification of Hallucinated answer#: Good justification."
    )
    mock_tg.Variable.return_value = mock_var
    mock_tg.TextLoss.return_value = MagicMock(return_value=MagicMock())
    mock_tg.TGD.return_value = MagicMock()

    from utils import optimize_with_textgrad
    result = optimize_with_textgrad(
        "initial", "truth", "q?", "knowledge", max_iterations=1
    )
    assert "#Hallucinated Answer#:" in result
    assert "#Justification of Hallucinated answer#:" in result


@patch("utils.tg")
def test_textgrad_called_with_correct_params(mock_tg):
    mock_var = MagicMock()
    mock_var.value = "optimized"
    mock_tg.Variable.return_value = mock_var
    mock_loss = MagicMock(return_value=MagicMock())
    mock_tg.TextLoss.return_value = mock_loss
    mock_optimizer = MagicMock()
    mock_tg.TGD.return_value = mock_optimizer

    from utils import optimize_with_textgrad
    optimize_with_textgrad("response", "truth", "question?", "knowledge", max_iterations=3)

    mock_tg.set_backward_engine.assert_called_once_with("gpt-4o-mini", override=True)
    mock_tg.Variable.assert_called_once()
    assert mock_tg.Variable.call_args.kwargs["requires_grad"] is True
    mock_tg.TextLoss.assert_called_once()
    mock_tg.TGD.assert_called_once()


@patch("utils.tg")
def test_textgrad_respects_max_iterations(mock_tg):
    mock_var = MagicMock()
    mock_var.value = "optimized"
    mock_tg.Variable.return_value = mock_var
    mock_loss_result = MagicMock()
    mock_loss = MagicMock(return_value=mock_loss_result)
    mock_tg.TextLoss.return_value = mock_loss
    mock_optimizer = MagicMock()
    mock_tg.TGD.return_value = mock_optimizer

    from utils import optimize_with_textgrad
    optimize_with_textgrad("response", "truth", "q?", "knowledge", max_iterations=3)

    assert mock_optimizer.step.call_count == 3
    assert mock_optimizer.zero_grad.call_count == 3
    assert mock_loss_result.backward.call_count == 3
