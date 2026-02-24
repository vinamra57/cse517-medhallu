"""Integration tests for the hallucination generation pipeline with mocked external APIs."""
import pytest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import generate_hallucinated_answer


SAMPLE_QUESTION = "What is the primary cause of Type 1 Diabetes?"
SAMPLE_KNOWLEDGE = "Type 1 Diabetes is typically diagnosed in children and young adults."
SAMPLE_GROUND_TRUTH = "The primary cause of Type 1 Diabetes is the autoimmune destruction of insulin-producing beta cells in the pancreas."
SAMPLE_HALLU_RESPONSE = (
    "#Hallucinated Answer#: The primary cause of Type 1 Diabetes is a viral infection targeting the pancreas. "
    "#Justification of Hallucinated answer#: According to a 2018 study in the Journal of Pancreatic Research."
)


@patch("main.get_min_similarity")
@patch("main.optimize_with_textgrad")
@patch("main.get_entailment")
@patch("main.evaluate_response_quality")
@patch("main.LLM")
def test_full_pipeline_early_return(mock_llm_class, mock_eval, mock_entail, mock_textgrad, mock_similarity):
    """First attempt passes quality + entailment checks -> early return."""
    mock_instance = MagicMock()
    mock_instance.get_response.return_value = SAMPLE_HALLU_RESPONSE
    mock_llm_class.return_value = mock_instance

    # All 3 LLMs fooled, low entailment
    mock_eval.return_value = [True, True, True]
    mock_entail.return_value = 0.1

    result, difficulty = generate_hallucinated_answer(SAMPLE_QUESTION, SAMPLE_KNOWLEDGE, SAMPLE_GROUND_TRUTH)

    assert result == SAMPLE_HALLU_RESPONSE
    assert difficulty == "Hard"
    mock_textgrad.assert_not_called()
    mock_similarity.assert_not_called()


@patch("main.get_min_similarity")
@patch("main.optimize_with_textgrad")
@patch("main.get_entailment")
@patch("main.evaluate_response_quality")
@patch("main.LLM")
def test_full_pipeline_textgrad_path(mock_llm_class, mock_eval, mock_entail, mock_textgrad, mock_similarity):
    """First attempt fails quality, TextGrad optimized version passes."""
    mock_instance = MagicMock()
    mock_instance.get_response.return_value = SAMPLE_HALLU_RESPONSE
    mock_llm_class.return_value = mock_instance

    optimized_response = (
        "#Hallucinated Answer#: Type 1 Diabetes is primarily caused by pancreatic viral destruction. "
        "#Justification of Hallucinated answer#: Dr. Smith's 2020 Lancet study confirmed this."
    )
    mock_textgrad.return_value = optimized_response

    # First eval: no LLMs fooled (initial response fails)
    # Second eval: 2 LLMs fooled (optimized response passes)
    mock_eval.side_effect = [[False, False, False], [True, True, False]]
    # Entailment always low
    mock_entail.return_value = 0.2

    result, difficulty = generate_hallucinated_answer(SAMPLE_QUESTION, SAMPLE_KNOWLEDGE, SAMPLE_GROUND_TRUTH)

    assert result == optimized_response
    assert difficulty == "Medium"
    mock_textgrad.assert_called_once()
    mock_similarity.assert_not_called()


@patch("main.get_min_similarity")
@patch("main.optimize_with_textgrad")
@patch("main.get_entailment")
@patch("main.evaluate_response_quality")
@patch("main.LLM")
def test_full_pipeline_fallback(mock_llm_class, mock_eval, mock_entail, mock_textgrad, mock_similarity):
    """All attempts fail -> fallback to get_min_similarity."""
    mock_instance = MagicMock()
    mock_instance.get_response.return_value = SAMPLE_HALLU_RESPONSE
    mock_llm_class.return_value = mock_instance

    # All evaluations fail (no LLMs fooled)
    mock_eval.return_value = [False, False, False]
    mock_entail.return_value = 0.9  # high entailment = too similar

    optimized_responses = [f"optimized_{i}" for i in range(4)]
    mock_textgrad.side_effect = optimized_responses

    fallback = "best fallback hallucination"
    mock_similarity.return_value = fallback

    result, difficulty = generate_hallucinated_answer(SAMPLE_QUESTION, SAMPLE_KNOWLEDGE, SAMPLE_GROUND_TRUTH)

    assert result == fallback
    assert difficulty == 1
    assert mock_textgrad.call_count == 4
    mock_similarity.assert_called_once()


@patch("main.get_min_similarity")
@patch("main.optimize_with_textgrad")
@patch("main.get_entailment")
@patch("main.evaluate_response_quality")
@patch("main.LLM")
def test_pipeline_entailment_too_high_rejects(mock_llm_class, mock_eval, mock_entail, mock_textgrad, mock_similarity):
    """Even if LLMs are fooled, high entailment should reject the response."""
    mock_instance = MagicMock()
    mock_instance.get_response.return_value = SAMPLE_HALLU_RESPONSE
    mock_llm_class.return_value = mock_instance

    # LLMs fooled but entailment too high (hallucination too similar to truth)
    mock_eval.return_value = [True, True, True]
    mock_entail.return_value = 0.9  # above threshold of 0.75

    mock_textgrad.return_value = "optimized"
    mock_similarity.return_value = "fallback"

    result, difficulty = generate_hallucinated_answer(SAMPLE_QUESTION, SAMPLE_KNOWLEDGE, SAMPLE_GROUND_TRUTH)

    # Should fall through to fallback
    assert result == "fallback"
    assert difficulty == 1
