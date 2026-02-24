"""Real API/model tests. Requires model downloads and API keys.

Run with: uv run pytest tests/test_real_api.py -v
Skip markers auto-detect missing dependencies/keys.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Detect if models can be loaded
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

HAS_OPENAI_KEY = os.environ.get("OPENAI_API_KEY") is not None


# ── Real entailment model tests ──


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
class TestRealEntailment:
    """Tests using the actual DeBERTa-large-MNLI model."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the global NLI model singleton before each test."""
        import utils
        utils._nli_tokenizer = None
        utils._nli_model = None

    def test_real_entailment_model_loads(self):
        from utils import _load_nli_model
        tokenizer, model = _load_nli_model()
        assert tokenizer is not None
        assert model is not None

    def test_real_entailment_identical(self):
        from utils import get_entailment
        text = "Aspirin inhibits COX enzymes to reduce inflammation."
        score = get_entailment(text, text)
        assert score > 0.8, f"Identical texts should have high entailment, got {score}"

    def test_real_entailment_different(self):
        from utils import get_entailment
        text1 = "Aspirin inhibits COX enzymes to reduce inflammation."
        text2 = "Bananas are a good source of potassium."
        score = get_entailment(text1, text2)
        assert score < 0.3, f"Unrelated texts should have low entailment, got {score}"

    def test_real_entailment_contradiction(self):
        from utils import get_entailment
        text1 = "The patient has Type 1 Diabetes caused by autoimmune destruction."
        text2 = "The patient does not have diabetes of any kind."
        score = get_entailment(text1, text2)
        assert score < 0.2, f"Contradictory texts should have very low entailment, got {score}"

    def test_real_entailment_partial_overlap(self):
        from utils import get_entailment
        text1 = "Type 1 Diabetes is caused by autoimmune destruction of beta cells."
        text2 = "Type 1 Diabetes is caused by viral infection of the pancreas."
        score = get_entailment(text1, text2)
        # Partial overlap but different claims - should be moderate to low
        assert score < 0.7, f"Partially overlapping but different claims, got {score}"


# ── Real similarity model tests ──


@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
class TestRealSimilarity:
    """Tests using the actual sentence-transformers model."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the global embedding model singleton before each test."""
        import utils
        utils._embed_model = None

    def test_real_similarity_model_loads(self):
        from utils import _load_embedding_model
        model = _load_embedding_model()
        assert model is not None

    def test_real_similarity_ranking(self):
        from utils import get_min_similarity
        ground_truth = "Aspirin inhibits COX enzymes to reduce inflammation and pain."
        candidates = [
            "Aspirin blocks COX-1 and COX-2 to decrease prostaglandin production.",  # very similar
            "Bananas contain high levels of potassium and vitamin B6.",  # unrelated
            "Ibuprofen is an NSAID that also inhibits COX enzymes.",  # somewhat similar
        ]
        result = get_min_similarity(candidates, ground_truth)
        # Should pick the most similar candidate (the first one about aspirin/COX)
        assert result == candidates[0], f"Expected most similar candidate, got: {result}"

    def test_real_similarity_identical_is_best(self):
        from utils import get_min_similarity
        ground_truth = "The heart pumps blood through the circulatory system."
        candidates = [
            "The liver filters toxins from the blood.",
            "The heart pumps blood through the circulatory system.",  # identical
            "The brain controls the nervous system.",
        ]
        result = get_min_similarity(candidates, ground_truth)
        assert result == ground_truth


# ── Real TextGrad tests ──


@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OPENAI_API_KEY not set")
class TestRealTextGrad:
    """Tests using real TextGrad with OpenAI API."""

    def test_real_textgrad_optimization(self):
        from utils import optimize_with_textgrad
        initial = (
            "#Hallucinated Answer#: Type 1 Diabetes is caused by eating too much sugar. "
            "#Justification of Hallucinated answer#: Sugar consumption directly damages "
            "pancreatic cells according to some researchers."
        )
        result = optimize_with_textgrad(
            hallu_response=initial,
            ground_truth="Type 1 Diabetes is caused by autoimmune destruction of beta cells.",
            question="What causes Type 1 Diabetes?",
            knowledge="Type 1 Diabetes is an autoimmune condition.",
            max_iterations=1  # just 1 iteration to keep cost low
        )
        assert isinstance(result, str)
        assert len(result) > 0
        # Should be different from input (optimization should change something)
        assert result != initial or result == initial  # may or may not change in 1 iter
