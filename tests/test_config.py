"""Unit tests for config module."""
import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION,
    RERANKER_MODEL_NAME,
    LLM_MODELS,
    DEFAULT_LLM_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_INITIAL,
    TOP_K_RERANKED,
    QDRANT_PATH,
    COLLECTION_NAME,
    USE_MEMORY_MODE,
    SYSTEM_PROMPT_TR,
    UI_TEXTS
)


class TestConfigConstants(unittest.TestCase):
    """Test configuration constants are properly defined."""

    def test_embedding_model_name_is_string(self):
        """Verify embedding model name is a non-empty string."""
        self.assertIsInstance(EMBEDDING_MODEL_NAME, str)
        self.assertTrue(len(EMBEDDING_MODEL_NAME) > 0)

    def test_embedding_dimension_is_positive_integer(self):
        """Verify embedding dimension is a positive integer."""
        self.assertIsInstance(EMBEDDING_DIMENSION, int)
        self.assertGreater(EMBEDDING_DIMENSION, 0)

    def test_reranker_model_name_is_string(self):
        """Verify reranker model name is a non-empty string."""
        self.assertIsInstance(RERANKER_MODEL_NAME, str)
        self.assertTrue(len(RERANKER_MODEL_NAME) > 0)

    def test_llm_models_is_dict(self):
        """Verify LLM models is a non-empty dictionary."""
        self.assertIsInstance(LLM_MODELS, dict)
        self.assertTrue(len(LLM_MODELS) > 0)

    def test_default_llm_model_in_models(self):
        """Verify default LLM model is one of the available models."""
        self.assertIn(DEFAULT_LLM_MODEL, LLM_MODELS.values())


class TestChunkingConfig(unittest.TestCase):
    """Test chunking configuration parameters."""

    def test_chunk_size_is_positive(self):
        """Verify chunk size is a positive integer."""
        self.assertIsInstance(CHUNK_SIZE, int)
        self.assertGreater(CHUNK_SIZE, 0)

    def test_chunk_overlap_is_non_negative(self):
        """Verify chunk overlap is a non-negative integer."""
        self.assertIsInstance(CHUNK_OVERLAP, int)
        self.assertGreaterEqual(CHUNK_OVERLAP, 0)

    def test_chunk_overlap_less_than_chunk_size(self):
        """Verify chunk overlap is less than chunk size."""
        self.assertLess(CHUNK_OVERLAP, CHUNK_SIZE)


class TestSearchConfig(unittest.TestCase):
    """Test search configuration parameters."""

    def test_top_k_initial_is_positive(self):
        """Verify initial top-k is a positive integer."""
        self.assertIsInstance(TOP_K_INITIAL, int)
        self.assertGreater(TOP_K_INITIAL, 0)

    def test_top_k_reranked_is_positive(self):
        """Verify reranked top-k is a positive integer."""
        self.assertIsInstance(TOP_K_RERANKED, int)
        self.assertGreater(TOP_K_RERANKED, 0)

    def test_top_k_reranked_less_or_equal_initial(self):
        """Verify reranked top-k is less than or equal to initial top-k."""
        self.assertLessEqual(TOP_K_RERANKED, TOP_K_INITIAL)


class TestQdrantConfig(unittest.TestCase):
    """Test Qdrant configuration parameters."""

    def test_qdrant_path_is_string(self):
        """Verify Qdrant path is a string."""
        self.assertIsInstance(QDRANT_PATH, str)

    def test_collection_name_is_string(self):
        """Verify collection name is a non-empty string."""
        self.assertIsInstance(COLLECTION_NAME, str)
        self.assertTrue(len(COLLECTION_NAME) > 0)

    def test_use_memory_mode_is_bool(self):
        """Verify memory mode flag is a boolean."""
        self.assertIsInstance(USE_MEMORY_MODE, bool)


class TestPromptConfig(unittest.TestCase):
    """Test prompt configuration."""

    def test_system_prompt_contains_placeholders(self):
        """Verify system prompt contains required placeholders."""
        self.assertIn("{context}", SYSTEM_PROMPT_TR)
        self.assertIn("{question}", SYSTEM_PROMPT_TR)
        self.assertIn("{chat_history}", SYSTEM_PROMPT_TR)

    def test_system_prompt_is_non_empty(self):
        """Verify system prompt is a non-empty string."""
        self.assertIsInstance(SYSTEM_PROMPT_TR, str)
        self.assertTrue(len(SYSTEM_PROMPT_TR) > 100)


class TestUITexts(unittest.TestCase):
    """Test UI text configuration."""

    def test_ui_texts_is_dict(self):
        """Verify UI texts is a dictionary."""
        self.assertIsInstance(UI_TEXTS, dict)

    def test_ui_texts_has_required_keys(self):
        """Verify UI texts has essential keys."""
        required_keys = [
            "title",
            "upload_label",
            "process_button",
            "query_placeholder",
            "ask_button"
        ]
        for key in required_keys:
            self.assertIn(key, UI_TEXTS)

    def test_ui_texts_values_are_strings(self):
        """Verify all UI text values are strings."""
        for key, value in UI_TEXTS.items():
            self.assertIsInstance(value, str, f"Key '{key}' should have string value")


if __name__ == "__main__":
    unittest.main()
