"""Unit tests for embeddings module."""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTurkishEmbedderInit(unittest.TestCase):
    """Test TurkishEmbedder initialization."""

    @patch('embeddings.SentenceTransformer')
    @patch('embeddings.torch')
    def test_embedder_init_with_cuda(self, mock_torch, mock_st):
        """Verify embedder initializes with CUDA when available."""
        mock_torch.cuda.is_available.return_value = True
        
        from embeddings import TurkishEmbedder
        embedder = TurkishEmbedder(model_name="test-model")
        
        self.assertEqual(embedder.model_name, "test-model")
        self.assertEqual(embedder.device, "cuda")

    @patch('embeddings.SentenceTransformer')
    @patch('embeddings.torch')
    def test_embedder_init_with_cpu(self, mock_torch, mock_st):
        """Verify embedder initializes with CPU when CUDA not available."""
        mock_torch.cuda.is_available.return_value = False
        
        from embeddings import TurkishEmbedder
        embedder = TurkishEmbedder(model_name="test-model")
        
        self.assertEqual(embedder.device, "cpu")

    @patch('embeddings.SentenceTransformer')
    @patch('embeddings.torch')
    def test_model_lazy_loading(self, mock_torch, mock_st):
        """Verify model is lazy loaded on first access."""
        mock_torch.cuda.is_available.return_value = False
        
        from embeddings import TurkishEmbedder
        embedder = TurkishEmbedder(model_name="test-model")
        
        # Model should not be loaded yet
        self.assertIsNone(embedder._model)
        
        # Access model property
        _ = embedder.model
        
        # Now model should be loaded
        mock_st.assert_called_once()


class TestTurkishEmbedderMethods(unittest.TestCase):
    """Test TurkishEmbedder embedding methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = MagicMock()
        self.mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        self.mock_model.get_sentence_embedding_dimension.return_value = 384

    @patch('embeddings.SentenceTransformer')
    @patch('embeddings.torch')
    def test_embed_documents_empty_list(self, mock_torch, mock_st):
        """Verify embed_documents returns empty list for empty input."""
        mock_torch.cuda.is_available.return_value = False
        
        from embeddings import TurkishEmbedder
        embedder = TurkishEmbedder()
        
        result = embedder.embed_documents([])
        self.assertEqual(result, [])

    @patch('embeddings.SentenceTransformer')
    @patch('embeddings.torch')
    def test_embed_documents_adds_prefix(self, mock_torch, mock_st):
        """Verify embed_documents adds passage prefix to texts."""
        mock_torch.cuda.is_available.return_value = False
        mock_st.return_value = self.mock_model
        
        from embeddings import TurkishEmbedder
        embedder = TurkishEmbedder()
        
        embedder.embed_documents(["test text"])
        
        call_args = self.mock_model.encode.call_args
        self.assertEqual(call_args[0][0], ["passage: test text"])

    @patch('embeddings.SentenceTransformer')
    @patch('embeddings.torch')
    def test_embed_query_adds_prefix(self, mock_torch, mock_st):
        """Verify embed_query adds query prefix to text."""
        mock_torch.cuda.is_available.return_value = False
        self.mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_st.return_value = self.mock_model
        
        from embeddings import TurkishEmbedder
        embedder = TurkishEmbedder()
        
        embedder.embed_query("test query")
        
        call_args = self.mock_model.encode.call_args
        self.assertEqual(call_args[0][0], "query: test query")

    @patch('embeddings.SentenceTransformer')
    @patch('embeddings.torch')
    def test_embed_documents_returns_list(self, mock_torch, mock_st):
        """Verify embed_documents returns list of lists."""
        mock_torch.cuda.is_available.return_value = False
        self.mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_st.return_value = self.mock_model
        
        from embeddings import TurkishEmbedder
        embedder = TurkishEmbedder()
        
        result = embedder.embed_documents(["text1", "text2"])
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    @patch('embeddings.SentenceTransformer')
    @patch('embeddings.torch')
    def test_embed_query_returns_list(self, mock_torch, mock_st):
        """Verify embed_query returns list of floats."""
        mock_torch.cuda.is_available.return_value = False
        self.mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_st.return_value = self.mock_model
        
        from embeddings import TurkishEmbedder
        embedder = TurkishEmbedder()
        
        result = embedder.embed_query("test")
        
        self.assertIsInstance(result, list)

    @patch('embeddings.SentenceTransformer')
    @patch('embeddings.torch')
    def test_embed_passages_returns_numpy(self, mock_torch, mock_st):
        """Verify embed_passages returns numpy array."""
        mock_torch.cuda.is_available.return_value = False
        self.mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_st.return_value = self.mock_model
        
        from embeddings import TurkishEmbedder
        embedder = TurkishEmbedder()
        
        result = embedder.embed_passages(["text1", "text2"])
        
        self.assertIsInstance(result, np.ndarray)

    @patch('embeddings.SentenceTransformer')
    @patch('embeddings.torch')
    def test_embedding_dimension_property(self, mock_torch, mock_st):
        """Verify embedding_dimension property returns correct value."""
        mock_torch.cuda.is_available.return_value = False
        mock_st.return_value = self.mock_model
        
        from embeddings import TurkishEmbedder
        embedder = TurkishEmbedder()
        
        dim = embedder.embedding_dimension
        
        self.assertEqual(dim, 384)


class TestGetEmbedder(unittest.TestCase):
    """Test get_embedder singleton function."""

    @patch('embeddings._embedder_instance', None)
    @patch('embeddings.TurkishEmbedder')
    def test_get_embedder_creates_singleton(self, mock_class):
        """Verify get_embedder creates singleton instance."""
        import embeddings
        embeddings._embedder_instance = None
        
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        
        result1 = embeddings.get_embedder()
        result2 = embeddings.get_embedder()
        
        # Should be the same instance
        self.assertEqual(result1, result2)


if __name__ == "__main__":
    unittest.main()
