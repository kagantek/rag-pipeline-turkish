"""Unit tests for retrieval module."""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRetrievalEngineInit(unittest.TestCase):
    """Test RetrievalEngine initialization."""

    @patch('retrieval.get_vector_store')
    def test_init_creates_vector_store(self, mock_get_vs):
        """Verify RetrievalEngine initializes with vector store."""
        mock_vs = MagicMock()
        mock_get_vs.return_value = mock_vs
        
        from retrieval import RetrievalEngine
        engine = RetrievalEngine()
        
        self.assertEqual(engine.vector_store, mock_vs)
        self.assertIsNone(engine._reranker)

    @patch('retrieval.get_vector_store')
    @patch('retrieval.Ranker')
    def test_reranker_lazy_loading(self, mock_ranker_class, mock_get_vs):
        """Verify reranker is lazy loaded on first access."""
        mock_vs = MagicMock()
        mock_get_vs.return_value = mock_vs
        mock_ranker = MagicMock()
        mock_ranker_class.return_value = mock_ranker
        
        from retrieval import RetrievalEngine
        engine = RetrievalEngine()
        
        # Reranker should not be loaded yet
        self.assertIsNone(engine._reranker)
        
        # Access reranker property
        _ = engine.reranker
        
        # Now reranker should be loaded
        mock_ranker_class.assert_called_once()


class TestRetrievalEngineRetrieve(unittest.TestCase):
    """Test RetrievalEngine retrieve method."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_vs = MagicMock()
        self.mock_vs.search.return_value = [
            {
                "id": "1",
                "score": 0.9,
                "text": "Test text 1",
                "source": "doc1.pdf",
                "chunk_index": 0,
                "page_number": 1,
                "metadata": {}
            },
            {
                "id": "2",
                "score": 0.8,
                "text": "Test text 2",
                "source": "doc2.pdf",
                "chunk_index": 1,
                "page_number": 2,
                "metadata": {}
            }
        ]

    @patch('retrieval.get_vector_store')
    def test_retrieve_without_reranking(self, mock_get_vs):
        """Verify retrieve works without reranking."""
        mock_get_vs.return_value = self.mock_vs
        
        from retrieval import RetrievalEngine
        engine = RetrievalEngine()
        
        results, debug_info = engine.retrieve(
            query="test query",
            use_reranking=False,
            top_k_final=2
        )
        
        self.assertEqual(len(results), 2)
        self.assertEqual(debug_info["final_method"], "vector_only")

    @patch('retrieval.get_vector_store')
    @patch('retrieval.Ranker')
    def test_retrieve_with_reranking(self, mock_ranker_class, mock_get_vs):
        """Verify retrieve works with reranking."""
        mock_get_vs.return_value = self.mock_vs
        
        mock_ranker = MagicMock()
        mock_ranker.rerank.return_value = [
            {
                "id": 0,
                "score": 0.95,
                "text": "Test text 1",
                "meta": self.mock_vs.search.return_value[0]
            }
        ]
        mock_ranker_class.return_value = mock_ranker
        
        from retrieval import RetrievalEngine
        engine = RetrievalEngine()
        
        results, debug_info = engine.retrieve(
            query="test query",
            use_reranking=True,
            top_k_final=1
        )
        
        self.assertEqual(debug_info["final_method"], "reranked")

    @patch('retrieval.get_vector_store')
    def test_retrieve_empty_results(self, mock_get_vs):
        """Verify retrieve handles empty results."""
        mock_vs = MagicMock()
        mock_vs.search.return_value = []
        mock_get_vs.return_value = mock_vs
        
        from retrieval import RetrievalEngine
        engine = RetrievalEngine()
        
        results, debug_info = engine.retrieve(
            query="test query",
            use_reranking=True
        )
        
        self.assertEqual(len(results), 0)
        self.assertEqual(debug_info["final_method"], "vector_only")

    @patch('retrieval.get_vector_store')
    def test_retrieve_handles_exception(self, mock_get_vs):
        """Verify retrieve handles exceptions gracefully."""
        mock_vs = MagicMock()
        mock_vs.search.side_effect = Exception("Search error")
        mock_get_vs.return_value = mock_vs
        
        from retrieval import RetrievalEngine
        engine = RetrievalEngine()
        
        results, debug_info = engine.retrieve(query="test")
        
        self.assertEqual(results, [])
        self.assertIn("error", debug_info)

    @patch('retrieval.get_vector_store')
    def test_retrieve_debug_info_contains_query(self, mock_get_vs):
        """Verify debug info contains query information."""
        mock_get_vs.return_value = self.mock_vs
        
        from retrieval import RetrievalEngine
        engine = RetrievalEngine()
        
        results, debug_info = engine.retrieve(
            query="my test query",
            use_reranking=False
        )
        
        self.assertEqual(debug_info["query"], "my test query")


class TestRetrievalEngineBuildContext(unittest.TestCase):
    """Test RetrievalEngine build_context method."""

    @patch('retrieval.get_vector_store')
    def test_build_context_empty_results(self, mock_get_vs):
        """Verify build_context handles empty results."""
        mock_get_vs.return_value = MagicMock()
        
        from retrieval import RetrievalEngine
        engine = RetrievalEngine()
        
        context = engine.build_context([])
        
        self.assertIn("No context found", context)

    @patch('retrieval.get_vector_store')
    def test_build_context_with_results(self, mock_get_vs):
        """Verify build_context formats results correctly."""
        mock_get_vs.return_value = MagicMock()
        
        from retrieval import RetrievalEngine
        engine = RetrievalEngine()
        
        results = [
            {
                "source": "test.pdf",
                "page_number": 5,
                "text": "Test content"
            }
        ]
        
        context = engine.build_context(results)
        
        self.assertIn("test.pdf", context)
        self.assertIn("Test content", context)
        self.assertIn("5", context)

    @patch('retrieval.get_vector_store')
    def test_build_context_without_page_number(self, mock_get_vs):
        """Verify build_context handles missing page number."""
        mock_get_vs.return_value = MagicMock()
        
        from retrieval import RetrievalEngine
        engine = RetrievalEngine()
        
        results = [
            {
                "source": "test.pdf",
                "page_number": -1,
                "text": "Test content"
            }
        ]
        
        context = engine.build_context(results)
        
        self.assertIn("test.pdf", context)
        self.assertNotIn("Sayfa -1", context)


class TestRetrievalEngineFormatSources(unittest.TestCase):
    """Test RetrievalEngine format_sources method."""

    @patch('retrieval.get_vector_store')
    def test_format_sources(self, mock_get_vs):
        """Verify format_sources returns correctly structured data."""
        mock_get_vs.return_value = MagicMock()
        
        from retrieval import RetrievalEngine
        engine = RetrievalEngine()
        
        results = [
            {
                "source": "doc.pdf",
                "page_number": 3,
                "text": "Content",
                "score": 0.9,
                "original_score": 0.85
            }
        ]
        
        sources = engine.format_sources(results)
        
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["index"], 1)
        self.assertEqual(sources[0]["source"], "doc.pdf")
        self.assertEqual(sources[0]["page_number"], 3)
        self.assertEqual(sources[0]["score"], 0.9)

    @patch('retrieval.get_vector_store')
    def test_format_sources_empty(self, mock_get_vs):
        """Verify format_sources handles empty results."""
        mock_get_vs.return_value = MagicMock()
        
        from retrieval import RetrievalEngine
        engine = RetrievalEngine()
        
        sources = engine.format_sources([])
        
        self.assertEqual(sources, [])


class TestGetRetrievalEngine(unittest.TestCase):
    """Test get_retrieval_engine singleton function."""

    @patch('retrieval._retrieval_engine_instance', None)
    @patch('retrieval.RetrievalEngine')
    def test_get_retrieval_engine_creates_singleton(self, mock_class):
        """Verify get_retrieval_engine creates singleton instance."""
        import retrieval
        retrieval._retrieval_engine_instance = None
        
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        
        result1 = retrieval.get_retrieval_engine()
        result2 = retrieval.get_retrieval_engine()
        
        self.assertEqual(result1, result2)


if __name__ == "__main__":
    unittest.main()
