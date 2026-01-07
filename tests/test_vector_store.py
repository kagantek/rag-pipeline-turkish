"""Unit tests for vector_store module."""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestVectorStoreInit(unittest.TestCase):
    """Test VectorStore initialization."""

    @patch('vector_store.get_embedder')
    @patch('vector_store.QdrantVectorStore')
    @patch('vector_store.QdrantClient')
    def test_init_memory_mode(self, mock_client_class, mock_vs_class, mock_embedder):
        """Verify VectorStore initializes in memory mode."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client_class.return_value = mock_client
        
        from vector_store import VectorStore
        store = VectorStore(use_memory=True)
        
        mock_client_class.assert_called_with(":memory:")
        self.assertTrue(store.use_memory)

    @patch('vector_store.get_embedder')
    @patch('vector_store.QdrantVectorStore')
    @patch('vector_store.QdrantClient')
    def test_init_disk_mode(self, mock_client_class, mock_vs_class, mock_embedder):
        """Verify VectorStore initializes in disk mode."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client_class.return_value = mock_client
        
        from vector_store import VectorStore
        store = VectorStore(path="./test_db", use_memory=False)
        
        mock_client_class.assert_called_with(path="./test_db")
        self.assertFalse(store.use_memory)

    @patch('vector_store.get_embedder')
    @patch('vector_store.QdrantVectorStore')
    @patch('vector_store.QdrantClient')
    def test_collection_created_if_not_exists(self, mock_client_class, mock_vs_class, mock_embedder):
        """Verify collection is created when it does not exist."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client_class.return_value = mock_client
        
        from vector_store import VectorStore
        store = VectorStore(collection_name="test_collection", use_memory=True)
        
        mock_client.create_collection.assert_called_once()


class TestVectorStoreAddDocuments(unittest.TestCase):
    """Test VectorStore add_documents method."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock()
        self.mock_client.get_collections.return_value.collections = []
        self.mock_vs = MagicMock()
        self.mock_embedder = MagicMock()

    @patch('vector_store.get_embedder')
    @patch('vector_store.QdrantVectorStore')
    @patch('vector_store.QdrantClient')
    def test_add_documents_with_metadata(self, mock_client_class, mock_vs_class, mock_embedder_func):
        """Verify add_documents works with metadata."""
        mock_client_class.return_value = self.mock_client
        mock_vs_class.return_value = self.mock_vs
        mock_embedder_func.return_value = self.mock_embedder
        self.mock_vs.add_documents.return_value = ["id1", "id2"]
        
        from vector_store import VectorStore
        store = VectorStore(use_memory=True)
        
        texts = ["text1", "text2"]
        metadatas = [{"source": "doc1"}, {"source": "doc2"}]
        
        result = store.add_documents(texts, metadatas)
        
        self.assertEqual(result, ["id1", "id2"])
        self.mock_vs.add_documents.assert_called_once()

    @patch('vector_store.get_embedder')
    @patch('vector_store.QdrantVectorStore')
    @patch('vector_store.QdrantClient')
    def test_add_documents_without_metadata(self, mock_client_class, mock_vs_class, mock_embedder_func):
        """Verify add_documents works without metadata."""
        mock_client_class.return_value = self.mock_client
        mock_vs_class.return_value = self.mock_vs
        mock_embedder_func.return_value = self.mock_embedder
        self.mock_vs.add_documents.return_value = ["id1"]
        
        from vector_store import VectorStore
        store = VectorStore(use_memory=True)
        
        result = store.add_documents(["text1"])
        
        self.assertEqual(len(result), 1)


class TestVectorStoreSearch(unittest.TestCase):
    """Test VectorStore search method."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock()
        self.mock_client.get_collections.return_value.collections = []
        self.mock_vs = MagicMock()
        self.mock_embedder = MagicMock()

    @patch('vector_store.get_embedder')
    @patch('vector_store.QdrantVectorStore')
    @patch('vector_store.QdrantClient')
    def test_search_returns_formatted_results(self, mock_client_class, mock_vs_class, mock_embedder_func):
        """Verify search returns properly formatted results."""
        mock_client_class.return_value = self.mock_client
        mock_vs_class.return_value = self.mock_vs
        mock_embedder_func.return_value = self.mock_embedder
        
        mock_doc = MagicMock()
        mock_doc.page_content = "test content"
        mock_doc.metadata = {
            "_id": "123",
            "source": "test.pdf",
            "chunk_index": 0,
            "page_number": 1
        }
        self.mock_vs.similarity_search_with_score.return_value = [(mock_doc, 0.95)]
        
        from vector_store import VectorStore
        store = VectorStore(use_memory=True)
        
        results = store.search("test query", top_k=5)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "test content")
        self.assertEqual(results[0]["score"], 0.95)
        self.assertEqual(results[0]["source"], "test.pdf")

    @patch('vector_store.get_embedder')
    @patch('vector_store.QdrantVectorStore')
    @patch('vector_store.QdrantClient')
    def test_search_handles_exception(self, mock_client_class, mock_vs_class, mock_embedder_func):
        """Verify search handles exceptions gracefully."""
        mock_client_class.return_value = self.mock_client
        mock_vs_class.return_value = self.mock_vs
        mock_embedder_func.return_value = self.mock_embedder
        self.mock_vs.similarity_search_with_score.side_effect = Exception("Search error")
        
        from vector_store import VectorStore
        store = VectorStore(use_memory=True)
        
        results = store.search("test query")
        
        self.assertEqual(results, [])


class TestVectorStoreStats(unittest.TestCase):
    """Test VectorStore statistics methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock()
        self.mock_client.get_collections.return_value.collections = []

    @patch('vector_store.get_embedder')
    @patch('vector_store.QdrantVectorStore')
    @patch('vector_store.QdrantClient')
    def test_get_collection_stats(self, mock_client_class, mock_vs_class, mock_embedder_func):
        """Verify get_collection_stats returns correct info."""
        mock_client_class.return_value = self.mock_client
        
        mock_info = MagicMock()
        mock_info.vectors_count = 100
        mock_info.points_count = 100
        mock_info.status = "green"
        self.mock_client.get_collection.return_value = mock_info
        
        from vector_store import VectorStore
        store = VectorStore(use_memory=True)
        
        stats = store.get_collection_stats()
        
        self.assertEqual(stats["vectors_count"], 100)
        self.assertEqual(stats["points_count"], 100)

    @patch('vector_store.get_embedder')
    @patch('vector_store.QdrantVectorStore')
    @patch('vector_store.QdrantClient')
    def test_collection_exists_true(self, mock_client_class, mock_vs_class, mock_embedder_func):
        """Verify collection_exists returns True when collection has data."""
        mock_client_class.return_value = self.mock_client
        
        mock_info = MagicMock()
        mock_info.vectors_count = 100
        mock_info.points_count = 100
        mock_info.status = "green"
        self.mock_client.get_collection.return_value = mock_info
        
        from vector_store import VectorStore
        store = VectorStore(use_memory=True)
        
        self.assertTrue(store.collection_exists())

    @patch('vector_store.get_embedder')
    @patch('vector_store.QdrantVectorStore')
    @patch('vector_store.QdrantClient')
    def test_collection_exists_false(self, mock_client_class, mock_vs_class, mock_embedder_func):
        """Verify collection_exists returns False when collection is empty."""
        mock_client_class.return_value = self.mock_client
        
        mock_info = MagicMock()
        mock_info.vectors_count = 0
        mock_info.points_count = 0
        mock_info.status = "green"
        self.mock_client.get_collection.return_value = mock_info
        
        from vector_store import VectorStore
        store = VectorStore(use_memory=True)
        
        self.assertFalse(store.collection_exists())


class TestVectorStoreClear(unittest.TestCase):
    """Test VectorStore clear_collection method."""

    @patch('vector_store.get_embedder')
    @patch('vector_store.QdrantVectorStore')
    @patch('vector_store.QdrantClient')
    def test_clear_collection(self, mock_client_class, mock_vs_class, mock_embedder_func):
        """Verify clear_collection deletes and recreates collection."""
        mock_client = MagicMock()
        mock_client.get_collections.return_value.collections = []
        mock_client_class.return_value = mock_client
        
        from vector_store import VectorStore
        store = VectorStore(collection_name="test", use_memory=True)
        
        store.clear_collection()
        
        mock_client.delete_collection.assert_called_with("test")


if __name__ == "__main__":
    unittest.main()
