"""Pytest configuration and shared fixtures."""
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        "KDV orani %18 olarak belirlenmistir.",
        "Fatura kesim tarihi ayin son is gunudur.",
        "Amortisman suresi 5 yil olarak hesaplanir."
    ]


@pytest.fixture
def sample_metadatas():
    """Provide sample metadata for testing."""
    return [
        {"source": "vergi_kanunu.pdf", "page_number": 1, "chunk_index": 0},
        {"source": "fatura_yonetmeligi.pdf", "page_number": 5, "chunk_index": 1},
        {"source": "amortisman_tablosu.pdf", "page_number": 3, "chunk_index": 2}
    ]


@pytest.fixture
def sample_search_results():
    """Provide sample search results for testing."""
    return [
        {
            "id": "1",
            "score": 0.95,
            "text": "KDV orani %18 olarak belirlenmistir.",
            "source": "vergi_kanunu.pdf",
            "chunk_index": 0,
            "page_number": 1,
            "metadata": {"source": "vergi_kanunu.pdf"}
        },
        {
            "id": "2",
            "score": 0.85,
            "text": "Fatura kesim tarihi ayin son is gunudur.",
            "source": "fatura_yonetmeligi.pdf",
            "chunk_index": 1,
            "page_number": 5,
            "metadata": {"source": "fatura_yonetmeligi.pdf"}
        }
    ]


@pytest.fixture
def sample_chat_history():
    """Provide sample chat history for testing."""
    return [
        {"role": "user", "content": "KDV orani nedir?"},
        {"role": "assistant", "content": "KDV orani %18'dir."}
    ]


@pytest.fixture
def mock_groq_response():
    """Provide mock Groq API response."""
    from unittest.mock import MagicMock
    
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Test response from LLM"))
    ]
    return mock_response


@pytest.fixture
def mock_embedding():
    """Provide mock embedding vector."""
    import numpy as np
    return np.random.rand(384).tolist()


@pytest.fixture
def mock_embeddings_batch():
    """Provide mock batch of embedding vectors."""
    import numpy as np
    return np.random.rand(3, 384).tolist()
