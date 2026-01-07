import torch
from sentence_transformers import SentenceTransformer
from typing import List
from langchain_core.embeddings import Embeddings
import numpy as np

from config import EMBEDDING_MODEL_NAME


class TurkishEmbedder(Embeddings):
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model: {model_name}")
        print(f"Device: {self.device}")
        
        self._model = None
    
    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
            print("Embedding model ready!")
        return self._model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        prefixed_texts = [f"passage: {text}" for text in texts]
        
        embeddings = self.model.encode(
            prefixed_texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 10,
            normalize_embeddings=True
        )
        
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        prefixed_query = f"query: {text}"
        
        embedding = self.model.encode(
            prefixed_query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        return embedding.tolist()
    
    def embed_passages(self, texts: List[str]) -> np.ndarray:
        return np.array(self.embed_documents(texts))
    
    @property
    def embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


_embedder_instance = None


def get_embedder() -> TurkishEmbedder:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = TurkishEmbedder()
    return _embedder_instance


if __name__ == "__main__":
    embedder = TurkishEmbedder()
    
    test_passages = [
        "KDV oranı %18 olarak belirlenmiştir.",
        "Fatura kesim tarihi ayın son iş günüdür.",
        "Amortisman süresi 5 yıl olarak hesaplanır."
    ]
    
    test_query = "KDV oranı kaç?"
    
    print("\nTest: Document Embeddings (LangChain compatible)")
    doc_embs = embedder.embed_documents(test_passages)
    print(f"Count: {len(doc_embs)}, Dimension: {len(doc_embs[0])}")
    
    print("\nTest: Query Embedding (LangChain compatible)")
    query_emb = embedder.embed_query(test_query)
    print(f"Dimension: {len(query_emb)}")
    
    print(f"\nEmbedding Dimension: {embedder.embedding_dimension}")
