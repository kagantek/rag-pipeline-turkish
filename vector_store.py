from typing import List, Dict, Any, Optional
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from config import (
    QDRANT_PATH, 
    COLLECTION_NAME, 
    EMBEDDING_DIMENSION,
    USE_MEMORY_MODE
)
from embeddings import get_embedder


class VectorStore:
    def __init__(
        self, 
        path: str = QDRANT_PATH, 
        collection_name: str = COLLECTION_NAME,
        use_memory: bool = USE_MEMORY_MODE
    ):

        self.collection_name = collection_name
        self.use_memory = use_memory
        self.path = path
        
        if use_memory:
            print("Qdrant başlatılıyor: Bellek modu (geçici)")
            self._client = QdrantClient(":memory:")
        else:
            print(f"Qdrant başlatılıyor: Disk modu ({path})")
            self._client = QdrantClient(path=path)
        
        self._ensure_collection_exists()

        self._embedder = get_embedder()

        self._vector_store = QdrantVectorStore(
            client=self._client,
            collection_name=self.collection_name,
            embedding=self._embedder
        )
        
        print(f"Koleksiyon hazır: {collection_name}")
    
    def _ensure_collection_exists(self):
        try:
            collections = self._client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                print(f"Yeni koleksiyon oluşturuluyor: {self.collection_name}")
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=EMBEDDING_DIMENSION,
                        distance=Distance.COSINE
                    )
                )
        except Exception as e:
            print(f"Koleksiyon kontrolü hatası: {e}")
            raise
    
    def add_documents(
        self, 
        texts: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]
        
        try:
            ids = self._vector_store.add_documents(documents)
            print(f"{len(documents)} döküman eklendi")
            return ids
        except Exception as e:
            print(f"Döküman ekleme hatası: {e}")
            raise
    
    def search(
        self, 
        query: str, 
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        try:
            results = self._vector_store.similarity_search_with_score(
                query=query,
                k=top_k
            )
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "id": doc.metadata.get("_id", ""),
                    "score": score,
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", "Bilinmeyen"),
                    "chunk_index": doc.metadata.get("chunk_index", -1),
                    "page_number": doc.metadata.get("page_number", -1),
                    "metadata": doc.metadata
                })
            
            return formatted_results
        except Exception as e:
            print(f"Arama hatası: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            info = self._client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            print(f"İstatistik alma hatası: {e}")
            return {
                "name": self.collection_name,
                "vectors_count": 0,
                "points_count": 0,
                "status": "error"
            }
    
    def clear_collection(self):
        try:
            print(f"Koleksiyon siliniyor: {self.collection_name}")
            self._client.delete_collection(self.collection_name)
            self._ensure_collection_exists()
            
            self._vector_store = QdrantVectorStore(
                client=self._client,
                collection_name=self.collection_name,
                embedding=self._embedder
            )
            print("Koleksiyon sıfırlandı")
        except Exception as e:
            print(f"Koleksiyon silme hatası: {e}")
            raise
    
    def collection_exists(self) -> bool:
        stats = self.get_collection_stats()
        return stats["points_count"] > 0


_vector_store_instance = None


def get_vector_store() -> VectorStore:
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance


if __name__ == "__main__":
    store = VectorStore(use_memory=True)
    
    texts = [
        "KDV oranı %18 olarak belirlenmiştir.",
        "Fatura kesim tarihi ayın son iş günüdür.",
        "Amortisman süresi 5 yıl olarak hesaplanır."
    ]
    metadatas = [
        {"source": "vergi_kanunu.pdf", "page_number": 1},
        {"source": "fatura_yonetmeligi.pdf", "page_number": 5},
        {"source": "amortisman_tablosu.pdf", "page_number": 3}
    ]
    
    ids = store.add_documents(texts, metadatas)
    print(f"Eklenen ID'ler: {ids}")

    results = store.search("KDV oranı nedir?", top_k=2)
    print(f"\nArama Sonuçları:")
    for r in results:
        print(f"  - Score: {r['score']:.4f} | Source: {r['source']}")
        print(f"    Text: {r['text'][:50]}...")
    
    # Stats
    stats = store.get_collection_stats()
    print(f"\nKoleksiyon İstatistikleri: {stats}")
