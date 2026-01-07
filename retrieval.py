from typing import List, Dict, Any, Tuple
from flashrank import Ranker, RerankRequest

from config import (
    TOP_K_INITIAL,
    TOP_K_RERANKED,
    RERANKER_MODEL_NAME
)
from vector_store import get_vector_store


class RetrievalEngine:
    def __init__(self):
        self.vector_store = get_vector_store()
        self._reranker = None
        print("Retrieval Engine initialized")
    
    @property
    def reranker(self):
        if self._reranker is None:
            print(f"Loading FlashRank: {RERANKER_MODEL_NAME}")
            try:
                self._reranker = Ranker(model_name=RERANKER_MODEL_NAME, cache_dir="./flashrank_cache")
                print("FlashRank ready!")
            except Exception as e:
                print(f"FlashRank could not be loaded: {e}")
                raise
        return self._reranker
    
    def retrieve(
        self, 
        query: str, 
        use_reranking: bool = True,
        top_k_initial: int = TOP_K_INITIAL,
        top_k_final: int = TOP_K_RERANKED
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        debug_info = {
            "query": query,
            "use_reranking": use_reranking,
            "top_k_initial": top_k_initial,
            "top_k_final": top_k_final
        }
        
        try:
            initial_results = self.vector_store.search(
                query=query, 
                top_k=top_k_initial
            )
            
            debug_info["stage1_count"] = len(initial_results)
            debug_info["stage1_top_score"] = initial_results[0]["score"] if initial_results else 0
            
            if not use_reranking or len(initial_results) == 0:
                final_results = initial_results[:top_k_final]
                debug_info["final_method"] = "vector_only"
                return final_results, debug_info
            
            passages = [
                {"id": i, "text": r["text"], "meta": r}
                for i, r in enumerate(initial_results)
            ]
            
            rerank_request = RerankRequest(query=query, passages=passages)
            reranked = self.reranker.rerank(rerank_request)
            
            reranked_results = []
            for item in reranked:
                original_meta = item["meta"]
                reranked_results.append({
                    "id": original_meta["id"],
                    "score": item["score"],
                    "original_score": original_meta["score"],
                    "text": item["text"],
                    "source": original_meta["source"],
                    "chunk_index": original_meta["chunk_index"],
                    "page_number": original_meta.get("page_number", -1),
                    "metadata": original_meta["metadata"]
                })
            
            reranked_results.sort(key=lambda x: x["score"], reverse=True)
            final_results = reranked_results[:top_k_final]
            
            debug_info["final_method"] = "reranked"
            debug_info["stage2_top_score"] = final_results[0]["score"] if final_results else 0
            
            return final_results, debug_info
            
        except Exception as e:
            print(f"Retrieval error: {e}")
            debug_info["error"] = str(e)
            return [], debug_info
    
    def build_context(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "No context found."
        
        context_parts = []
        for i, r in enumerate(results, 1):
            source = r.get("source", "Unknown Source")
            page = r.get("page_number", -1)
            text = r.get("text", "")
            
            if page > 0:
                context_parts.append(f"[Kaynak {i}: {source}, Sayfa {page}]\n{text}")
            else:
                context_parts.append(f"[Kaynak {i}: {source}]\n{text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def format_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        sources = []
        for i, r in enumerate(results, 1):
            sources.append({
                "index": i,
                "source": r.get("source", "Bilinmeyen"),
                "page_number": r.get("page_number", -1),
                "text": r.get("text", ""),
                "score": r.get("score", 0),
                "original_score": r.get("original_score", r.get("score", 0))
            })
        return sources


_retrieval_engine_instance = None


def get_retrieval_engine() -> RetrievalEngine:
    global _retrieval_engine_instance
    if _retrieval_engine_instance is None:
        _retrieval_engine_instance = RetrievalEngine()
    return _retrieval_engine_instance


if __name__ == "__main__":
    print("Bu modül test için önce vector_store'a döküman eklenmesi gerekir.")
    print("Tam test için app.py'yi çalıştırın.")
