"""
Retriever Module - Handles retrieval logic for RAG
"""
from typing import List, Dict

class Retriever:
    def __init__(self, vector_store, embeddings_module):
        """
        Initialize the retriever
        
        Args:
            vector_store: Instance of VectorStore class
            embeddings_module: Instance of EmbeddingsModule class
        """
        self.vector_store = vector_store
        self.embeddings_module = embeddings_module
        print("✅ Retriever initialized")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of retrieved chunks with scores
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embeddings_module.generate_embedding(query)
            
            if not query_embedding:
                print("❌ Failed to generate query embedding")
                return []
            
            # Query the vector store
            results = self.vector_store.query_vectors(
                query_embedding=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            return results
            
        except Exception as e:
            print(f"❌ Error in retrieval: {e}")
            return []
    
    def retrieve_with_score_threshold(self, query: str, top_k: int = 3, min_score: float = 0.0) -> List[Dict]:
        """
        Retrieve relevant chunks with a minimum score threshold
        
        Args:
            query: Query text
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of retrieved chunks above the score threshold
        """
        results = self.retrieve(query, top_k)
        
        # Filter results by score
        filtered_results = [
            result for result in results 
            if result.get('score', 0) >= min_score
        ]
        
        return filtered_results
    
    def get_retrieval_info(self, query: str, top_k: int = 3) -> Dict:
        """
        Get detailed retrieval information including scores
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            Dictionary with retrieval info
        """
        results = self.retrieve(query, top_k)
        
        return {
            'query': query,
            'num_results': len(results),
            'results': results,
            'max_score': max([r['score'] for r in results], default=0),
            'min_score': min([r['score'] for r in results], default=0),
            'avg_score': sum([r['score'] for r in results]) / len(results) if results else 0
        }
