"""
Vector Store Module - Handles Pinecone operations (upsert/query)
"""
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from typing import List, Dict
import uuid

load_dotenv()

class VectorStore:
    def __init__(self, index_name: str):
        """
        Initialize the vector store
        
        Args:
            index_name: Name of the Pinecone index
        """
        self.index_name = index_name
        self.pc = Pinecone(api_key=os.getenv("PC_API_KEY"))
        self.index = self.pc.Index(index_name)
        print(f"✅ Vector store initialized with index: {index_name}")
    
    def upsert_vectors(self, texts: List[str], embeddings: List[List[float]], metadata: List[Dict] = None):
        """
        Insert or update vectors in Pinecone
        
        Args:
            texts: List of text chunks
            embeddings: List of embeddings for each text
            metadata: Optional list of metadata dictionaries
            
        Returns:
            Boolean indicating success
        """
        if not texts or not embeddings:
            print("❌ Empty texts or embeddings provided")
            return False
        
        if len(texts) != len(embeddings):
            print("❌ Mismatch between texts and embeddings count")
            return False
        
        if metadata is None:
            metadata = [{}] * len(texts)
        
        vectors_to_upsert = []
        
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            vector_id = str(uuid.uuid4())
            vectors_to_upsert.append({
                'id': vector_id,
                'values': embedding,
                'metadata': {
                    'text': text,
                    'chunk_index': i,
                    **metadata[i]
                }
            })
        
        try:
            self.index.upsert(vectors=vectors_to_upsert)
            print(f"✅ Upserted {len(vectors_to_upsert)} vectors to index")
            return True
        except Exception as e:
            print(f"❌ Error upserting vectors: {e}")
            return False
    
    def query_vectors(self, query_embedding: List[float], top_k: int = 3, include_metadata: bool = True) -> List[Dict]:
        """
        Query vectors from Pinecone
        
        Args:
            query_embedding: Embedding of the query text
            top_k: Number of results to return
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of matched vectors with scores and metadata
        """
        if not query_embedding:
            print("❌ Empty query embedding")
            return []
        
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=include_metadata
            )
            
            matches = []
            for match in results.matches:
                matches.append({
                    'text': match.metadata.get('text', ''),
                    'score': match.score,
                    'id': match.id,
                    'metadata': match.metadata if include_metadata else {}
                })
            
            return matches
            
        except Exception as e:
            print(f"❌ Error querying vectors: {e}")
            return []
    
    def get_index_stats(self):
        """
        Get statistics about the index
        
        Returns:
            Dictionary with index statistics
        """
        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            print(f"❌ Error getting index stats: {e}")
            return None
    
    def delete_all_vectors(self):
        """
        Delete all vectors from the index
        
        WARNING: This will delete all data!
        
        Returns:
            Boolean indicating success
        """
        try:
            # Delete all vectors
            self.index.delete(delete_all=True)
            print(f"✅ Deleted all vectors from index")
            return True
        except Exception as e:
            print(f"❌ Error deleting vectors: {e}")
            return False
