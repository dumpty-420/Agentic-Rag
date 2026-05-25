"""
Vector Store Module - Handles ChromaDB operations (upsert/query)
"""
import chromadb
from chromadb.config import Settings
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
import uuid

load_dotenv()

class VectorStore:
    def __init__(self, collection_name: str, persist_directory: str = "./chroma_db"):
        """
        Initialize the vector store with ChromaDB
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = "nikhil-agent-module-chromadb"
        self.persist_directory = "./chroma_db"
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"✅ Connected to existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "RAG system document chunks"}
            )
            print(f"✅ Created new collection: {collection_name}")
    
    def upsert_vectors(self, texts: List[str], embeddings: List[List[float]], metadata: List[Dict] = None):
        """
        Insert or update vectors in ChromaDB
        
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
        
        # Generate unique IDs
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        try:
            # Add metadata to each document
            enhanced_metadata = []
            for i, meta in enumerate(metadata):
                enhanced_metadata.append({
                    'chunk_index': i,
                    'text_length': len(texts[i]),
                    **meta
                })
            
            # Upsert to ChromaDB
            self.collection.upsert(
                embeddings=embeddings,
                documents=texts,
                metadatas=enhanced_metadata,
                ids=ids
            )
            
            print(f"✅ Upserted {len(texts)} vectors to ChromaDB collection")
            return True
            
        except Exception as e:
            print(f"❌ Error upserting vectors to ChromaDB: {e}")
            return False
    
    def query_vectors(self, query_embedding: List[float], top_k: int = 3, include_metadata: bool = True) -> List[Dict]:
        """
        Query vectors from ChromaDB
        
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
            # ChromaDB expects a list of embeddings for query
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            matches = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (ChromaDB returns L2 distance)
                    # Lower distance = higher similarity
                    similarity_score = 1 / (1 + distance) if distance > 0 else 1.0
                    
                    matches.append({
                        'text': doc,
                        'score': similarity_score,
                        'id': results['ids'][0][i] if results['ids'] else f"result_{i}",
                        'metadata': metadata if include_metadata else {}
                    })
            
            return matches
            
        except Exception as e:
            print(f"❌ Error querying ChromaDB: {e}")
            return []
    
    def get_index_stats(self):
        """
        Get statistics about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            # ChromaDB doesn't have a direct stats method, so we count
            count = self.collection.count()
            return {
                "total_vectors": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            print(f"❌ Error getting collection stats: {e}")
            return None
    
    def delete_all_vectors(self):
        """
        Delete all vectors from the collection
        
        WARNING: This will delete all data!
        
        Returns:
            Boolean indicating success
        """
        try:
            # Get all IDs and delete them
            results = self.collection.get()
            if results['ids']:
                self.collection.delete(ids=results['ids'])
            print(f"✅ Deleted all vectors from collection")
            return True
        except Exception as e:
            print(f"❌ Error deleting vectors: {e}")
            return False
    
    def get_collection_info(self):
        """
        Get detailed information about the collection
        
        Returns:
            Dictionary with collection information
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "total_documents": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            print(f"❌ Error getting collection info: {e}")
            return None