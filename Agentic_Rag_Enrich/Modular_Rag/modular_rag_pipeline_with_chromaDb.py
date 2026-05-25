"""
Modular RAG Pipeline - The orchestrator that connects all modules (ChromaDB Version)
"""
from embeddings_module import EmbeddingsModule
from document_preprocessor import DocumentPreprocessor
from vector_store_chromadb import VectorStore
from retriever import Retriever
from response_generator import ResponseGenerator
from typing import List, Dict
import os

class ModularRAGPipeline:
    def __init__(self, 
                 collection_name: str = "nikhil-agent-module-chromadb",
                 persist_directory: str = "./chroma_db"):
        """
        Initialize the modular RAG pipeline with ChromaDB
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data
        """
        print("\n" + "="*60)
        print("🚀 Initializing Modular RAG Pipeline (ChromaDB)")
        print("="*60)
        
        # Initialize all modules
        self.embeddings_module = EmbeddingsModule()
        self.preprocessor = DocumentPreprocessor()
        
        # ChromaDB Vector Store
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        
        self.retriever = Retriever(self.vector_store, self.embeddings_module)
        self.response_generator = ResponseGenerator()
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        print("="*60)
        print("✅ Modular RAG Pipeline Ready! (ChromaDB)")
        print("="*60 + "\n")
    
    def ingest_documents(self, file_path: str = "sample.txt"):
        """
        Ingest documents into ChromaDB
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Boolean indicating success
        """
        print(f"\n📄 Ingesting documents from: {file_path}")
        print("-" * 60)
        
        # Step 1: Load and preprocess the document
        chunks = self.preprocessor.process_from_file(file_path)
        
        if not chunks:
            print("❌ No chunks created from document")
            return False
        
        print(f"✅ Created {len(chunks)} chunks")
        
        # Step 2: Generate embeddings for chunks
        print("\n🔗 Generating embeddings...")
        embeddings = self.embeddings_module.generate_embeddings_batch(chunks)
        
        if not embeddings:
            print("❌ Failed to generate embeddings")
            return False
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        
        # Step 3: Upsert vectors to ChromaDB
        print("\n💾 Storing vectors in ChromaDB...")
        success = self.vector_store.upsert_vectors(chunks, embeddings)
        
        if success:
            print("✅ Document ingestion complete!")
            # Print collection stats
            stats = self.vector_store.get_index_stats()
            if stats:
                print(f"📊 Collection stats: {stats.get('total_documents', 0)} total documents")
        else:
            print("❌ Failed to store vectors")
        
        return success
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        """
        Query the RAG system
        
        Args:
            question: User's question
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary with answer and metadata
        """
        print(f"\n🔍 Processing Query: {question}")
        print("-" * 60)
        
        # Step 1: Retrieve relevant chunks
        print("📚 Retrieving relevant context...")
        retrieved_chunks = self.retriever.retrieve(question, top_k)
        
        if not retrieved_chunks:
            print("⚠️ No relevant context found")
            return {
                'answer': 'No relevant information found in the knowledge base.',
                'query': question,
                'sources': [],
                'scores': []
            }
        
        print(f"✅ Found {len(retrieved_chunks)} relevant chunks")
        
        # Display retrieval scores
        if retrieved_chunks:
            print("\n📊 Relevance Scores:")
            for i, chunk in enumerate(retrieved_chunks, 1):
                score = chunk.get('score', 0)
                preview = chunk.get('text', '')[:50] + "..."
                print(f"  {i}. Score: {score:.4f} - {preview}")
        
        # Step 2: Generate response
        print("\n💬 Generating answer...")
        result = self.response_generator.generate_response_with_scores(question, retrieved_chunks)
        
        print("\n✅ Query processed successfully!")
        print("=" * 60)
        
        return result
    
    def chat(self, question: str, top_k: int = 3):
        """
        Chat interface for the RAG system
        
        Args:
            question: User's question
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary with answer
        """
        result = self.query(question, top_k)
        
        print(f"\n❓ Question: {result.get('query', question)}")
        print(f"💡 Answer: {result.get('response', result.get('answer', ''))}")
        
        # Calculate average score if available
        scores = result.get('scores', [])
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"📊 Relevance: {avg_score:.4f} (avg)")
        else:
            print(f"📊 Relevance: {result.get('avg_score', 0):.4f} (avg)")
            
        print("=" * 60)
        
        return result
    
    def get_pipeline_info(self) -> Dict:
        """
        Get information about the pipeline components
        
        Returns:
            Dictionary with pipeline information
        """
        stats = self.vector_store.get_index_stats()
        
        return {
            'embeddings_model': self.embeddings_module.model_name,
            'chunk_size': self.preprocessor.chunk_size,
            'chunk_overlap': self.preprocessor.chunk_overlap,
            'collection_name': self.collection_name,
            'persist_directory': self.persist_directory,
            'total_documents': stats.get('total_documents', 0) if stats else 0,
            'llm_model': self.response_generator.model_name,
            'temperature': self.response_generator.temperature
        }
    
    def clear_database(self):
        """
        Clear all vectors from ChromaDB collection
        
        WARNING: This will delete all data!
        """
        print("\n⚠️  Clearing ChromaDB collection...")
        success = self.vector_store.delete_all_vectors()
        if success:
            print("✅ Database cleared successfully!")
        else:
            print("❌ Failed to clear database")
        return success


# Example usage
if __name__ == "__main__":
    # Initialize the pipeline with ChromaDB
    rag = ModularRAGPipeline(
        collection_name="nikhil-rag-chromadb",
        persist_directory="./chroma_data"
    )
    
    # Ingest documents
    print("\n📚 Starting document ingestion...")
    rag.ingest_documents("sample.txt")
    
    # Show pipeline info
    info = rag.get_pipeline_info()
    print(f"\n📋 Pipeline Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Example queries
    print("\n\n" + "="*60)
    print("🤖 Starting RAG Queries")
    print("="*60)
    
    queries = [
        "what is gravitational wave?"
    ]
    
    for query in queries:
        rag.chat(query)
        print("\n")