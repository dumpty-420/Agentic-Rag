"""
Modular RAG Pipeline - The orchestrator that connects all modules
"""
from embeddings_module import EmbeddingsModule
from document_preprocessor import DocumentPreprocessor
from vector_store import VectorStore

from retriever import Retriever
from response_generator import ResponseGenerator
from typing import List, Dict
import os

class ModularRAGPipeline:
    def __init__(self, index_name: str = None):
        """
        Initialize the modular RAG pipeline
        
        Args:
            index_name: Name of the Pinecone index to use
        """
        if index_name is None:
            index_name = os.getenv("PINECONE_INDEX_NAME", "nikhil-agent-module-gemini")
            
        print("\n" + "="*60)
        print("🚀 Initializing Modular RAG Pipeline")
        print("="*60)
        
        # Initialize all modules
        self.embeddings_module = EmbeddingsModule()
        self.preprocessor = DocumentPreprocessor()
        self.vector_store = VectorStore(index_name)
        self.retriever = Retriever(self.vector_store, self.embeddings_module)
        self.response_generator = ResponseGenerator()
        
        print("="*60)
        print("✅ Modular RAG Pipeline Ready!")
        print("="*60 + "\n")
    
    def ingest_documents(self, file_path: str = "sample.txt"):
        """
        Ingest documents into the vector store
        
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
        
        # Step 3: Upsert vectors to Pinecone
        print("\n\n💾 Storing vectors in Pinecone...")
        success = self.vector_store.upsert_vectors(chunks, embeddings)
        
        if success:
            print("✅ Document ingestion complete!")
            # Print index stats
            stats = self.vector_store.get_index_stats()
            if stats:
                print(f"📊 Index stats: {stats.total_vector_count} total vectors")
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
        print(f"📊 Relevance: {result.get('avg_score', 0):.4f} (avg)")
        print("=" * 60)
        
        return result
    
    def get_pipeline_info(self) -> Dict:
        """
        Get information about the pipeline components
        
        Returns:
            Dictionary with pipeline information
        """
        return {
            'embeddings_model': self.embeddings_module.model_name,
            'chunk_size': self.preprocessor.chunk_size,
            'chunk_overlap': self.preprocessor.chunk_overlap,
            'index_name': self.vector_store.index_name,
            'llm_model': self.response_generator.model_name,
            'temperature': self.response_generator.temperature
        }


# Example usage
if __name__ == "__main__":
    # Initialize the pipeline
    rag = ModularRAGPipeline()
    
    # Ingest documents
    print("\n📚 Starting document ingestion...")
    rag.ingest_documents("sample.txt")
    
    # Example queries
    print("\n\n" + "="*60)
    print("🤖 Starting RAG Queries")
    print("="*60)
    
    queries = [
        "what is masked multi-head attention and where is it used?"
    ]
    
    for query in queries:
        rag.chat(query)
        print("\n")
