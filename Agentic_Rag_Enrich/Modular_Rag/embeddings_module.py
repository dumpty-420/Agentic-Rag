"""
Embeddings Module - Handles embedding generation using Google Gemini
"""
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

class EmbeddingsModule:
    def __init__(self, model_name="models/gemini-embedding-001"):
        """
        Initialize the embeddings module
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        print(f"✅ Embeddings module initialized with model: {model_name}")
    
    def generate_embedding(self, text: str):
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding
        """
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            return None
    
    def generate_embeddings_batch(self, texts: list):
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        try:
            embeddings = self.embeddings.embed_documents(texts)
            return embeddings
        except Exception as e:
            print(f"❌ Error generating batch embeddings: {e}")
            return []
    
    def get_embedding_dimension(self):
        """
        Get the dimension of embeddings produced by this model
        
        Returns:
            Integer representing embedding dimension
        """
        # Test embedding to get dimension
        test_embedding = self.generate_embedding("test")
        if test_embedding:
            return len(test_embedding)
        return None
