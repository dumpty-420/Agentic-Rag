"""
Embeddings Module - Handles embedding generation using Google Gemini
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()


class EmbeddingsModule:
    def __init__(self, model_name="models/gemini-embedding-001", target_dimension=3072):
        """
        Initialize the embeddings module

        Args:
            model_name: Name of the embedding model to use
            target_dimension: Target dimension to truncate vectors to (e.g., for Pinecone)
        """
        self.model_name = model_name
        self.target_dimension = target_dimension
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name, google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        print(
            f"✅ Embeddings module initialized with model: {model_name} (target_dim: {target_dimension})"
        )

    def generate_embedding(self, text: str):
        """
        Generate embedding for a single text

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding (truncated if needed)
        """
        try:
            embedding = self.embeddings.embed_query(text)
            # Truncate to target dimension if it's larger
            if embedding and len(embedding) > self.target_dimension:
                embedding = embedding[: self.target_dimension]
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
            List of embeddings (truncated if needed)
        """
        try:
            embeddings = self.embeddings.embed_documents(texts)
            # Truncate each embedding in the batch
            if embeddings and len(embeddings[0]) > self.target_dimension:
                embeddings = [emb[: self.target_dimension] for emb in embeddings]
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
