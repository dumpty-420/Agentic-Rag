"""
Embeddings Module - Uses sentence-transformers (all-MiniLM-L6-v2)
"""
from typing import List, Optional

from sentence_transformers import SentenceTransformer


class SentenceEmbeddingsModule:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embeddings module with sentence-transformers

        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        # all-MiniLM-L6-v2 outputs 384-d vectors
        self._dimension = 384
        print(f"✅ Sentence embeddings initialized with model: {model_name}")

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True).tolist()
            return embedding
        except Exception as e:
            print(f"❌ Error generating sentence embedding: {e}")
            return None

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True).tolist()
            return embeddings
        except Exception as e:
            print(f"❌ Error generating batch sentence embeddings: {e}")
            return []

    def get_embedding_dimension(self) -> int:
        return self._dimension


