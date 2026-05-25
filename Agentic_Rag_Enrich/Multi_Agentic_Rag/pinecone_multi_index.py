"""
Pinecone multi-index setup and helpers.
Uses Pydantic PineconeConfig for validated configuration.
"""
import os
from typing import Optional

from dotenv import load_dotenv
from pinecone import Pinecone

from schemas import PineconeConfig


load_dotenv()


# Using the existing 'orders-index' as the consolidated container to avoid project limits
MULTI_AGENT_RAG_INDEX = "orders-index"
INDEX_DIMENSION = 384
INDEX_METRIC = "cosine"


class PineconeIndexManager:
    def __init__(
        self,
        config: Optional[PineconeConfig] = None,
        api_key_env: str = "PC_API_KEY",
    ):
        """
        Initialize the Pinecone index manager.

        Args:
            config: Optional PineconeConfig with validated settings.
                    Falls back to environment variables if not provided.
            api_key_env: Environment variable name for the API key.
        """
        if config is not None:
            self.config = config
        else:
            # Build config from environment defaults
            self.config = PineconeConfig(
                index_name=MULTI_AGENT_RAG_INDEX,
                dimension=INDEX_DIMENSION,
                metric=INDEX_METRIC,
                api_key=os.getenv(api_key_env),
                cloud=os.getenv("PC_CLOUD", "aws"),
                region=os.getenv("PC_REGION", "us-east-1"),
            )

        api_key = self.config.api_key or os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} not set in environment and no api_key in config")
        self.pc = Pinecone(api_key=api_key)

    def ensure_index(self, name: Optional[str] = None) -> None:
        """Ensure the single consolidated index exists."""
        target = name or self.config.index_name

        # pinecone>=5 returns response with .names(); older returns list/dicts
        try:
            existing = set(self.pc.list_indexes().names())  # type: ignore[attr-defined]
        except Exception:
            try:
                existing = {idx["name"] for idx in self.pc.list_indexes()}
            except Exception:
                existing = set()

        if target in existing:
            print(f"✅ Pinecone index exists: {target}")
            return

        print(f"🛠️ Creating consolidated Pinecone index: {target} (dim={self.config.dimension})")
        try:
            self.pc.create_index(
                name=target,
                dimension=self.config.dimension,
                metric=self.config.metric,
                spec={
                    "serverless": {
                        "cloud": self.config.cloud,
                        "region": self.config.region,
                    }
                },
            )
        except TypeError:
            # Fallback for older SDKs
            self.pc.create_index(
                name=target,
                dimension=self.config.dimension,
                metric=self.config.metric,
            )
        print(f"✅ Created consolidated index: {target}")

    def ensure_indexes(self, configs=None) -> None:
        """Legacy support for backward compatibility."""
        self.ensure_index()
