"""
Pinecone multi-index setup and helpers.
"""
import os
from typing import List, Dict

from dotenv import load_dotenv
from pinecone import Pinecone


load_dotenv()


DEFAULT_INDEX_CONFIGS: List[Dict] = [
    {"name": "orders-index", "dimension": 384, "metric": "cosine"},
    {"name": "products-index", "dimension": 384, "metric": "cosine"},
    {"name": "support-index", "dimension": 384, "metric": "cosine"},
]


class PineconeIndexManager:
    def __init__(self, api_key_env: str = "PC_API_KEY"):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError("PC_API_KEY not set in environment")
        self.pc = Pinecone(api_key=api_key)

    def ensure_indexes(self, index_configs: List[Dict] = None):
        configs = index_configs or DEFAULT_INDEX_CONFIGS

        # pinecone>=5 returns response with .names(); older returns list/dicts
        try:
            existing = set(self.pc.list_indexes().names())  # type: ignore[attr-defined]
        except Exception:
            try:
                existing = {idx["name"] for idx in self.pc.list_indexes()}
            except Exception:
                existing = set()

        for cfg in configs:
            name = cfg["name"]
            dimension = cfg["dimension"]
            metric = cfg.get("metric", "cosine")

            if name in existing:
                print(f"✅ Pinecone index exists: {name}")
                continue

            print(f"🛠️ Creating Pinecone index: {name} (dim={dimension}, metric={metric})")
            cloud = os.getenv("PC_CLOUD", "aws")
            region = os.getenv("PC_REGION", "us-east-1")
            try:
                self.pc.create_index(
                    name=name,
                    dimension=dimension,
                    metric=metric,
                    spec={"serverless": {"cloud": cloud, "region": region}},
                )
            except TypeError:
                # Fallback for older SDKs without serverless spec
                self.pc.create_index(name=name, dimension=dimension, metric=metric)
            print(f"✅ Created index: {name}")


