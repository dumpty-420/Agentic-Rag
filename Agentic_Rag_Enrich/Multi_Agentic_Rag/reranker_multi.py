"""
Reranker Module - Multi-Index Relevance Scoring with Pydantic Validation
"""
import os
import json
import asyncio
from typing import Any, List, Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from schemas import LLMConfig, RerankerScore, RetrievedDocument, DomainEnum, parse_llm_json

load_dotenv()


class MultiReranker:
    def __init__(
        self,
        config: LLMConfig | None = None,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.1,
        threshold: float = 0.3,
    ):
        """
        Initialize the global multi-index reranker.

        Args:
            config: Optional LLMConfig for validated initialization.
            model_name: Fallback model name.
            temperature: Fallback temperature.
            threshold: Minimum relevance score to keep a chunk.
        """
        if config is not None:
            self.config = config
        else:
            self.config = LLMConfig(
                model_name=model_name,
                temperature=temperature,
                api_key=os.getenv("GOOGLE_API_KEY"),
            )

        self.model_name = self.config.model_name
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.model_name,
            google_api_key=self.config.api_key or os.getenv("GOOGLE_API_KEY"),
            temperature=self.config.temperature,
        )
        self.threshold = threshold

        self.rerank_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a highly capable AI specialized in cross-domain information relevance. 
Your task is to rank the relevance of a retrieved context chunk to a user's question.

Guidelines:
- Score the relevance from 0 to 1.
- Provide a brief justification.
- Return only a JSON object: 'relevance_score' (float) and 'justification' (string)."""),
            ("user", "Question: {query}\n\nContext Chunk:\n{chunk}")
        ])

        print(f"✅ Multi-Reranker initialized with model: {self.config.model_name} (Threshold: {threshold})")

    async def score_chunks(
        self,
        query: str,
        chunks: List[Any],
        top_n: int = 5,
    ) -> List[Dict]:
        """
        Rerank chunks retrieved from multiple indexes (Async).

        Parses each LLM score response through RerankerScore Pydantic model.

        Args:
            query: The user's question.
            chunks: List of LangChain Document objects or similar.
            top_n: Maximum number of top chunks to return.

        Returns:
            List of scored chunk dicts, sorted by relevance.
        """
        print(f"📊 Global Multi-Reranking {len(chunks)} chunks for query: '{query}'")

        async def _score_single(chunk: Any) -> Dict | None:
            text = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
            metadata = chunk.metadata if hasattr(chunk, "metadata") else {}
            try:
                chain = self.rerank_prompt | self.llm
                response = await chain.ainvoke({"query": query, "chunk": text})

                cleaned = parse_llm_json(response.content)

                # Validate through Pydantic
                try:
                    score_obj = RerankerScore.model_validate_json(cleaned)
                except Exception:
                    # Fallback: manual parse
                    raw = json.loads(cleaned)
                    score_obj = RerankerScore(
                        relevance_score=float(raw.get("relevance_score", 0)),
                        justification=raw.get("justification", ""),
                    )

                if score_obj.relevance_score >= self.threshold:
                    return {
                        "text": text,
                        "score": score_obj.relevance_score,
                        "justification": score_obj.justification,
                        "metadata": metadata,
                    }
            except Exception as e:
                print(f"⚠️ Multi-Scoring error: {e}")
            return None

        # Parallelize scoring
        tasks = [_score_single(c) for c in chunks]
        results = await asyncio.gather(*tasks)
        scored_chunks = [r for r in results if r is not None]

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
        final_chunks = scored_chunks[:top_n]

        print(f"✅ Global Rerank completed. Kept {len(final_chunks)} high-quality chunks.")
        return final_chunks

    async def score_to_documents(
        self,
        query: str,
        chunks: List[Any],
        domain: DomainEnum,
        top_n: int = 5,
    ) -> List[RetrievedDocument]:
        """
        Score chunks and return as validated RetrievedDocument models.

        Args:
            query: The user's question.
            chunks: Raw document objects.
            domain: The DomainEnum these chunks belong to.
            top_n: Maximum results.

        Returns:
            List of RetrievedDocument Pydantic models.
        """
        scored = await self.score_chunks(query, chunks, top_n)
        documents = []
        for item in scored:
            doc = RetrievedDocument(
                content=item["text"],
                source=item["metadata"].get("source", "Unknown"),
                domain=domain,
                relevance_score=item["score"],
            )
            documents.append(doc)
        return documents


if __name__ == "__main__":
    # Test stub
    pass
