"""
Critic Module - Multi-Domain Reflection and Verification with Pydantic Validation
"""
import os
import json
from typing import Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from schemas import (
    LLMConfig,
    CriticVerdict,
    CriticCategory,
    parse_llm_json,
)

load_dotenv()


class MultiCritic:
    def __init__(
        self,
        config: LLMConfig | None = None,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.2,
    ):
        """
        Initialize the multi-domain critic.

        Args:
            config: Optional LLMConfig for validated initialization.
            model_name: Fallback model name.
            temperature: Fallback temperature.
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

        self.critic_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a meticulous Quality Auditor for a Multi-Agentic RAG system.
Evaluate the correctness of the answer based on provided context and findings.

Categorize any failures into exactly one of these:
- DATA_GAP: Important info is missing from context. Suggestion should target RESEARCH.
- LOGIC_ERROR: Info is in context but the reasoning/plan was flawed. Suggestion should target RE-PLANNING.
- HALLUCINATION: Answer makes claims not in the context. Suggestion should target ANALYSIS.
- NONE: Answer is perfect and verified.

Returns a JSON object:
{{
  "is_verified": boolean,
  "category": "DATA_GAP" | "LOGIC_ERROR" | "HALLUCINATION" | "NONE",
  "feedback": "detailed reason",
  "missing_info": ["list", "of", "specific", "facts"],
  "suggestion": "how to fix"
}}
"""),
            ("user", "Question: {query}\n\nContext:\n{context}\n\nFindings:\n{findings}\n\nFinal Answer:\n{answer}")
        ])

        print(f"✅ Multi-Critic initialized with model: {self.config.model_name}")

    async def verify(
        self,
        query: str,
        context: str,
        findings: str,
        answer: str,
        domains: List[str],
    ) -> Dict:
        """
        Reflect on query coverage across multiple domains (Async).

        Parses the LLM verdict through CriticVerdict Pydantic model.

        Args:
            query: Original user question.
            context: Retrieved and reranked context.
            findings: Analyst's extracted findings.
            answer: Current synthesized answer.
            domains: Active domain name strings.

        Returns:
            Dict representation of CriticVerdict (or safe fallback).
        """
        try:
            print(f"🧐 Global Multi-Critic evaluating answer for domains: {domains}")
            chain = self.critic_prompt | self.llm
            response = await chain.ainvoke({
                "query": query,
                "context": context,
                "findings": findings,
                "answer": answer,
                "domains": ", ".join(domains),
            })

            cleaned = parse_llm_json(response.content)

            # Validate through Pydantic
            try:
                verdict = CriticVerdict.model_validate_json(cleaned)
            except Exception:
                # Fallback: manual parse with validation
                raw = json.loads(cleaned)
                verdict = CriticVerdict(
                    is_verified=raw.get("is_verified", False),
                    category=CriticCategory(raw.get("category", "NONE")),
                    feedback=raw.get("feedback", ""),
                    missing_info=raw.get("missing_info", []),
                    suggestion=raw.get("suggestion", ""),
                )

            if verdict.is_verified:
                print("✅ Answer verified by Global Critic")
            else:
                print(f"❌ Answer failed verification [{verdict.category.value}]: {verdict.feedback}")

            return verdict.model_dump()

        except Exception as e:
            print(f"⚠️ Global Critic error: {e}")
            fallback = CriticVerdict(
                is_verified=True,
                category=CriticCategory.NONE,
                feedback=f"Critic Error: {e}",
                missing_info=[],
                suggestion="Proceed with current answer",
            )
            return fallback.model_dump()

    async def verify_typed(
        self,
        query: str,
        context: str,
        findings: str,
        answer: str,
        domains: List[str],
    ) -> CriticVerdict:
        """
        Same as verify() but returns the typed CriticVerdict directly.
        Used by the advanced agentic graph for typed state updates.
        """
        raw = await self.verify(query, context, findings, answer, domains)
        return CriticVerdict.model_validate(raw)


if __name__ == "__main__":
    # Test stub
    pass
