"""
Planner Module - Multi-Domain Task Decomposition with Pydantic Validation
"""
import os
import json
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from schemas import (
    LLMConfig,
    PlannerTask,
    PlannerResponse,
    DomainEnum,
    TaskPriority,
    parse_llm_json,
)

load_dotenv()


class MultiPlanner:
    def __init__(
        self,
        config: LLMConfig | None = None,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.2,
    ):
        """
        Initialize the multi-domain planner.

        Args:
            config: Optional LLMConfig for validated initialization.
            model_name: Fallback model name if config is not provided.
            temperature: Fallback temperature if config is not provided.
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

        self.planner_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert query planner for a Multi-Agentic RAG system across these domains: {domains}.
Your goal is to decompose a complex query into a sequence of atomic tasks.

Guidelines:
- Identify dependencies: If Task B needs information from Task A, mark Task B as 'depends_on' Task A's ID.
- Assign a priority to each task: "critical", "high", "medium", or "low".
- Example: 
  Query: "Find the product in order #1234 and its support manual"
  Tasks: 
  1. Find the product name/ID in order #1234 (Domain: orders, priority: critical)
  2. Get maintenance info for [Product from Task 1] (Domain: support, depends_on: [1], priority: high)

- Output ONLY a JSON object with the following structure:
{{
  "tasks": [
    {{ "id": 1, "query": "sub-question 1", "domain": "domain_name", "depends_on": [], "priority": "critical" }},
    {{ "id": 2, "query": "sub-question 2", "domain": "domain_name", "depends_on": [1], "priority": "high" }}
  ],
  "strategy": "describe your decomposition strategy briefly"
}}
"""),
            ("user", "Question: {query}")
        ])

        print(f"✅ Hierarchical Multi-Planner initialized (Model: {self.config.model_name})")

    async def decompose(
        self, query: str, active_domains: List[str]
    ) -> List[dict]:
        """
        Decompose a multi-domain query into a dependent task graph (Async).

        Args:
            query: The user's question.
            active_domains: List of domain name strings to plan across.

        Returns:
            List of task dicts validated via PlannerTask.
        """
        try:
            print(f"🧩 Multi-Planner Decomposing query: '{query}' for domains: {active_domains}")
            chain = self.planner_prompt | self.llm
            response = await chain.ainvoke({
                "query": query,
                "domains": ", ".join(active_domains),
            })

            cleaned = parse_llm_json(response.content)

            # Validate through Pydantic
            try:
                planner_resp = PlannerResponse.model_validate_json(cleaned)
                tasks = [t.model_dump() for t in planner_resp.tasks]
                print(f"✅ Generated {len(tasks)} hierarchical tasks (strategy: {planner_resp.strategy})")
                return tasks
            except Exception as parse_err:
                # Fallback: try raw JSON parsing and manual PlannerTask validation
                print(f"⚠️ Pydantic parse failed ({parse_err}), attempting manual parse...")
                raw_data = json.loads(cleaned)
                raw_tasks = raw_data.get("tasks", [])

                validated_tasks = []
                for rt in raw_tasks:
                    # Coerce domain to valid enum value
                    domain_val = rt.get("domain", active_domains[0] if active_domains else "products")
                    try:
                        DomainEnum(domain_val)
                    except ValueError:
                        domain_val = active_domains[0] if active_domains else "products"

                    task = PlannerTask(
                        id=rt.get("id", len(validated_tasks) + 1),
                        query=rt.get("query", query),
                        domain=DomainEnum(domain_val),
                        depends_on=rt.get("depends_on", []),
                        priority=TaskPriority(rt.get("priority", "medium")),
                    )
                    validated_tasks.append(task.model_dump())

                if validated_tasks:
                    print(f"✅ Manually validated {len(validated_tasks)} tasks")
                    return validated_tasks

                raise parse_err  # Re-raise if nothing worked

        except Exception as e:
            print(f"⚠️ Multi-Planner error: {e}. Falling back to single task.")
            fallback = PlannerTask(
                id=1,
                query=query,
                domain=DomainEnum(active_domains[0] if active_domains else "products"),
                depends_on=[],
                priority=TaskPriority.HIGH,
            )
            return [fallback.model_dump()]


if __name__ == "__main__":
    import asyncio

    async def test():
        planner = MultiPlanner()
        tasks = await planner.decompose(
            "Check my order #555 and reset my password",
            ["orders", "support"],
        )
        for t in tasks:
            print(t)

    asyncio.run(test())
