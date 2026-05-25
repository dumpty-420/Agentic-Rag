"""
Multi-Agent RAG with LangGraph over multiple Pinecone indexes

Architecture:
  - Orchestrator: routes query to domain(s): orders, products, support
  - Domain Researcher (fan-out): parallel per-domain retrieval via Send()
  - Aggregator (fan-in): merges domain results + builds unified context
  - Analyst: synthesizes retrieved snippets into structured findings
  - Critic: checks gaps/consistency and requests follow-up retrieval if needed
  - Synthesizer: produces final answer

Vector stores: Pinecone indexes per domain using sentence-transformers embeddings
LLM: Gemini 2.0 Flash

LangGraph features used:
  - add_messages: accumulates conversation message trace across nodes
  - Send(): fan-out to parallel domain research branches
  - Annotated reducers: operator.add for document aggregation
  - Conditional edges: critique-driven self-correction loop
"""
import os
import json
from typing import Dict, Any, List, TypedDict, Annotated, Sequence
import operator

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import AIMessage, HumanMessage, AnyMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.types import Send

from pinecone_multi_index import PineconeIndexManager, MULTI_AGENT_RAG_INDEX
from schemas import (
    LLMConfig,
    PineconeConfig,
    DomainEnum,
    DomainConfig,
    OrchestratorResponse,
    DomainResearchResult,
    RetrievedDocument,
    CriticVerdict,
    CriticCategory,
    AgentState,
    parse_llm_json,
)


load_dotenv()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DOMAIN CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DOMAIN_CONFIGS: Dict[DomainEnum, DomainConfig] = {
    DomainEnum.ORDERS: DomainConfig(namespace="orders", csv_path="knowledgebase/order.csv"),
    DomainEnum.PRODUCTS: DomainConfig(namespace="products", csv_path="knowledgebase/product.csv"),
    DomainEnum.SUPPORT: DomainConfig(namespace="support", csv_path="knowledgebase/support_catalogue.csv"),
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STATE (TypedDict for LangGraph, mirrors AgentState Pydantic model)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MultiAgentState(TypedDict):
    """
    LangGraph state using TypedDict with annotated reducers.

    - messages: uses add_messages for conversation trace accumulation
    - retrieved_docs: uses operator.add for fan-in from parallel branches
    - domain_results: uses operator.add for typed fan-in accumulation
    """
    question: str
    domains: List[str]
    messages: Annotated[Sequence[AnyMessage], add_messages]
    retrieved_docs: Annotated[List[Any], operator.add]
    domain_results: Annotated[List[dict], operator.add]
    context: str
    findings: str
    critique: str
    critic_report: Dict[str, Any]
    answer: str
    loop_count: int


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_llm(config: LLMConfig | None = None) -> ChatGoogleGenerativeAI:
    """Create an LLM instance, optionally from a validated LLMConfig."""
    if config:
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.api_key or os.getenv("GOOGLE_API_KEY"),
            temperature=config.temperature,
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
    )


def get_embeddings() -> SentenceTransformerEmbeddings:
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def ensure_indexes() -> None:
    mgr = PineconeIndexManager()
    mgr.ensure_index()


def ingest_csvs(knowledge_dir: str = "knowledgebase") -> None:
    """Ingest CSVs into the consolidated Pinecone index using namespaces."""
    ensure_indexes()
    embeddings = get_embeddings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for domain, cfg in DOMAIN_CONFIGS.items():
        csv_path = os.path.join(knowledge_dir, os.path.basename(cfg.csv_path))
        if not os.path.exists(csv_path):
            print(f"⚠️ Missing CSV: {csv_path}")
            continue
        print(f"📥 Ingesting {csv_path} → Namespace: {cfg.namespace}")
        loader = CSVLoader(file_path=csv_path)
        docs = loader.load()
        chunks = splitter.split_documents(docs)

        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=MULTI_AGENT_RAG_INDEX,
            namespace=cfg.namespace,
        )
        print(f"✅ Ingested {len(chunks)} chunks into namespace: {cfg.namespace}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MULTI-AGENT RAG SYSTEM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MultiAgentRAG:
    """
    Multi-Agent RAG system with:
    - Fan-out domain research (orchestrator → parallel domain_researcher nodes via Send())
    - Fan-in aggregation (all domain results merge into analyst)
    - add_messages for full LLM conversation trace
    - Conditional critique loop for self-correction
    """

    def __init__(self, llm_config: LLMConfig | None = None):
        self.llm_config = llm_config or LLMConfig()
        self.llm = get_llm(self.llm_config)
        self.embeddings = get_embeddings()
        ensure_indexes()

        # Build retrievers for EACH namespace within the SAME index
        self.retrievers: Dict[str, Any] = {}
        for domain, cfg in DOMAIN_CONFIGS.items():
            vs = PineconeVectorStore(
                index_name=MULTI_AGENT_RAG_INDEX,
                embedding=self.embeddings,
                namespace=cfg.namespace,
            )
            self.retrievers[domain.value] = vs.as_retriever(
                search_kwargs={"k": cfg.retriever_k}
            )

        # Legacy aliases for backward compatibility
        self.orders_ret = self.retrievers.get("orders")
        self.products_ret = self.retrievers.get("products")
        self.support_ret = self.retrievers.get("support")

        self.max_loops = 3
        self.graph = self._build_graph()

    # ━━━━━━━━━━━━━━━━━  NODES  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def orchestrate(self, state: MultiAgentState) -> Dict[str, Any]:
        """
        Classify user question into domains using structured output.
        Appends an AIMessage to the conversation trace via add_messages.
        """
        question = state["question"]
        system = (
            "You are the Orchestrator. Classify the user question into one or more of "
            "these domains: orders, products, support. Output a JSON with keys:\n"
            '  "domains": array of domain strings\n'
            '  "confidence": float 0-1\n'
            '  "reasoning": brief explanation\n'
            "If unclear, pick the most likely one."
        )
        prompt = f"{system}\n\nQuestion: {question}\nRespond with JSON only."
        resp = await self.llm.ainvoke(prompt)

        # Parse through Pydantic
        domains: List[str] = []
        try:
            cleaned = parse_llm_json(resp.content)
            orchestrator_resp = OrchestratorResponse.model_validate_json(cleaned)
            domains = [d.value for d in orchestrator_resp.domains]
        except Exception:
            try:
                data = json.loads(parse_llm_json(resp.content))
                if isinstance(data, dict) and isinstance(data.get("domains"), list):
                    domains = [d for d in data["domains"] if d in ["orders", "products", "support"]]
            except Exception:
                pass

        if not domains:
            domains = ["products"]

        # Track in message history via add_messages
        return {
            "domains": domains,
            "messages": [
                HumanMessage(content=f"[Orchestrator Input] {question}"),
                AIMessage(content=f"[Orchestrator] Routed to domains: {domains}"),
            ],
            "loop_count": 0,
        }

    async def domain_research(self, state: MultiAgentState) -> Dict[str, Any]:
        """
        Retrieve documents from relevant domain indexes.
        Called once — iterates over all classified domains.
        Fan-out version (via Send) is in the advanced agentic_multi_rag.py.
        """
        question = state["question"]
        domains: List[str] = state.get("domains", [])
        all_docs = []
        domain_results = []

        for domain_name in domains:
            retriever = self.retrievers.get(domain_name)
            if not retriever:
                continue
            docs = await retriever.ainvoke(question)
            all_docs.extend(docs)

            # Build typed result
            typed_docs = []
            for d in docs:
                typed_docs.append(RetrievedDocument(
                    content=d.page_content,
                    source=d.metadata.get("source", "Unknown"),
                    domain=DomainEnum(domain_name),
                ).model_dump())

            result = DomainResearchResult(
                domain=DomainEnum(domain_name),
                documents=[RetrievedDocument.model_validate(td) for td in typed_docs],
                context_snippet="\n".join(d.page_content[:200] for d in docs),
            ).model_dump()
            domain_results.append(result)

        # Build context from all docs with source metadata
        context_parts = []
        for d in all_docs:
            src = d.metadata.get("source", "Unknown")
            part = f"--- [Source: {src}] ---\n{d.page_content}"
            context_parts.append(part)

        context = "\n\n".join(context_parts)

        return {
            "retrieved_docs": all_docs,
            "domain_results": domain_results,
            "context": context,
            "messages": [
                AIMessage(content=f"[Researcher] Retrieved {len(all_docs)} docs across {len(domains)} domains"),
            ],
        }

    async def analyze(self, state: MultiAgentState) -> Dict[str, Any]:
        """Extract key findings from retrieved context with citations."""
        question = state["question"]
        context = state.get("context", "")
        sys = (
            "You are the Lead Analyst. Read the retrieved context and extract key facts "
            "relevant to the question. Every single fact must include a citation in brackets "
            "matching the source defined in the context (e.g., [Source: order.csv])."
        )
        prompt = f"{sys}\n\nQuestion: {question}\nContext:\n{context}\n\nFindings:"
        resp = await self.llm.ainvoke(prompt)

        return {
            "findings": resp.content,
            "messages": [
                AIMessage(content=f"[Analyst] Produced findings ({len(resp.content)} chars)"),
            ],
        }

    async def critique(self, state: MultiAgentState) -> Dict[str, Any]:
        """Review findings for accuracy and completeness."""
        question = state["question"]
        findings = state.get("findings", "")
        sys = (
            "You are the Quality Critic. Assess if findings answer the question fully. "
            "Ensure citations are present. Return JSON with keys: "
            '"is_verified" (bool), "category" (DATA_GAP|LOGIC_ERROR|HALLUCINATION|NONE), '
            '"feedback" (str), "missing_info" (list), "suggestion" (str).'
        )
        prompt = f"{sys}\n\nQuestion: {question}\nFindings:\n{findings}\n\nCritique (JSON only):"
        resp = await self.llm.ainvoke(prompt)

        # Parse through Pydantic CriticVerdict
        try:
            cleaned = parse_llm_json(resp.content)
            verdict = CriticVerdict.model_validate_json(cleaned)
        except Exception:
            verdict = CriticVerdict(
                is_verified=True,
                category=CriticCategory.NONE,
                feedback=resp.content[:500],
                suggestion="",
            )

        return {
            "critique": verdict.feedback,
            "critic_report": verdict.model_dump(),
            "messages": [
                AIMessage(content=f"[Critic] Verified={verdict.is_verified}, Category={verdict.category.value}"),
            ],
        }

    async def synthesize(self, state: MultiAgentState) -> Dict[str, Any]:
        """Synthesize final grounded answer from context and findings."""
        question = state["question"]
        context = state.get("context", "")
        findings = state.get("findings", "")
        critique = state.get("critique", "")

        # Also check typed critic report
        report = state.get("critic_report", {})
        if not critique and report:
            critique = f"Verification: {report.get('is_verified')}. Feedback: {report.get('feedback')}"

        sys = (
            "You are the final Knowledge Architect. Using the context, findings, and critique, "
            "produce a detailed, helpful answer. Maintain all source citations from the findings."
        )
        prompt = (
            f"{sys}\n\nQuestion: {question}\nContext:\n{context}\n\n"
            f"Findings:\n{findings}\n\nCritique:\n{critique}\n\nFinal Answer:"
        )
        resp = await self.llm.ainvoke(prompt)

        return {
            "answer": resp.content,
            "messages": [
                AIMessage(content=f"[Synthesizer] Final answer produced ({len(resp.content)} chars)"),
            ],
        }

    # ━━━━━━━━━━━━━ CONDITIONAL EDGE ━━━━━━━━━━━━━━━━━━━━━━━

    def _should_loop_back(self, state: MultiAgentState) -> str:
        """Decide if the critique loop should continue or finalize."""
        report = state.get("critic_report", {})
        loop = state.get("loop_count", 0)

        if report.get("is_verified", True) or loop >= self.max_loops:
            return "synthesizer"

        category = report.get("category", "NONE")
        if category in ("DATA_GAP", "LOGIC_ERROR", "HALLUCINATION"):
            return "researcher"  # Loop back for more research

        return "synthesizer"

    # ━━━━━━━━━━━━━  FAN-OUT DISPATCHER  ━━━━━━━━━━━━━━━━━━━━

    def _dispatch_domain_research(self, state: MultiAgentState) -> list:
        """
        Fan-out: Generate Send() objects to dispatch parallel
        domain-specific research nodes.

        One Send() per classified domain → each runs domain_research_single.
        """
        question = state["question"]
        domains = state.get("domains", ["products"])

        sends = []
        for domain_name in domains:
            sends.append(
                Send(
                    "domain_research_single",
                    {
                        "question": question,
                        "target_domain": domain_name,
                        "domains": domains,
                        "messages": [],
                        "retrieved_docs": [],
                        "domain_results": [],
                        "context": "",
                        "findings": "",
                        "critique": "",
                        "critic_report": {},
                        "answer": "",
                        "loop_count": state.get("loop_count", 0),
                    },
                )
            )

        print(f"🔀 Fan-out: dispatching {len(sends)} parallel domain research branches")
        return sends

    async def domain_research_single(self, state: dict) -> Dict[str, Any]:
        """
        Single-domain research node — invoked per Send() branch.
        Receives a state with 'target_domain' specifying which domain to search.
        """
        question = state["question"]
        domain_name = state.get("target_domain", "products")
        retriever = self.retrievers.get(domain_name)

        if not retriever:
            print(f"⚠️ No retriever for domain: {domain_name}")
            return {"retrieved_docs": [], "domain_results": []}

        docs = await retriever.ainvoke(question)

        typed_docs = []
        for d in docs:
            typed_docs.append(RetrievedDocument(
                content=d.page_content,
                source=d.metadata.get("source", "Unknown"),
                domain=DomainEnum(domain_name),
            ).model_dump())

        result = DomainResearchResult(
            domain=DomainEnum(domain_name),
            documents=[RetrievedDocument.model_validate(td) for td in typed_docs],
            context_snippet="\n".join(d.page_content[:200] for d in docs),
        ).model_dump()

        print(f"  📚 [{domain_name}] Retrieved {len(docs)} documents")

        return {
            "retrieved_docs": docs,
            "domain_results": [result],
            "messages": [
                AIMessage(content=f"[DomainResearcher:{domain_name}] Retrieved {len(docs)} docs"),
            ],
        }

    async def aggregate_results(self, state: MultiAgentState) -> Dict[str, Any]:
        """
        Fan-in aggregator: merge all domain results into a unified context.
        Runs after all domain_research_single branches complete.
        """
        all_docs = state.get("retrieved_docs", [])
        domain_results = state.get("domain_results", [])

        context_parts = []
        for d in all_docs:
            if hasattr(d, "page_content"):
                src = d.metadata.get("source", "Unknown") if hasattr(d, "metadata") else "Unknown"
                part = f"--- [Source: {src}] ---\n{d.page_content}"
                context_parts.append(part)

        # Also incorporate domain result snippets
        for dr in domain_results:
            if isinstance(dr, dict) and dr.get("context_snippet"):
                domain_name = dr.get("domain", "unknown")
                context_parts.append(f"--- [Domain: {domain_name}] ---\n{dr['context_snippet']}")

        context = "\n\n".join(context_parts)

        return {
            "context": context,
            "messages": [
                AIMessage(
                    content=f"[Aggregator] Merged {len(all_docs)} docs from {len(domain_results)} domains"
                ),
            ],
        }

    # ━━━━━━━━━━━━━  GRAPH CONSTRUCTION  ━━━━━━━━━━━━━━━━━━━━

    def _build_graph(self):
        """
        Construct the LangGraph workflow with fan-out/fan-in edges.

        Topology:
                                    ┌─→ domain_research_single (orders) ──┐
          orchestrator → (Send()) ──┼─→ domain_research_single (products) ─┼─→ aggregator → analyst → critic ─┬→ synthesizer → END
                                    └─→ domain_research_single (support) ─┘                                  └→ researcher (loop)
        """
        workflow = StateGraph(MultiAgentState)

        # Register nodes
        workflow.add_node("orchestrator", self.orchestrate)
        workflow.add_node("domain_research_single", self.domain_research_single)
        workflow.add_node("aggregator", self.aggregate_results)
        workflow.add_node("analyst", self.analyze)
        workflow.add_node("critic", self.critique)
        workflow.add_node("synthesizer", self.synthesize)
        # Fallback researcher for critique loop-back
        workflow.add_node("researcher", self.domain_research)

        # Entry point
        workflow.set_entry_point("orchestrator")

        # Fan-out: orchestrator → Send() to multiple domain_research_single
        workflow.add_conditional_edges(
            "orchestrator",
            self._dispatch_domain_research,
            ["domain_research_single"],
        )

        # Fan-in: all domain_research_single → aggregator
        workflow.add_edge("domain_research_single", "aggregator")

        # Linear flow after aggregation
        workflow.add_edge("aggregator", "analyst")
        workflow.add_edge("analyst", "critic")

        # Conditional critique loop
        workflow.add_conditional_edges(
            "critic",
            self._should_loop_back,
            {
                "synthesizer": "synthesizer",
                "researcher": "researcher",
            },
        )

        # Loop-back researcher feeds back to analyst
        workflow.add_edge("researcher", "analyst")

        # Final
        workflow.add_edge("synthesizer", END)

        return workflow.compile()

    # ━━━━━━━━━━━━━  PUBLIC API  ━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def invoke(self, question: str) -> Dict[str, Any]:
        """Execute the full multi-agent RAG pipeline."""
        initial_state: MultiAgentState = {
            "question": question,
            "domains": [],
            "messages": [HumanMessage(content=question)],
            "retrieved_docs": [],
            "domain_results": [],
            "context": "",
            "findings": "",
            "critique": "",
            "critic_report": {},
            "answer": "",
            "loop_count": 0,
        }
        result = await self.graph.ainvoke(initial_state)
        return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def main():
    # Optional: initial ingest if needed
    if os.getenv("INGEST", "false").lower() in {"1", "true", "yes"}:
        ingest_csvs()

    rag = MultiAgentRAG()
    queries = [
        "Where is my order #1234 and expected delivery date?",
        "List features of product ABC-900 and its price",
        "How to reset my password and contact support?",
    ]
    for q in queries:
        print("\n" + "=" * 60)
        print(f"Q: {q}")
        out = await rag.invoke(q)
        print("--- Findings ---")
        print(out.get("findings", ""))
        print("--- Critique ---")
        print(out.get("critique", ""))
        print("--- Answer ---")
        print(out.get("answer", ""))
        print("\n--- Message Trace ---")
        for msg in out.get("messages", []):
            print(f"  [{msg.__class__.__name__}] {msg.content[:120]}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
