"""
Multi-Agentic RAG Orchestrator - Advanced Domain Management with LangGraph (Async)

Advanced LangGraph features:
  - Send() for fan-out one-to-many domain research
  - add_messages for full conversation trace
  - update_state for programmatic state corrections in critique loops
  - Annotated reducers (operator.add) for fan-in document aggregation
  - Conditional edges for self-correction routing
  - Pydantic models for all data contracts
"""
import os
import asyncio
from typing import Dict, Any, List, Sequence, Annotated
import operator

from langchain_core.messages import AIMessage, HumanMessage, AnyMessage, SystemMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.types import Send
from dotenv import load_dotenv

# Import the base modules
from multi_agent_rag import (
    MultiAgentRAG,
    MultiAgentState,
    get_llm,
    get_embeddings,
    ensure_indexes,
    DOMAIN_CONFIGS,
)

# Import new modular components
from planner_multi import MultiPlanner
from reranker_multi import MultiReranker
from critic_multi import MultiCritic
from memory_multi import MultiMemory

# Import Pydantic schemas
from schemas import (
    LLMConfig,
    DomainEnum,
    CriticVerdict,
    CriticCategory,
    DomainResearchResult,
    RetrievedDocument,
    PlannerTask,
    parse_llm_json,
)

load_dotenv()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EXTENDED STATE for Advanced Agentic System
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from typing import TypedDict


class AdvancedAgentState(TypedDict):
    """
    Extended state for the advanced multi-agentic workflow.
    Adds planner tasks, reranking, typed critic verdicts, and memory context
    on top of the base MultiAgentState fields.

    Reducers:
      - messages: add_messages (accumulates conversation trace)
      - retrieved_docs: operator.add (fan-in from parallel branches)
      - domain_results: operator.add (typed fan-in)
    """
    # Core
    question: str
    domains: List[str]
    messages: Annotated[Sequence[AnyMessage], add_messages]

    # Planner
    tasks: List[Dict[str, Any]]
    step_results: Dict[int, str]

    # Research (fan-out / fan-in)
    retrieved_docs: Annotated[List[Any], operator.add]
    domain_results: Annotated[List[dict], operator.add]

    # Analysis
    context: str
    findings: str
    answer: str

    # Critic
    critic_report: Dict[str, Any]
    critic_verdict: Dict[str, Any]

    # Control
    loop_count: int
    memory_context: str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ADVANCED AGENTIC RAG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AgenticMultiRAG(MultiAgentRAG):
    """
    Advanced Multi-Agentic RAG system extending MultiAgentRAG with:

    Graph Topology (one-to-many edges):
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                                                                          │
    │  orchestrate → plan → dispatcher ─┬→ Send(domain_research, orders)   ─┐ │
    │                                   ├→ Send(domain_research, products)  ─┤ │
    │                                   └→ Send(domain_research, support)   ─┘ │
    │                                                                     ↓    │
    │                                          aggregator → rerank → analyze   │
    │                                                                     ↓    │
    │                        ┌─ (DATA_GAP) ← critique ────────────────────┘    │
    │                        │─ (LOGIC_ERROR) → replan → dispatcher            │
    │                        │─ (HALLUCINATION) → analyze                      │
    │                        └─ (VERIFIED) → synthesize → END                  │
    └──────────────────────────────────────────────────────────────────────────┘

    LangGraph APIs used:
      - Send(): fan-out one-to-many domain research
      - add_messages: conversation trace accumulation
      - graph.update_state(): programmatic state correction in critique loop
      - Conditional edges: self-correction routing by error category
    """

    def __init__(self, llm_config: LLMConfig | None = None):
        """Initialize the Advanced Multi-Agentic RAG system."""
        super().__init__(llm_config=llm_config)

        print("\n" + "🌐" * 30)
        print("🚀 INITIALIZING ADVANCED ASYNC AGENTIC RAG")
        print("🌐" * 30 + "\n")

        # Initialize modular components with validated configs
        planner_config = LLMConfig(model_name="gemini-2.0-flash", temperature=0.2)
        reranker_config = LLMConfig(model_name="gemini-2.0-flash", temperature=0.1)
        critic_config = LLMConfig(model_name="gemini-2.0-flash", temperature=0.2)

        self.planner = MultiPlanner(config=planner_config)
        self.reranker = MultiReranker(config=reranker_config, threshold=0.3)
        self.critic = MultiCritic(config=critic_config)
        self.memory = MultiMemory(window_size=5)

        self.max_loops = 3

        # Build the advanced graph
        self.workflow = self._create_advanced_workflow()
        self.app = self.workflow.compile()

        print("\n✅ Advanced Agentic System is ONLINE!")
        print("-" * 60)

    # ━━━━━━━━━━━━━  GRAPH CONSTRUCTION  ━━━━━━━━━━━━━━━━━━━━

    def _create_advanced_workflow(self) -> StateGraph:
        """
        Build the advanced LangGraph workflow with:
        - Fan-out via Send() for parallel domain research
        - Fan-in via aggregator node
        - Conditional critique loop with category-based routing
        """
        workflow = StateGraph(AdvancedAgentState)

        # Register all nodes
        workflow.add_node("orchestrate", self.node_orchestrate)
        workflow.add_node("plan", self.node_plan)
        workflow.add_node("dispatcher", self.node_dispatcher)
        workflow.add_node("domain_research", self.node_domain_research)
        workflow.add_node("aggregator", self.node_aggregator)
        workflow.add_node("rerank", self.node_rerank)
        workflow.add_node("analyze", self.node_analyze)
        workflow.add_node("critique", self.node_critique)
        workflow.add_node("synthesize", self.node_synthesize)
        workflow.add_node("replan", self.node_replan)

        # ── EDGES ──

        # Entry
        workflow.set_entry_point("orchestrate")

        # Linear: orchestrate → plan → dispatcher
        workflow.add_edge("orchestrate", "plan")
        workflow.add_edge("plan", "dispatcher")

        # Fan-out: dispatcher → Send() to multiple domain_research nodes
        workflow.add_conditional_edges(
            "dispatcher",
            self._fan_out_domain_research,
            ["domain_research"],
        )

        # Fan-in: all domain_research → aggregator
        workflow.add_edge("domain_research", "aggregator")

        # Post-aggregation linear flow
        workflow.add_edge("aggregator", "rerank")
        workflow.add_edge("rerank", "analyze")
        workflow.add_edge("analyze", "critique")

        # Conditional self-correction routing from critique
        workflow.add_conditional_edges(
            "critique",
            self._route_after_critique,
            {
                "dispatcher": "dispatcher",
                "replan": "replan",
                "analyze": "analyze",
                "end": "synthesize",
            },
        )

        # Replan loops back to dispatcher for new fan-out
        workflow.add_edge("replan", "dispatcher")

        # Final
        workflow.add_edge("synthesize", END)

        return workflow

    # ━━━━━━━━━━━━━  ASYNC NODES  ━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def node_orchestrate(self, state: AdvancedAgentState) -> Dict:
        """Classify query into domains with conversation history context."""
        print("🚩 Node: Orchestrate")
        try:
            result = await self.orchestrate(state)
            print(f"✅ Orchestrated domains: {result['domains']}")
            return {
                "domains": result["domains"],
                "loop_count": 0,
                "memory_context": self.memory.get_context_for_llm(),
                "retrieved_docs": [],
                "domain_results": [],
                "step_results": {},
                "tasks": [],
                "critic_verdict": {},
                "messages": [
                    SystemMessage(content=f"[Memory Context] {self.memory.get_context_for_llm()[:500]}"),
                    HumanMessage(content=f"[User Query] {state['question']}"),
                    AIMessage(content=f"[Orchestrator] Domains classified: {result['domains']}"),
                ],
            }
        except Exception as e:
            print(f"❌ node_orchestrate error: {e}")
            return {
                "domains": ["products"],
                "loop_count": 0,
                "messages": [
                    AIMessage(content=f"[Orchestrator] Error: {e}. Defaulting to products domain."),
                ],
            }

    async def node_plan(self, state: AdvancedAgentState) -> Dict:
        """Hierarchical decomposition of the query into domain-specific tasks."""
        print("🚩 Node: Plan")
        query = state["question"]
        domains = state["domains"]
        loop = state.get("loop_count", 0)

        if loop > 0:
            report = state.get("critic_report", {})
            suggestion = report.get("suggestion", "Provide more detail")
            missing = report.get("missing_info", [])
            plan_query = (
                f"{query}. NOTE: Previous attempt failed (loop {loop}). "
                f"Suggestion: {suggestion}. Missing info: {missing}"
            )
            tasks = await self.planner.decompose(plan_query, domains)
        else:
            tasks = await self.planner.decompose(query, domains)

        return {
            "tasks": tasks,
            "loop_count": loop + 1,
            "messages": [
                AIMessage(
                    content=f"[Planner] Decomposed into {len(tasks)} tasks: "
                    + ", ".join(f"T{t['id']}({t['domain']})" for t in tasks)
                ),
            ],
        }

    async def node_dispatcher(self, state: AdvancedAgentState) -> Dict:
        """
        Pass-through node before fan-out.
        The actual Send() dispatch happens via _fan_out_domain_research conditional edge.
        """
        print("🚩 Node: Dispatcher (preparing fan-out)")
        return {
            "messages": [
                AIMessage(content=f"[Dispatcher] Preparing to fan-out across {len(state['domains'])} domains"),
            ],
        }

    def _fan_out_domain_research(self, state: AdvancedAgentState) -> list:
        """
        Fan-out: Generate Send() objects for parallel domain research.
        Each Send dispatches to the same 'domain_research' node
        with a different target_domain in the state.
        """
        question = state["question"]
        domains = state.get("domains", ["products"])
        tasks = state.get("tasks", [])

        sends = []
        for domain_name in domains:
            # Find tasks relevant to this domain
            domain_tasks = [
                t for t in tasks
                if t.get("domain") == domain_name or (isinstance(t.get("domain"), str) and t["domain"] == domain_name)
            ]
            domain_query = question
            if domain_tasks:
                domain_query = " | ".join(t.get("query", question) for t in domain_tasks)

            sends.append(
                Send(
                    "domain_research",
                    {
                        "question": question,
                        "target_domain": domain_name,
                        "domain_query": domain_query,
                        "domains": domains,
                        "messages": [],
                        "tasks": tasks,
                        "step_results": state.get("step_results", {}),
                        "retrieved_docs": [],
                        "domain_results": [],
                        "context": "",
                        "findings": "",
                        "answer": "",
                        "critic_report": {},
                        "critic_verdict": {},
                        "loop_count": state.get("loop_count", 0),
                        "memory_context": state.get("memory_context", ""),
                    },
                )
            )

        print(f"🔀 Fan-out: dispatching {len(sends)} parallel domain research branches → {domains}")
        return sends

    async def node_domain_research(self, state: dict) -> Dict:
        """
        Single-domain research node — invoked per Send() branch.
        Each branch receives 'target_domain' to know which retriever to use.
        """
        question = state.get("question", "")
        domain_name = state.get("target_domain", "products")
        domain_query = state.get("domain_query", question)
        results = dict(state.get("step_results", {}))

        retriever = self.retrievers.get(domain_name)
        if not retriever:
            print(f"⚠️ No retriever for domain: {domain_name}")
            return {
                "retrieved_docs": [],
                "domain_results": [],
                "messages": [
                    AIMessage(content=f"[DomainResearch:{domain_name}] No retriever available"),
                ],
            }

        # Execute retrieval
        print(f"  🕵️ [{domain_name}] Researching: {domain_query[:100]}")
        docs = await retriever.ainvoke(domain_query)

        # Build typed documents
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
            "step_results": results,
            "messages": [
                AIMessage(content=f"[DomainResearch:{domain_name}] Retrieved {len(docs)} docs"),
            ],
        }

    async def node_aggregator(self, state: AdvancedAgentState) -> Dict:
        """
        Fan-in aggregator: merge all domain results into unified context.
        Runs after all domain_research branches complete (operator.add reducer).
        """
        print("🚩 Node: Aggregator (fan-in)")
        all_docs = state.get("retrieved_docs", [])
        domain_results = state.get("domain_results", [])

        # Deduplicate
        seen = set()
        unique_parts = []
        for d in all_docs:
            if hasattr(d, "page_content"):
                txt = d.page_content
                if txt not in seen:
                    seen.add(txt)
                    src = d.metadata.get("source", "Unknown") if hasattr(d, "metadata") else "Unknown"
                    unique_parts.append(f"--- [Source: {src}] ---\n{txt}")

        context = "\n\n".join(unique_parts)
        print(f"✅ Aggregated {len(unique_parts)} unique chunks from {len(domain_results)} domains")

        return {
            "context": context,
            "messages": [
                AIMessage(
                    content=f"[Aggregator] Merged {len(unique_parts)} unique docs from {len(domain_results)} domains"
                ),
            ],
        }

    async def node_rerank(self, state: AdvancedAgentState) -> Dict:
        """Global rerank and context distillation."""
        print("🚩 Node: Rerank")
        query = state["question"]
        docs = state.get("retrieved_docs", [])

        # Deduplicate before reranking
        unique_docs = []
        seen = set()
        for d in docs:
            txt = d.page_content if hasattr(d, "page_content") else str(d)
            if txt not in seen:
                unique_docs.append(d)
                seen.add(txt)

        reranked = await self.reranker.score_chunks(query, unique_docs, top_n=10)

        # Build enriched context with relevance info
        context_parts = []
        for c in reranked:
            src = c["metadata"].get("source", "Unknown")
            part = (
                f"--- [Source: {src}] [Relevance: {c['score']}] ---\n"
                f"{c['text']}\n"
                f"Justification: {c['justification']}"
            )
            context_parts.append(part)

        print(f"✅ Reranking complete. Kept {len(context_parts)} chunks.")
        return {
            "context": "\n\n".join(context_parts),
            "messages": [
                AIMessage(content=f"[Reranker] Reranked to {len(context_parts)} high-relevance chunks"),
            ],
        }

    async def node_analyze(self, state: AdvancedAgentState) -> Dict:
        """Extract key findings from the global context."""
        print("🚩 Node: Analyze")
        if not state.get("context"):
            print("⚠️ No context available for analysis.")
            return {
                "findings": "No relevant information found in the knowledge base.",
                "messages": [AIMessage(content="[Analyst] No context available")],
            }
        result = await self.analyze(state)
        print(f"✅ Analysis complete. {len(result.get('findings', ''))} chars produced.")
        return {
            "findings": result["findings"],
            "messages": [
                AIMessage(content=f"[Analyst] Findings produced ({len(result.get('findings', ''))} chars)"),
            ],
        }

    async def node_critique(self, state: AdvancedAgentState) -> Dict:
        """
        Global Critic verification using Pydantic CriticVerdict.
        Uses update_state pattern — returns typed corrections to state.
        """
        print("🚩 Node: Critique")
        query = state["question"]
        context = state.get("context", "")
        findings = state.get("findings", "")
        domains = state["domains"]
        answer = state.get("answer", "Not yet synthesized")

        # Get typed verdict
        verdict = await self.critic.verify_typed(
            query, context, findings, answer, domains
        )

        # Build state update with typed verdict
        state_update: Dict[str, Any] = {
            "critic_report": verdict.model_dump(),
            "critic_verdict": verdict.model_dump(),
            "messages": [
                AIMessage(
                    content=(
                        f"[Critic] Verified={verdict.is_verified}, "
                        f"Category={verdict.category.value}, "
                        f"Feedback={verdict.feedback[:200]}"
                    )
                ),
            ],
        }

        # update_state pattern: if not verified, programmatically inject
        # corrective instructions into the message trace
        if not verdict.is_verified:
            correction_msg = SystemMessage(
                content=(
                    f"[STATE_CORRECTION] The critic identified a {verdict.category.value} issue. "
                    f"Missing info: {verdict.missing_info}. "
                    f"Suggestion: {verdict.suggestion}. "
                    f"The system will now attempt self-correction."
                )
            )
            state_update["messages"].append(correction_msg)

            # If hallucination, clear the findings to force re-analysis
            if verdict.category == CriticCategory.HALLUCINATION:
                state_update["findings"] = ""

        return state_update

    async def node_synthesize(self, state: AdvancedAgentState) -> Dict:
        """Final answer synthesis with memory update and full message trace."""
        print("🚩 Node: Synthesize")
        result = await self.synthesize(state)

        # Update memory with Pydantic-validated turn
        self.memory.add_interaction(
            state["question"],
            result["answer"],
            state["domains"],
        )

        # Compile the full message trace summary
        msg_count = len(state.get("messages", []))

        return {
            "answer": result["answer"],
            "messages": [
                AIMessage(
                    content=(
                        f"[Synthesizer] Final answer produced. "
                        f"Total message trace: {msg_count + 1} messages. "
                        f"Loops: {state.get('loop_count', 0)}."
                    )
                ),
            ],
        }

    async def node_replan(self, state: AdvancedAgentState) -> Dict:
        """Context-aware re-planning after logic failure."""
        print("🚩 Node: Replan")
        query = state["question"]
        report = state.get("critic_report", {})
        print(f"🔄 Re-planning due to: {report.get('category', 'unknown')} - {report.get('feedback', '')[:100]}")

        new_query = (
            f"{query}. REDESIGN PLAN: {report.get('suggestion', 'Try different approach')}. "
            f"Previous findings: {state.get('findings', '')[:500]}"
        )
        tasks = await self.planner.decompose(new_query, state["domains"])

        return {
            "tasks": tasks,
            "messages": [
                AIMessage(
                    content=f"[Replanner] Generated {len(tasks)} new tasks after {report.get('category', 'unknown')} error"
                ),
            ],
        }

    # ━━━━━━━━━━━━━  CONDITIONAL ROUTING  ━━━━━━━━━━━━━━━━━━

    async def _route_after_critique(self, state: AdvancedAgentState) -> str:
        """
        Decide recovery path or termination based on CriticVerdict category.

        Returns:
            "end" → synthesize
            "dispatcher" → re-dispatch fan-out research (DATA_GAP)
            "replan" → rebuild task plan (LOGIC_ERROR)
            "analyze" → re-analyze without new data (HALLUCINATION)
        """
        report = state.get("critic_report", {})
        loop = state.get("loop_count", 0)

        # Validate through Pydantic
        try:
            verdict = CriticVerdict.model_validate(report)
        except Exception:
            return "end"

        if verdict.is_verified or loop >= self.max_loops:
            if loop >= self.max_loops and not verdict.is_verified:
                print(f"⚠️ Max loops ({self.max_loops}) reached. Finalizing despite issues.")
            return "end"

        category = verdict.category
        print(f"🚨 Self-Correction Required: {category.value} (loop {loop})")

        if category == CriticCategory.DATA_GAP:
            return "dispatcher"  # Fan-out again
        if category == CriticCategory.LOGIC_ERROR:
            return "replan"
        if category == CriticCategory.HALLUCINATION:
            return "analyze"

        return "end"

    # ━━━━━━━━━━━━━  PUBLIC API  ━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def query(self, question: str) -> Dict:
        """
        Execute the multi-agentic flow (Async).

        Args:
            question: The user's question.

        Returns:
            Dict with answer, domains, report, loops, and message_trace.
        """
        print(f"\n🎯 GLOBAL QUERY Processing Initiated: '{question}'")
        print("=" * 60)

        initial_state: AdvancedAgentState = {
            "question": question,
            "domains": [],
            "messages": [HumanMessage(content=question)],
            "tasks": [],
            "step_results": {},
            "retrieved_docs": [],
            "domain_results": [],
            "context": "",
            "findings": "",
            "answer": "",
            "critic_report": {},
            "critic_verdict": {},
            "loop_count": 0,
            "memory_context": "",
        }

        final_state = await self.app.ainvoke(initial_state)

        print("\n✅ GLOBAL AGENTIC Result Ready!")
        print("=" * 60)

        # Build message trace summary
        messages = final_state.get("messages", [])
        message_trace = []
        for msg in messages:
            message_trace.append({
                "role": msg.__class__.__name__,
                "content": msg.content[:300] if hasattr(msg, "content") else str(msg)[:300],
            })

        return {
            "query": question,
            "answer": final_state.get("answer", ""),
            "domains": final_state.get("domains", []),
            "report": final_state.get("critic_report", {}),
            "loops": final_state.get("loop_count", 0),
            "message_trace": message_trace,
            "total_messages": len(messages),
        }

    async def query_with_state_update(self, question: str) -> Dict:
        """
        Execute the flow with explicit update_state for mid-run corrections.
        Demonstrates LangGraph's update_state API.

        This version compiles the graph with interrupts and uses
        graph.update_state() to inject corrections.
        """
        print(f"\n🎯 QUERY WITH STATE UPDATE: '{question}'")
        print("=" * 60)

        # Compile with interrupt_before for critique node
        interruptable_graph = self.workflow.compile(
            interrupt_before=["critique"],
        )

        initial_state: AdvancedAgentState = {
            "question": question,
            "domains": [],
            "messages": [HumanMessage(content=question)],
            "tasks": [],
            "step_results": {},
            "retrieved_docs": [],
            "domain_results": [],
            "context": "",
            "findings": "",
            "answer": "",
            "critic_report": {},
            "critic_verdict": {},
            "loop_count": 0,
            "memory_context": "",
        }

        config = {"configurable": {"thread_id": f"update_state_{question[:20]}"}}

        # Run until interrupt (before critique)
        pre_critique_state = await interruptable_graph.ainvoke(initial_state, config)

        # Programmatic update_state: inject additional context before critique runs
        await interruptable_graph.aupdate_state(
            config,
            {
                "messages": [
                    SystemMessage(
                        content=(
                            "[STATE_UPDATE] Pre-critique injection: "
                            "Ensure all findings have source citations. "
                            "Flag any unsupported claims."
                        )
                    ),
                ],
            },
        )

        # Resume execution
        final_state = await interruptable_graph.ainvoke(None, config)

        return {
            "query": question,
            "answer": final_state.get("answer", ""),
            "domains": final_state.get("domains", []),
            "report": final_state.get("critic_report", {}),
            "loops": final_state.get("loop_count", 0),
            "state_updated": True,
        }

    async def astream_query(self, question: str):
        """Stream events from the multi-agentic flow."""
        initial_state: AdvancedAgentState = {
            "question": question,
            "domains": [],
            "messages": [HumanMessage(content=question)],
            "tasks": [],
            "step_results": {},
            "retrieved_docs": [],
            "domain_results": [],
            "context": "",
            "findings": "",
            "answer": "",
            "critic_report": {},
            "critic_verdict": {},
            "loop_count": 0,
            "memory_context": "",
        }

        async for event in self.app.astream(initial_state):
            yield event


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    async def test():
        agent = AgenticMultiRAG()

        # Standard query
        res = await agent.query(
            "Check status for order #1234 and tell me features of the product in that order."
        )
        print(f"\n💡 Answer:\n{res['answer']}")
        print(f"\n📊 Domains: {res['domains']}")
        print(f"🔄 Loops: {res['loops']}")
        print(f"📨 Total Messages: {res['total_messages']}")
        print("\n📝 Message Trace:")
        for msg in res.get("message_trace", []):
            print(f"  [{msg['role']}] {msg['content'][:120]}")

    asyncio.run(test())
