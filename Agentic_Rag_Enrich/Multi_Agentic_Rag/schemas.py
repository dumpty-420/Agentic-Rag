"""
Centralized Pydantic Schemas for Multi-Agentic RAG System

All data contracts — configuration, LLM response parsing, graph state,
fan-out payloads, API request/response — live in this module.
"""
from __future__ import annotations

import operator
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Annotated

from pydantic import BaseModel, Field, field_validator, model_validator
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ENUMS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DomainEnum(str, Enum):
    """Valid knowledge domains in the RAG system."""
    ORDERS = "orders"
    PRODUCTS = "products"
    SUPPORT = "support"


class CriticCategory(str, Enum):
    """Error categories assigned by the Critic agent."""
    DATA_GAP = "DATA_GAP"
    LOGIC_ERROR = "LOGIC_ERROR"
    HALLUCINATION = "HALLUCINATION"
    NONE = "NONE"


class TaskPriority(str, Enum):
    """Priority levels for planner-generated tasks."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONFIGURATION MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LLMConfig(BaseModel):
    """Configuration for the language model."""
    model_name: str = Field(default="gemini-2.0-flash", description="LLM model identifier")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Sampling temperature")
    api_key: Optional[str] = Field(default=None, description="Google API key (reads from env if None)")

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model_name cannot be empty")
        return v.strip()


class PineconeConfig(BaseModel):
    """Configuration for Pinecone vector store."""
    index_name: str = Field(default="orders-index", description="Pinecone index name")
    dimension: int = Field(default=384, gt=0, description="Embedding dimension")
    metric: str = Field(default="cosine", description="Distance metric")
    api_key: Optional[str] = Field(default=None, description="Pinecone API key (reads from env if None)")
    cloud: str = Field(default="aws", description="Cloud provider")
    region: str = Field(default="us-east-1", description="Cloud region")

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        allowed = {"cosine", "euclidean", "dotproduct"}
        if v not in allowed:
            raise ValueError(f"metric must be one of {allowed}")
        return v


class DomainConfig(BaseModel):
    """Configuration for a single knowledge domain."""
    namespace: str = Field(..., description="Pinecone namespace for this domain")
    csv_path: str = Field(..., description="Path to the CSV file for ingestion")
    retriever_k: int = Field(default=5, ge=1, le=50, description="Number of docs to retrieve")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LLM RESPONSE MODELS (Structured Output Parsing)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class OrchestratorResponse(BaseModel):
    """Structured response from the Orchestrator LLM call."""
    domains: List[DomainEnum] = Field(..., description="Classified domains for the query")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Classification confidence")
    reasoning: str = Field(default="", description="Rationale for domain classification")

    @field_validator("domains")
    @classmethod
    def ensure_at_least_one_domain(cls, v: List[DomainEnum]) -> List[DomainEnum]:
        if not v:
            return [DomainEnum.PRODUCTS]
        return v


class PlannerTask(BaseModel):
    """A single decomposed task from the Planner."""
    id: int = Field(..., ge=1, description="Unique task identifier")
    query: str = Field(..., min_length=3, description="The sub-question to research")
    domain: DomainEnum = Field(..., description="Target domain for this task")
    depends_on: List[int] = Field(default_factory=list, description="IDs of prerequisite tasks")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Execution priority")

    @field_validator("depends_on")
    @classmethod
    def no_self_dependency(cls, v: List[int], info) -> List[int]:
        task_id = info.data.get("id")
        if task_id and task_id in v:
            raise ValueError(f"Task {task_id} cannot depend on itself")
        return v


class PlannerResponse(BaseModel):
    """Structured response from the Planner LLM call."""
    tasks: List[PlannerTask] = Field(..., min_length=1, description="Ordered list of decomposed tasks")
    strategy: str = Field(default="sequential", description="Execution strategy description")

    @model_validator(mode="after")
    def validate_dependency_graph(self) -> "PlannerResponse":
        task_ids = {t.id for t in self.tasks}
        for task in self.tasks:
            for dep in task.depends_on:
                if dep not in task_ids:
                    raise ValueError(f"Task {task.id} depends on non-existent task {dep}")
        return self


class RerankerScore(BaseModel):
    """Score output from the Reranker for a single chunk."""
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    justification: str = Field(default="", description="Why this score was assigned")


class CriticVerdict(BaseModel):
    """Structured verdict from the Critic agent."""
    is_verified: bool = Field(..., description="Whether the answer passes quality check")
    category: CriticCategory = Field(default=CriticCategory.NONE, description="Error category")
    feedback: str = Field(default="", description="Detailed feedback on the answer")
    missing_info: List[str] = Field(default_factory=list, description="Specific missing facts")
    suggestion: str = Field(default="", description="Actionable suggestion to fix issues")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DOCUMENT & PAYLOAD MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RetrievedDocument(BaseModel):
    """A single document retrieved from the vector store."""
    content: str = Field(..., description="Document text content")
    source: str = Field(default="Unknown", description="Original source file")
    domain: DomainEnum = Field(..., description="Domain this document belongs to")
    relevance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    model_config = {"arbitrary_types_allowed": True}


class DomainResearchPayload(BaseModel):
    """Payload sent via LangGraph Send() for fan-out domain research."""
    domain: DomainEnum = Field(..., description="Target domain to research")
    query: str = Field(..., description="Search query for this domain")
    task_id: int = Field(default=1, description="Associated planner task ID")


class DomainResearchResult(BaseModel):
    """Result from a single domain research branch."""
    domain: DomainEnum = Field(..., description="Domain that was researched")
    documents: List[RetrievedDocument] = Field(default_factory=list)
    task_id: int = Field(default=1)
    context_snippet: str = Field(default="", description="Formatted context from this domain")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MEMORY MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ConversationTurn(BaseModel):
    """A single turn in the conversation history."""
    user: str = Field(..., description="User query")
    assistant: str = Field(..., description="AI response")
    domains: List[DomainEnum] = Field(default_factory=list, description="Domains consulted")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LANGGRAPH STATE MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AgentState(BaseModel):
    """
    Central state schema for the LangGraph multi-agent workflow.

    Uses Pydantic for validation + LangGraph annotations for reducers:
    - `messages` uses `add_messages` reducer to accumulate conversation traces
    - `domain_results` uses `operator.add` for fan-in accumulation from parallel branches
    """
    # ── Core Query ──
    question: str = Field(default="", description="Original user question")
    domains: List[DomainEnum] = Field(default_factory=list, description="Active domains")

    # ── LangGraph Message History (add_messages reducer) ──
    messages: Annotated[Sequence[AnyMessage], add_messages] = Field(default_factory=list)

    # ── Planner Output ──
    tasks: List[PlannerTask] = Field(default_factory=list, description="Decomposed tasks from planner")
    step_results: Dict[int, str] = Field(default_factory=dict, description="Results per task ID")

    # ── Research Fan-out/Fan-in ──
    domain_results: Annotated[List[DomainResearchResult], operator.add] = Field(default_factory=list)
    retrieved_docs: Annotated[List[Any], operator.add] = Field(default_factory=list)

    # ── Analysis ──
    context: str = Field(default="", description="Aggregated and reranked context")
    findings: str = Field(default="", description="Analyst extracted findings")
    answer: str = Field(default="", description="Final synthesized answer")

    # ── Critique ──
    critic_verdict: Optional[CriticVerdict] = Field(default=None, description="Critic's typed verdict")
    critic_report: Dict[str, Any] = Field(default_factory=dict, description="Raw critic report for compat")

    # ── Control Flow ──
    loop_count: int = Field(default=0, ge=0, description="Current critique loop iteration")
    memory_context: str = Field(default="", description="Formatted conversation history")

    model_config = {"arbitrary_types_allowed": True}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  API REQUEST / RESPONSE MODELS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class QueryRequest(BaseModel):
    """Incoming query request to the API."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        examples=["What is the status of order #1234 and what are the product features?"],
    )
    max_loops: int = Field(default=3, ge=1, le=10, description="Max critique loops")


class CriticReportResponse(BaseModel):
    """Critic report in API response format."""
    is_verified: bool = Field(default=True)
    category: str = Field(default="NONE")
    feedback: str = Field(default="OK")
    missing_info: List[str] = Field(default_factory=list)
    suggestion: str = Field(default="None")


class QueryResponse(BaseModel):
    """API response for a processed query."""
    query: str
    answer: str
    domains: List[str]
    loops: int
    critic_report: Optional[CriticReportResponse] = None


class IngestRequest(BaseModel):
    """Request to trigger CSV data ingestion."""
    knowledge_dir: str = Field(default="knowledgebase", description="Path to knowledge CSVs")


class IngestResponse(BaseModel):
    """Response from ingestion endpoint."""
    message: str
    success: bool


class GraphNodeInfo(BaseModel):
    """Information about a node in the compiled graph."""
    node_name: str
    edges_to: List[str] = Field(default_factory=list)
    is_conditional: bool = Field(default=False)


class GraphStructureResponse(BaseModel):
    """Serialized graph topology for visualization."""
    nodes: List[GraphNodeInfo]
    entry_point: str
    total_nodes: int
    total_edges: int


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HELPER: Safe JSON → Pydantic parsing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_llm_json(raw_content: str) -> str:
    """
    Extract JSON from LLM output that may be wrapped in markdown code fences.
    Returns cleaned JSON string ready for model_validate_json().
    """
    content = raw_content.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    return content
