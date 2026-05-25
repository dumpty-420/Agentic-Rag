"""
Multi-Agentic RAG Integrated with FastAPI
Features: Multi-domain routing, Fan-out research, Task decomposition,
          Global reranking, Reflection loops, Pydantic contracts.
"""
import os
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from dotenv import load_dotenv
import uvicorn

# Import the AgenticMultiRAG system
from agentic_multi_rag import AgenticMultiRAG
from multi_agent_rag import ensure_indexes, ingest_csvs

# Import Pydantic schemas from centralized module
from schemas import (
    LLMConfig,
    QueryRequest,
    QueryResponse,
    CriticReportResponse,
    IngestRequest,
    IngestResponse,
    GraphNodeInfo,
    GraphStructureResponse,
)

load_dotenv()

app = FastAPI(
    title="Multi-Agentic RAG API",
    description=(
        "Advanced Multi-Domain RAG system with autonomous agents, "
        "fan-out/fan-in research, Pydantic-validated schemas, "
        "and self-correction loops powered by LangGraph."
    ),
    version="2.0.0",
)

# Initialize the Multi-Agent System (Singleton)
print("🚀 Initializing Global Agentic Multi-RAG system...")
agent = AgenticMultiRAG()


# ━━━━━━━━━━━━━━━━  ENDPOINTS  ━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/")
async def root():
    return {
        "message": "Multi-Agentic RAG API is Online",
        "version": "2.0.0",
        "features": [
            "Pydantic-validated schemas",
            "Fan-out/fan-in domain research",
            "add_messages conversation trace",
            "Self-correction critique loops",
        ],
    }


@app.get("/health")
async def health_check():
    """Check system health and active domains."""
    try:
        return {
            "status": "healthy",
            "active_domains": ["orders", "products", "support"],
            "llm_model": agent.planner.model_name,
            "loops_allowed": agent.max_loops,
            "memory_window": agent.memory.window_size,
            "memory_turns": len(agent.memory.history),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_multi_agent(request: QueryRequest):
    """Execute the full Agentic RAG loop across multiple domains."""
    try:
        print(f"\n📩 API Request Received: '{request.question}'")

        # Update max loops if requested
        if request.max_loops:
            agent.max_loops = request.max_loops

        # Run the async agentic loop
        result = await agent.query(request.question)

        # Format Critic Report through Pydantic
        raw_report = result.get("report", {})
        critic_report = CriticReportResponse(
            is_verified=raw_report.get("is_verified", True),
            category=raw_report.get("category", "NONE"),
            feedback=raw_report.get("feedback", "OK"),
            missing_info=raw_report.get("missing_info", []),
            suggestion=raw_report.get("suggestion", "None"),
        )

        return QueryResponse(
            query=result.get("query", request.question),
            answer=result.get("answer", ""),
            domains=result.get("domains", []),
            loops=result.get("loops", 0),
            critic_report=critic_report,
        )
    except Exception as e:
        print(f"❌ API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def trigger_ingestion(request: IngestRequest, background_tasks: BackgroundTasks):
    """Trigger the ingestion of multi-domain CSV data into Pinecone."""
    try:
        if not os.path.exists(request.knowledge_dir):
            raise HTTPException(status_code=404, detail="Knowledgebase directory not found")

        print(f"📥 Triggering batch ingestion from: {request.knowledge_dir}")
        await asyncio.to_thread(ingest_csvs, request.knowledge_dir)

        return IngestResponse(
            message=f"Ingestion process completed for {request.knowledge_dir}",
            success=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph", response_model=GraphStructureResponse)
async def get_graph_structure():
    """Return the compiled graph topology for visualization."""
    try:
        graph = agent.app
        graph_data = graph.get_graph()

        nodes = []
        for node_name in graph_data.nodes:
            edges_to = []
            for edge in graph_data.edges:
                if edge[0] == node_name:
                    edges_to.append(edge[1])

            nodes.append(GraphNodeInfo(
                node_name=node_name,
                edges_to=edges_to,
                is_conditional=False,
            ))

        return GraphStructureResponse(
            nodes=nodes,
            entry_point="orchestrate",
            total_nodes=len(nodes),
            total_edges=len(graph_data.edges),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get graph structure: {str(e)}")


@app.get("/memory")
async def get_memory():
    """Return current conversation memory."""
    return {
        "turns": agent.memory.serialize(),
        "total_turns": len(agent.memory.history),
        "window_size": agent.memory.window_size,
    }


@app.delete("/memory")
async def clear_memory():
    """Clear conversation memory."""
    agent.memory.clear()
    return {"message": "Memory cleared", "success": True}


if __name__ == "__main__":
    print("🛠️ Ensuring Pinecone indexes exist...")
    ensure_indexes()
    uvicorn.run(app, host="0.0.0.0", port=8000)
