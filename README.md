# Agentic RAG 🤖

A progressive RAG system that evolves from a basic modular pipeline to a full multi-domain agentic system with reflection loops. Built with LangChain, LangGraph, Pinecone, and Google Gemini.

---

## Repository Structure

```
Agentic-Rag/
├── Agentic_Rag/                        # Version 1 — Basic RAG
│   ├── Single_agentic-rag/             # Basic modular RAG pipeline
│   └── Mullti_Agentic_Rag/             # Basic multi-agent RAG (minimal)
│
└── Agentic_Rag_Enrich/                 # Version 2 — Enriched Agentic RAG
    ├── Single_agentic-rag/             # Enriched single agent with full agentic loop
    └── Multi_Agentic_Rag/              # Full multi-agent system with reflection loops
```

---

## What's Different Between the Two Versions

| Feature | Agentic_Rag (Basic) | Agentic_Rag_Enrich (Enriched) |
|---|---|---|
| Query Planning | ❌ | ✅ `planner.py` / `planner_multi.py` |
| Result Reranking | ❌ | ✅ `reranker.py` / `reranker_multi.py` |
| Answer Criticism | ❌ | ✅ `critic.py` / `critic_multi.py` |
| Conversation Memory | ❌ | ✅ `memory.py` / `memory_multi.py` |
| Agentic Loop | ❌ | ✅ `agentic_rag.py` / `agentic_multi_rag.py` |
| Reflection Loops | ❌ | ✅ loops back if gaps found (max 3x) |
| LangGraph Tools | ❌ | ✅ `tools.py` |
| Schemas/DTOs | ❌ | ✅ `schemas.py` |
| Tests | ❌ | ✅ `test_agentic_rag.py` / `test_agentic_multi_rag.py` |
| FastAPI Server | ✅ | ✅ |
| Pinecone Multi-Index | ✅ (basic) | ✅ (enriched with RRF reranking) |

---

## Part 1 — Agentic_Rag (Basic)

### Single_agentic-rag
A clean modular RAG pipeline. Each component is its own module. No agentic loop — just ingest → retrieve → generate.

```
Single_agentic-rag/
├── main.py                          # FastAPI entry point
├── ingest_all.py                    # Document ingestion into Pinecone
├── modular_rag_pipeline.py          # Pipeline orchestration
├── modular_rag_pipeline_with_chromaDb.py  # ChromaDB alternative
├── embeddings_module.py             # Gemini embedding model
├── document_preprocessor.py        # Chunking and preprocessing
├── retriever.py                     # Semantic retrieval
├── response_generator.py           # LLM answer generation
├── vector_store.py                  # Pinecone vector store
├── vector_store_chromadb.py        # ChromaDB alternative
└── knowledgebase/                   # Source documents
```

### Mullti_Agentic_Rag (Basic)
A minimal multi-agent RAG with just the core routing logic. No planning, no criticism, no memory.

```
Mullti_Agentic_Rag/
├── multi_agent_rag.py               # Basic LangGraph agent
├── rag_with_langgraph.py            # LangGraph graph definition
├── embeddings_sentance.py           # HuggingFace embeddings
├── pinecone_multi_index.py          # Creates domain indexes
└── knowledgebase/                   # Domain CSV files
```

**Run:**
```bash
cd Agentic_Rag/Single_agentic-rag
uv sync
uv run python ingest_all.py
uv run python main.py
```

---

## Part 2 — Agentic_Rag_Enrich (Enriched)

### Single_agentic-rag (Enriched)
Same modular foundation as the basic version, but with a full agentic loop added on top.

**New files added:**
- `agentic_rag.py` — core agentic loop that iterates until answer is satisfactory
- `planner.py` — decomposes complex queries into sub-questions
- `reranker.py` — scores and reranks retrieved chunks for precision
- `critic.py` — verifies if the answer is complete, flags gaps
- `memory.py` — stores conversation history for multi-turn support
- `tools.py` — LangChain tools for agentic actions

```
Single_agentic-rag/
├── main.py
├── ingest_all.py
├── agentic_rag.py                   # ✨ NEW — agentic loop
├── modular_rag_pipeline.py
├── planner.py                       # ✨ NEW — query decomposition
├── reranker.py                      # ✨ NEW — result reranking
├── critic.py                        # ✨ NEW — answer verification
├── memory.py                        # ✨ NEW — conversation memory
├── tools.py                         # ✨ NEW — LangChain tools
├── embeddings_module.py
├── document_preprocessor.py
├── retriever.py
├── response_generator.py
├── vector_store.py
└── knowledgebase/
```

### Multi_Agentic_Rag (Enriched)
A full multi-domain agentic RAG with LangGraph, reflection loops, global reranking, and conversation memory. Routes queries across 3 separate Pinecone indexes (orders, products, support).

**New files compared to basic multi-agent:**
- `agentic_multi_rag.py` — extended agent with reflection loops
- `planner_multi.py` — multi-domain query decomposition
- `reranker_multi.py` — global RRF reranking across all domains
- `critic_multi.py` — multi-domain answer verification
- `memory_multi.py` — multi-turn conversation memory
- `schemas.py` — Pydantic data models
- `multi_agent_rag_new.py` — updated entry point with FastAPI
- `test_agentic_multi_rag.py` — test suite

```
Multi_Agentic_Rag/
├── multi_agent_rag.py               # Base LangGraph agent (5 nodes)
├── agentic_multi_rag.py             # ✨ Extended agent with reflection
├── multi_agent_rag_new.py           # ✨ FastAPI entry point
├── planner_multi.py                 # ✨ Multi-domain decomposition
├── reranker_multi.py                # ✨ Global RRF reranking
├── critic_multi.py                  # ✨ Answer verification
├── memory_multi.py                  # ✨ Conversation memory
├── schemas.py                       # ✨ Pydantic models
├── rag_with_langgraph.py            # LangGraph graph definition
├── embeddings_sentance.py           # HuggingFace embeddings
├── pinecone_multi_index.py          # Creates 3 Pinecone indexes
├── test_agentic_multi_rag.py        # ✨ Test suite
└── knowledgebase/
    ├── order.csv
    ├── product.csv
    └── support_catalogue.csv
```

**Multi-Agent Flow:**
```
User Query
    ↓
Orchestrator → classifies into domains (orders / products / support)
    ↓
Planner → decomposes into sub-questions
    ↓
Researcher → retrieves from 3 domain Pinecone indexes
    ↓
Reranker → global RRF reranking + deduplication
    ↓
Analyst → extracts key findings
    ↓
Critic → verifies answer, detects gaps
    ↓ (loops back to Planner if gaps found, max 3x)
Synthesizer → final answer + updates memory
```

**Run:**
```bash
cd Agentic_Rag_Enrich/Multi_Agentic_Rag
uv sync

# Create Pinecone indexes
uv run python pinecone_multi_index.py

# Ingest domain data
uv run python -c "from multi_agent_rag import ingest_csvs; ingest_csvs()"

# Run the server
uv run python multi_agent_rag_new.py
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini (`gemini-2.0-flash`) |
| Embeddings | Gemini (`gemini-embedding-001`) / HuggingFace (`all-MiniLM-L6-v2`) |
| Vector DB | Pinecone (cosine, 384/1024 dims) |
| Agent Framework | LangGraph |
| Orchestration | LangChain |
| API | FastAPI |
| Package Manager | uv |

---

## Environment Variables

```env
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PC_API_KEY=your_pinecone_api_key
PC_CLOUD=aws
PC_REGION=us-east-1
TRANSFORMERS_OFFLINE=1  # optional: load HuggingFace models from cache
```

---

## RAG Evolution

```
Agentic_Rag (Basic)
  Ingest → Retrieve → Generate

Agentic_Rag_Enrich/Single (Enriched)
  Plan → Retrieve → Rerank → Critique → Generate → Remember

Agentic_Rag_Enrich/Multi (Full Multi-Agent)
  Orchestrate → Plan → Research (3 domains) → Rerank → Analyze → Critique (loop) → Synthesize
```

---

## Author

Built by [Seerat Chugh](https://github.com/dumpty-420)
