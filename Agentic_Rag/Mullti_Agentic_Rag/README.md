## Multi-Agent RAG over Multiple Pinecone Indexes (LangGraph + Gemini 2.5 Flash)

This project demonstrates a production-ready, multi-agent RAG pipeline built with LangGraph. It orchestrates specialized agents to retrieve and reason over multiple domain-specific Pinecone indexes populated from CSV knowledge bases.



Agents
- Orchestrator: Classifies the query into domains (orders/products/support) and decides which retriever(s) to use.
- Researcher: Retrieves top-k chunks from the proper Pinecone index(es).
- Analyst: Extracts concise, structured findings from retrieved context.
- Critic: Checks for gaps/consistency and highlights missing information.
- Synthesizer: Produces a clear final answer grounded in context, findings, and critique.

### Components
- LLM: Gemini 2.5 Flash
- Embeddings: Sentence-Transformers `all-MiniLM-L6-v2` (384-d)
- Vector Store: Pinecone with three indexes: `orders-index`, `products-index`, `support-index`
- Graph: LangGraph nodes wired to match the architecture above

### Repository Layout
- `multi_agent_rag.py`: Multi-agent RAG pipeline, LangGraph wiring, ingestion runner.
- `pinecone_multi_index.py`: Pinecone index manager and creation utilities.
- `embeddings_sentance.py`: Minimal sentence-transformers wrapper (reference).
- `rag_with_langgraph.py`: The original single-agent/simple RAG example (reference).
- `knowledgebase/`: CSVs to ingest
  - `order.csv` -> `orders-index`
  - `product.csv` -> `products-index`
  - `support_catalogue.csv` -> `support-index`

### Prerequisites
Set environment variables (e.g., via `.env`):

```bash
GOOGLE_API_KEY=your_google_api_key
PC_API_KEY=your_pinecone_api_key
# optional (serverless defaults)
PC_CLOUD=aws
PC_REGION=us-east-1
```

Install dependencies (minimum):

```bash
pip install \
  langchain langgraph \
  sentence-transformers \
  langchain-google-genai \
  langchain-pinecone \
  pinecone-client python-dotenv
```

### Ingest Knowledge and Run
Ingest the CSVs into their respective Pinecone indexes and run sample queries:

```bash
INGEST=true python multi_agent_rag.py
```

Run without re-ingesting (uses existing indexes):

```bash
python multi_agent_rag.py
```

### How It Works (High-Level)
1. Index setup: `pinecone_multi_index.py` ensures `orders-index`, `products-index`, `support-index` exist (serverless by default).
2. Ingestion: `multi_agent_rag.py` can chunk CSVs and upsert into the appropriate index using sentence-transformers embeddings.
3. Orchestration: The Orchestrator selects domains based on the user query.
4. Retrieval: The Researcher queries one or more retrievers (per index) to collect context.
5. Reasoning: Analyst summarizes findings; Critic checks completeness; Synthesizer produces the final grounded response.

### Notes
- Default embedding model: `all-MiniLM-L6-v2` (dimension=384) — aligned with the index configs.
- You can customize chunking, k-values, and prompts inside `multi_agent_rag.py`.


