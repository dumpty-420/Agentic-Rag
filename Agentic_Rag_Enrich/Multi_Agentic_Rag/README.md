 Agents in Multi-Agentic RAG
1. Orchestrator Agent
File: multi_agent_rag.py → orchestrate() node
Role: The entry point. Reads the user query and classifies it into one or more domains — orders, products, or support.
How: Uses Gemini LLM to output a JSON like {"domains": ["orders", "products"]}. If unclear, defaults to products.

2. Planner Agent
File: planner_multi.py
Role: Decomposes the user query into sub-questions — one per domain. If the Critic found gaps in a previous loop, it enriches the sub-questions with the missing info before retrying.
How: Takes the query + domains → generates targeted sub-questions like "What is the status of order #1234?" and "What are the features of the product in that order?"

3. Researcher Agent
File: agentic_multi_rag.py → node_research()
Role: For each domain and each sub-question, retrieves relevant documents from the corresponding Pinecone index.
How: Routes to the right retriever — orders_ret, products_ret, or support_ret — and collects all matching chunks.

4. Reranker Agent
File: reranker_multi.py
Role: Globally deduplicates and reranks all retrieved chunks from all domains using scoring. Returns the top-N most relevant chunks.
How: Scores each chunk against the original query, removes duplicates, returns top 8.

5. Analyst Agent
File: multi_agent_rag.py → analyze() node
Role: Reads the reranked context and extracts key facts relevant to the question as a bullet list.
How: LLM prompt — "Extract key facts from this context relevant to the question."

6. Critic Agent
File: critic_multi.py
Role: Verifies if the Analyst's findings fully answer the question. Detects gaps, missing info, and inconsistencies.
How: Returns a structured report with is_verified, feedback, missing_info, and suggestion. If not verified and loops < 3, sends back to Planner.

7. Synthesizer Agent
File: multi_agent_rag.py → synthesize() node
Role: Produces the final answer using context + findings + critique. Also updates the Memory module.
How: LLM prompt combining all prior outputs into a coherent, cited response.

Why It's Multi-Agent
Each agent has a single, specialized responsibility and passes its output to the next:
User Query
    ↓
Orchestrator  →  classifies domains
    ↓
Planner       →  decomposes into sub-questions
    ↓
Researcher    →  retrieves from 3 Pinecone indexes in parallel
    ↓
Reranker      →  merges and scores all results globally
    ↓
Analyst       →  extracts structured findings
    ↓
Critic        →  verifies completeness
    ↓ (loops back to Planner if gaps found, max 3x)
Synthesizer   →  generates final answer + updates memory
What makes it truly multi-agent:

Each agent is independently implemented in its own file
Agents communicate through a shared LangGraph state (MultiAgentState)
The Critic creates a reflection loop — it can send the workflow back to the Planner if the answer is incomplete
3 separate Pinecone indexes are queried in parallel by the Researcher
The Memory module persists context across turns for conversational support
