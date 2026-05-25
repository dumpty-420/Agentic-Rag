# Agentic RAG System (Single-Agent Router + Modular RAG)

A clean, modular RAG system extended with a single-agent LangGraph router that decides which Pinecone index(es) to query based on the user request, checks sufficiency, and then uses an LLM to generate the final response.

## 📁 Project Structure

```
agentic-rag/
├── agentic_single.py                 → Single-agent LangGraph router (orders/products/support)
├── embeddings_sentence.py            → Sentence-transformers embeddings (all-MiniLM-L6-v2)
├── embeddings_module.py              → Gemini embeddings (optional)
├── ingest_knowledgebase.py           → Ingest CSVs into Pinecone multi-index
├── pinecone_multi_index.py           → Ensure/create Pinecone indexes
├── vector_store.py                   → Pinecone single-index operations
├── vector_store_chromadb.py          → ChromaDB vector store (optional)
├── retriever.py                      → Retrieval using embedder + vector store
├── response_generator.py             → Prompt + LLM (Gemini 2.5 Flash)
├── modular_rag_pipeline.py           → Modular RAG (Pinecone)
├── modular_rag_pipeline_with_chromaDb.py → Modular RAG (ChromaDB)
├── document_preprocessor.py          → Text cleaning/splitting
├── knowledgebase/
│   ├── order.csv                     → Orders KB → orders-index
│   ├── product.csv                   → Products KB → products-index
│   └── support_catalogue.csv         → Support KB → support-index
└── README.md
```

## 🔧 Module Descriptions

### 1. `embeddings_sentence.py`
**Purpose**: Generate embeddings using `sentence-transformers` (cost-effective)

**Key Functions**:
- `generate_embedding(text)` - Generate embedding for single text
- `generate_embeddings_batch(texts)` - Batch embedding generation
- `get_embedding_dimension()` - Returns 384 for `all-MiniLM-L6-v2`

### 2. `embeddings_module.py`
**Purpose**: Generate embeddings using Google Gemini models (optional)

### 3. `document_preprocessor.py`
**Purpose**: Preprocess and split documents into chunks

**Key Functions**:
- `split_text(text)` - Split text into chunks
- `clean_text(text)` - Clean and normalize text
- `process_document(text)` - Complete preprocessing pipeline
- `process_from_file(file_path)` - Load and process from file

### 4. `vector_store.py`
**Purpose**: Manage vector storage in Pinecone

**Key Functions**:
- `upsert_vectors(texts, embeddings)` - Insert/update vectors
- `query_vectors(query_embedding, top_k)` - Query similar vectors
- `get_index_stats()` - Get index statistics
- `delete_all_vectors()` - Clear all vectors (with caution!)

### 5. `retriever.py`
**Purpose**: Handle retrieval logic for RAG

**Key Functions**:
- `retrieve(query, top_k)` - Retrieve relevant chunks
- `retrieve_with_score_threshold(query, min_score)` - Filter by score
- `get_retrieval_info(query)` - Get detailed retrieval info

### 6. `response_generator.py`
**Purpose**: Generate responses using LLM

**Key Functions**:
- `generate_response(query, context_chunks)` - Generate answer
- `generate_response_with_scores()` - Include score metadata
- `generate_custom_prompt_response()` - Custom instructions

### 7. `modular_rag_pipeline.py`
**Purpose**: Orchestrate all modules into a complete RAG system

**Key Functions**:
- `ingest_documents(file_path)` - Process and store documents
- `query(question, top_k)` - Query the knowledge base
- `chat(question, top_k)` - Simple chat interface
- `get_pipeline_info()` - Get pipeline configuration

### 8. `pinecone_multi_index.py`
**Purpose**: Ensure/create required Pinecone indexes for the agentic router

Indexes (for `all-MiniLM-L6-v2`, 384-d, cosine):
- `orders-index`
- `products-index`
- `support-index`

### 9. `ingest_knowledgebase.py`
**Purpose**: Ingest CSV files into the respective Pinecone indexes

Mappings:
- `knowledgebase/order.csv` → `orders-index`
- `knowledgebase/product.csv` → `products-index`
- `knowledgebase/support_catalogue.csv` → `support-index`

### 10. `agentic_single.py`
**Purpose**: Single-agent LangGraph router that:
- Routes query to relevant index(es)
- Retrieves and merges results
- Checks sufficiency across domains
- Generates final answer using the LLM

## 🚀 Usage (Modular RAG)

### Basic Usage

```python
from modular_rag_pipeline import ModularRAGPipeline

# Initialize the pipeline
rag = ModularRAGPipeline(index_name="nikhil-agent-module-gemini")

# Ingest documents
rag.ingest_documents("sample.txt")

# Query the system
result = rag.query("What is attention?")
print(result['answer'])

# Or use the chat interface
rag.chat("How does attention work?")
```

### Advanced Usage

```python
# Use individual modules
from embeddings_module import EmbeddingsModule
from document_preprocessor import DocumentPreprocessor
from vector_store import VectorStore

# Custom preprocessing
preprocessor = DocumentPreprocessor(chunk_size=500, chunk_overlap=100)
chunks = preprocessor.process_from_file("document.txt")

# Custom embedding
embeddings_module = EmbeddingsModule()
embeddings = embeddings_module.generate_embeddings_batch(chunks)

# Custom vector operations
vector_store = VectorStore("my-index")
vector_store.upsert_vectors(chunks, embeddings)
```

## 📋 Requirements

```bash
pip install langchain langchain-google-genai pinecone-client python-dotenv sentence-transformers langgraph
```

## 🔑 Environment Variables

Create a `.env` file with:

```bash
PC_API_KEY=your_pinecone_api_key
GOOGLE_API_KEY=your_google_api_key
```

## 🧭 Agentic Mode (Single-Agent Router)

This repo now includes a single-agent LangGraph that routes queries to the correct Pinecone index(es), retrieves sufficient context, and uses the existing `ResponseGenerator` to answer.

- Indexes:
  - `orders-index` (384d, cosine)
  - `products-index` (384d, cosine)
  - `support-index` (384d, cosine)

- Embeddings: `sentence-transformers` `all-MiniLM-L6-v2` (dimension 384)

### Setup (first time)

```bash
# 1) Ensure indexes exist
python -c "from pinecone_multi_index import PineconeIndexManager; PineconeIndexManager().ensure_indexes()"

# 2) Ingest knowledgebase CSVs into their respective indexes
python ingest_knowledgebase.py

# 3) Run the agent
python agentic_single.py
```

The agent will:
- Route to one or more indexes based on the query
- Retrieve top results per routed index
- Check sufficiency across requested domains
- Generate a final answer with the existing LLM

### Example queries

```text
What is the price and description of the Wireless Mouse?
Show order status for order_id 5003 and details of the product.
How do I return a product and what is the warranty on electronics?
```

## ✨ Features

- ✅ **Modular Design**: Each component is independent and reusable
- ✅ **Gemini Integration**: Uses Google Gemini 2.5 Flash for LLM
- ✅ **Sentence-Transformers Embeddings**: Default embedding via `all-MiniLM-L6-v2` (384-d, free tier-friendly)
- ✅ **Pinecone Storage**: Efficient vector storage and retrieval
- ✅ **LangChain**: Powered by LangChain for text processing
- ✅ **Clean Architecture**: Separated concerns for maintainability
- ✅ **Error Handling**: Robust error handling throughout
- ✅ **Detailed Logging**: Progress updates and statistics

## 🎯 Example Workflow

1. **Initialize**: Create pipeline instance
2. **Ingest**: Process and store documents
3. **Query**: Ask questions and get answers
4. **Iterate**: Chain multiple queries

## 📊 Output Example

```
🚀 Initializing Modular RAG Pipeline
============================================================
✅ Embeddings module initialized with model: models/gemini-embedding-001
✅ Document preprocessor initialized with chunk_size=200, overlap=50
✅ Vector store initialized with index: nikhil-agent-module-gemini
✅ Retriever initialized
✅ Response generator initialized with model: gemini-2.5-flash
============================================================
✅ Modular RAG Pipeline Ready!
============================================================

📄 Ingesting documents from: sample.txt
------------------------------------------------------------
✅ Created 10 chunks
🔗 Generating embeddings...
✅ Generated 10 embeddings
💾 Storing vectors in Pinecone...
✅ Upserted 10 vectors to index
✅ Document ingestion complete!
📊 Index stats: 10 total vectors

🔍 Processing Query: What is attention?
------------------------------------------------------------
📚 Retrieving relevant context...
✅ Found 3 relevant chunks
📊 Relevance Scores:
  1. Score: 0.8542 - Attention is an extraordinary cognitive...
  2. Score: 0.8321 - Yet, it is attention that acts as an...
  3. Score: 0.8105 - Developing strong attentional control...
💬 Generating answer...
✅ Query processed successfully!

❓ Question: What is attention?
💡 Answer: Attention is a cognitive mechanism that acts as an
internal spotlight, selectively illuminating information relevant
to our goals while relegating the rest to the periphery...
📊 Relevance: 0.8323 (avg)
```

## 🛠️ Customization

- **Chunk Size**: Adjust in `document_preprocessor.py`
- **Top K**: Change retrieval count in `query()` calls
- **Temperature**: Modify in `response_generator.py`
- **Models**: Switch embedding/LLM models in respective modules

## 📝 License

Free to use and modify for your projects!
