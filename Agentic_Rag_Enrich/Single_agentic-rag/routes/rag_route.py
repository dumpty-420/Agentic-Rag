from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import asyncio
from modular_rag_pipeline import ModularRAGPipeline

router = APIRouter(prefix="/rag", tags=["rag"])

# Initialize the pipeline
# We can use an environment variable to set the index name, or default to the one in the code
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "nikhil-agent-module-gemini")
pipeline = ModularRAGPipeline(index_name=INDEX_NAME)

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class QueryResponse(BaseModel):
    query: str
    response: str
    num_sources: int
    avg_score: float
    max_score: float
    min_score: float

class IngestRequest(BaseModel):
    file_path: str

class IngestResponse(BaseModel):
    message: str
    success: bool

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the RAG system with a question.
    """
    try:
        # Run the synchronous pipeline query in a separate thread
        result = await asyncio.to_thread(pipeline.query, request.question, request.top_k)
        
        # Map the dictionary result to the QueryResponse model
        return QueryResponse(
            query=result.get('query', request.question),
            response=result.get('response', result.get('answer', '')),
            num_sources=result.get('num_sources', 0),
            avg_score=result.get('avg_score', 0.0),
            max_score=result.get('max_score', 0.0),
            min_score=result.get('min_score', 0.0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    """
    Trigger document ingestion from a file path.
    """
    if not os.path.exists(request.file_path):
        # Check if the file exists relative to the project root or folder
        folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), request.file_path)
        if not os.path.exists(folder_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
        request.file_path = folder_path

    try:
        success = await asyncio.to_thread(pipeline.ingest_documents, request.file_path)
        if success:
            return IngestResponse(message=f"Successfully ingested {request.file_path}", success=True)
        else:
            return IngestResponse(message=f"Ingestion failed for {request.file_path}", success=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """
    Check the health and configuration of the RAG pipeline.
    """
    try:
        info = await asyncio.to_thread(pipeline.get_pipeline_info)
        return {
            "status": "healthy",
            "pipeline_info": info
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
