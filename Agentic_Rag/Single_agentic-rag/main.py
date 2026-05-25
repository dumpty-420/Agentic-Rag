from fastapi import FastAPI
import uvicorn
from route.rag_route import router as rag_router
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Single Agentic RAG API",
    description="API for querying and ingesting documents into the Single Agentic RAG system.",
    version="1.0.0"
)

# Include the RAG router
app.include_router(rag_router)

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Single Agentic RAG API",
        "docs_url": "/docs",
        "health_check": "/rag/health"
    }

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
