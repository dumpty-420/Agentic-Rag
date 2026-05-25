"""
Tools Module - Wraps available actions/tools for the Agent
"""
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# Import the existing retriever
from retriever import Retriever
from vector_store import VectorStore
from embeddings_module import EmbeddingsModule

load_dotenv()

class ToolBox:
    def __init__(self, index_name: str = None):
        """
        Initialize the toolbox with necessary services
        """
        if index_name is None:
            index_name = os.getenv("PINECONE_INDEX_NAME", "nikhil-agent-module-gemini")
            
        # Initialize the underlying RAG components
        self.embeddings_module = EmbeddingsModule()
        self.vector_store = VectorStore(index_name)
        self.retriever = Retriever(self.vector_store, self.embeddings_module)
        
        print(f"🛠️ ToolBox initialized with index: {index_name}")

    def vector_retrieval(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Tool: Search the knowledge base for a specific query
        """
        print(f"📚 Tool Call: 'vector_retrieval' for query: '{query}'")
        results = self.retriever.retrieve(query, top_k)
        
        # Format results for the agent
        formatted_results = []
        for res in results:
            formatted_results.append({
                'text': res.get('text', ''),
                'score': res.get('score', 0),
                'source': res.get('metadata', {}).get('source', 'unknown')
            })
            
        print(f"✅ Retrieved {len(formatted_results)} results")
        return formatted_results

    def get_tool_definitions(self) -> List[Dict]:
        """
        Return tool definitions for LLM tool-calling (not used in simple custom loop but good for future)
        """
        return [
            {
                "name": "vector_retrieval",
                "description": "Searches the internal knowledge base for relevant documents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query to look up"},
                        "top_k": {"type": "integer", "description": "Number of results to return", "default": 5}
                    },
                    "required": ["query"]
                }
            }
        ]

if __name__ == "__main__":
    toolbox = ToolBox()
    results = toolbox.vector_retrieval("masked attention")
    print(results)
