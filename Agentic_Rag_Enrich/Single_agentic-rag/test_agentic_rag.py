"""
Testing Script for Agentic RAG
"""
from agentic_rag import AgenticRAG
import os
from dotenv import load_dotenv

load_dotenv()

def test_agentic_rag():
    print("🧪 Starting Agentic RAG Test Suite")
    print("-" * 60)
    
    agent = AgenticRAG()
    
    # Test queries
    queries = [
        "What is the function of the Transformer encoder and where is masked multi-head attention used?",
        "Compare the role of the encoder and decoder in a Transformer model."
    ]
    
    for q in queries:
        print(f"\n🚀 TEST CASE: {q}")
        try:
            result = agent.query(q)
            
            print("\n✅ Final Result:")
            print(f"Answer: {result['answer'][:300]}...")
            print(f"Loops: {result['loops']}")
            print(f"Critic Status: {result['report'].get('is_verified', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Test Failed: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n🏁 Test Suite Finished")

if __name__ == "__main__":
    # Ensure PINECONE_INDEX_NAME is set
    if not os.getenv("PINECONE_INDEX_NAME"):
        print("⚠️ PINECONE_INDEX_NAME not found in .env, using default.")
        
    test_agentic_rag()
