"""
Test script for the Modular RAG System
"""
from modular_rag_pipeline import ModularRAGPipeline

def main():
    print("="*70)
    print("🧪 Testing Modular RAG System")
    print("="*70)
    
    # Initialize the pipeline
    rag = ModularRAGPipeline()
    
    # Show pipeline info
    print("\n📋 Pipeline Configuration:")
    print("-" * 70)
    info = rag.get_pipeline_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Ingest documents (optional - skip if already indexed)
    print("\n📚 Document Ingestion:")
    print("-" * 70)
    ingest = input("Do you want to re-ingest documents? (y/n): ").lower()
    if ingest == 'y':
        rag.ingest_documents("sample.txt")
    else:
        print("⏭️ Skipping ingestion (using existing indexed data)")
    
    # Interactive query loop
    print("\n" + "="*70)
    print("💬 Interactive RAG Queries (type 'exit' to quit)")
    print("="*70)
    
    while True:
        query = input("\n🔍 Your question: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not query:
            continue
        
        # Process query
        result = rag.chat(query)
        
        # Show additional info
        if 'scores' in result:
            print(f"\n📊 Retrieved {result['num_sources']} sources")
            print(f"   Score range: {result['min_score']:.4f} - {result['max_score']:.4f}")

if __name__ == "__main__":
    main()
