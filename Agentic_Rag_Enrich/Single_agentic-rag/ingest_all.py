import os
import glob
from modular_rag_pipeline import ModularRAGPipeline
from dotenv import load_dotenv

def ingest_all():
    print("="*60)
    print("📥 Starting Bulk Data Ingestion")
    print("="*60)
    
    # Load environment variables
    load_dotenv()
    
    # Initialize the pipeline
    # It will use PINECONE_INDEX_NAME from .env by default now
    rag = ModularRAGPipeline()
    
    knowledgebase_dir = "knowledgebase"
    files = glob.glob(os.path.join(knowledgebase_dir, "*.csv")) + glob.glob(os.path.join(knowledgebase_dir, "*.txt"))
    
    if not files:
        print(f"❌ No files found in {knowledgebase_dir}")
        return

    print(f"Found {len(files)} files to ingest.")
    
    for file_path in files:
        print(f"\n📂 Processing: {file_path}")
        success = rag.ingest_documents(file_path)
        if success:
            print(f"✅ Successfully ingested {file_path}")
        else:
            print(f"❌ Failed to ingest {file_path}")
            
    print("\n" + "="*60)
    stats = rag.vector_store.get_index_stats()
    print(f"📊 Final Ingestion Summary: {stats.total_vector_count if stats else 'Error getting stats'} total vectors in index")
    print("="*60)

if __name__ == "__main__":
    ingest_all()
