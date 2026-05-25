import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langgraph.graph import Graph
from typing import Dict, Any
from pinecone import Pinecone

class SimpleStudentRAG:
    def __init__(self):
        # Initialize embeddings and LLM
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PC_API_KEY"))
        self.index_name = "students-demo-index"
        
    def setup_vectorstore(self):
        """Create and populate the vector store with student data"""
        print("🚀 Setting up vector store...")
        
        # Create index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=768,  # Gemini embeddings are 768-dimensional
                metric="cosine",
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"
                    }
                }
            )
            print(f"✅ Created index: {self.index_name}")
        
        # Load and process student data
        loader = TextLoader("students.txt")
        documents = loader.load()
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=200,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(documents)
        print(f"📄 Split into {len(chunks)} chunks")
        
        # Create vector store
        self.vectorstore = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            index_name=self.index_name
        )
        print("✅ Vector store populated with student data")
    
    def build_rag_graph(self):
        """Build the LangGraph RAG pipeline"""
        print("🔨 Building LangGraph RAG pipeline...")
        
        # Define the graph
        workflow = Graph()
        
        # Add nodes
        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("generate", self.generate_answer)
        
        # Define edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "__end__")
        
        # Compile the graph
        self.graph = workflow.compile()
        print("✅ LangGraph RAG pipeline built successfully")
        return self.graph
    
    def retrieve_documents(self, state: Dict[str, Any]):
        """Retrieve relevant documents based on query"""
        query = state["question"]
        print(f"🔍 Retrieving documents for: {query}")
        
        # Perform similarity search
        docs = self.vectorstore.similarity_search(query, k=3)
        
        # Format context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        return {
            "context": context,
            "documents": docs
        }
    
    def generate_answer(self, state: Dict[str, Any]):
        """Generate answer using retrieved context"""
        question = state["question"]
        context = state["context"]
        
        print("🤖 Generating answer with Gemini...")
        
        # Create prompt with context
        prompt = f"""Based on the following student information, answer the question.

Student Data:
{context}

Question: {question}

Answer: """
        
        # Generate response
        response = self.llm.invoke(prompt)
        
        return {
            "answer": response.content,
            "question": question,
            "context_used": context
        }
    
    def query(self, question: str):
        """Execute the RAG pipeline for a question"""
        print(f"\n🎯 Question: {question}")
        print("-" * 50)
        
        # Run the graph
        result = self.graph.invoke({"question": question})
        
        print(f"\n✅ Answer: {result['answer']}")
        print(f"\n📚 Context used: {result['context_used']}")
        
        return result

# Demo execution
def main():
    # Initialize the RAG system
    rag_system = SimpleStudentRAG()
    
    # Setup vector store (run this once)
    rag_system.setup_vectorstore()
    
    # Build the graph
    graph = rag_system.build_rag_graph()
    
    # Demo queries
    demo_questions = [
        "Which students are in grade 10?",
        "Who has the highest GPA?",
        "Which students are taking Math?",
        "Tell me about students taking Biology"
    ]
    
    print("\n" + "="*60)
    print("🎓 STUDENT RAG DEMO WITH LANGGRAPH")
    print("="*60)
    
    for question in demo_questions:
        rag_system.query(question)
        print("\n" + "="*60)

if __name__ == "__main__":
    main()