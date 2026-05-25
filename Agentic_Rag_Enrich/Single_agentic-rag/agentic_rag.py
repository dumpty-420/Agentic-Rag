"""
Agentic RAG Orchestrator - Main entry point connecting all modules using LangGraph
"""
import os
from typing import Dict, Any, List, TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Import local modules
from planner import Planner
from tools import ToolBox
from reranker import Reranker
from response_generator import ResponseGenerator
from critic import Critic
from memory import Memory

load_dotenv()

# Define the state of our agent
class AgentState(TypedDict):
    question: str
    sub_questions: List[str]
    context_chunks: Annotated[List[Dict], operator.add]
    final_context: str
    answer: str
    critic_report: Dict[str, Any]
    memory_context: str
    loop_count: int

class AgenticRAG:
    def __init__(self, index_name: str = None):
        """
        Initialize the Agentic RAG system
        """
        print("\n" + "🤖"*30)
        print("🚀 INITIALIZING AGENTIC RAG SYSTEM")
        print("🤖"*30 + "\n")
        
        # Initialize components
        self.planner = Planner()
        self.toolbox = ToolBox(index_name)
        self.reranker = Reranker()
        self.generator = ResponseGenerator()
        self.critic = Critic()
        self.memory = Memory()
        
        self.max_loops = 3
        
        # Build the graph
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile()
        
        print("\n✅ Agentic RAG System is ONLINE!")
        print("-" * 60)

    def _create_workflow(self) -> StateGraph:
        """
        Define the LangGraph workflow
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("plan", self.node_plan)
        workflow.add_node("retrieve", self.node_retrieve)
        workflow.add_node("rerank", self.node_rerank)
        workflow.add_node("synthesize", self.node_synthesize)
        workflow.add_node("critique", self.node_critique)
        
        # Define edges
        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "synthesize")
        workflow.add_edge("synthesize", "critique")
        
        # Conditional edge for the reflection loop
        workflow.add_conditional_edges(
            "critique",
            self._should_continue,
            {
                "continue": "plan",
                "end": END
            }
        )
        
        return workflow

    # ============ Nodes ============

    def node_plan(self, state: AgentState) -> Dict:
        """Analyze question and generate sub-questions or plan update"""
        query = state["question"]
        loop = state.get("loop_count", 0)
        
        if loop > 0:
            # If we are in a loop, provide the critic suggestion to the planner
            report = state.get("critic_report", {})
            suggestion = report.get("suggestion", "Provide more detail")
            missing = report.get("missing_info", [])
            plan_query = f"{query}. NOTE: Previous attempt failed. Suggestion: {suggestion}. Missing info: {missing}"
            sub_questions = self.planner.decompose(plan_query)
        else:
            sub_questions = self.planner.decompose(query)
            
        return {
            "sub_questions": sub_questions, 
            "loop_count": loop + 1,
            "memory_context": self.memory.get_context_for_llm()
        }

    def node_retrieve(self, state: AgentState) -> Dict:
        """Retrieve context for each sub-question"""
        sub_questions = state["sub_questions"]
        all_chunks = []
        
        for sq in sub_questions:
            chunks = self.toolbox.vector_retrieval(sq, top_k=3)
            all_chunks.extend(chunks)
            
        return {"context_chunks": all_chunks}

    def node_rerank(self, state: AgentState) -> Dict:
        """Rerank and filter retrieved chunks"""
        query = state["question"]
        chunks = state["context_chunks"]
        
        # Unique chunks by text to avoid redundancy
        seen = set()
        unique_chunks = []
        for c in chunks:
            if c['text'] not in seen:
                unique_chunks.append(c)
                seen.add(c['text'])
        
        # Rerank for the main question
        reranked = self.reranker.rerank(query, unique_chunks, top_n=5)
        
        # Create final context string
        final_context = "\n\n".join([c['text'] for c in reranked])
        
        return {"final_context": final_context}

    def node_synthesize(self, state: AgentState) -> Dict:
        """Synthesize final response from context"""
        query = state["question"]
        context = state["final_context"]
        mem_context = state.get("memory_context", "")
        
        prompt_with_mem = f"Conversation History:\n{mem_context}\n\nBased on history and new context...\nQuestion: {query}"
        
        answer = self.generator.generate_custom_prompt_response(
            query=query, 
            context_chunks=[context], 
            custom_instructions="Be thorough but concise. Use the history if helpful."
        )
        
        return {"answer": str(answer)}

    def node_critique(self, state: AgentState) -> Dict:
        """Evaluate the quality of the answer"""
        query = state["question"]
        context = state["final_context"]
        answer = state["answer"]
        
        report = self.critic.verify(query, context, answer)
        return {"critic_report": report}

    # ============ Conditional Edge Logic ============

    def _should_continue(self, state: AgentState) -> str:
        """Decide if we should end or loop back"""
        report = state["critic_report"]
        loop = state["loop_count"]
        
        if report.get("is_verified", False) or loop >= self.max_loops:
            if loop >= self.max_loops:
                print(f"⚠️ Max loops reached ({self.max_loops}). Terminating.")
            return "end"
        
        print(f"♻️ Critic requested changes. Looping back (Attempt {loop + 1}/{self.max_loops})")
        return "continue"

    # ============ Main Invoke ============

    def query(self, question: str) -> Dict:
        """
        Execute the agentic flow for a given question
        """
        print(f"\n🎯 Query Processing Initiated: '{question}'")
        print("=" * 60)
        
        initial_state = {
            "question": question,
            "sub_questions": [],
            "context_chunks": [],
            "final_context": "",
            "answer": "",
            "critic_report": {},
            "memory_context": "",
            "loop_count": 0
        }
        
        final_state = self.app.invoke(initial_state)
        
        # Update memory
        self.memory.add_interaction(question, final_state["answer"])
        
        print("\n✅ Final Result Ready!")
        print("=" * 60)
        
        return {
            "query": question,
            "answer": final_state["answer"],
            "report": final_state["critic_report"],
            "loops": final_state["loop_count"]
        }

if __name__ == "__main__":
    agent = AgenticRAG()
    res = agent.query("Where is masked multi-head attention used and why?")
    print(f"\n💡 Answer:\n{res['answer']}")
