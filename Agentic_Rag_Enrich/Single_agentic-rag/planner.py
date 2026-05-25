"""
Planner Module - Handles task decomposition and sub-query generation
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict
import os
import json
from dotenv import load_dotenv

load_dotenv()

class Planner:
    def __init__(self, model_name="gemini-2.0-flash", temperature=0.2):
        """
        Initialize the planner
        
        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature for response generation
        """
        self.model_name = model_name
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature
        )
        
        self.planner_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert query planner for a RAG system. 
Your task is to decompose a complex user query into 1-3 simpler sub-questions.
These sub-questions will be used to retrieve relevant information from a knowledge base.

Guidelines:
- If the query is simple, return it as the only sub-question.
- If the query is complex (e.g., comparative, multi-step), break it down logically.
- Return only a JSON object with the key 'sub_questions' which is a list of strings.
- Avoid using conversational text, only return the JSON."""),
            ("user", "Original Query: {query}")
        ])
        
        print(f"✅ Planner initialized with model: {model_name}")

    def decompose(self, query: str) -> List[str]:
        """
        Decompose a query into sub-questions
        
        Args:
            query: The user's original query
            
        Returns:
            List of sub-questions
        """
        try:
            print(f"🧩 Decomposing query: '{query}'")
            chain = self.planner_prompt | self.llm
            response = chain.invoke({"query": query})
            
            # Clean response to ensure it's valid JSON
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            data = json.loads(content)
            sub_questions = data.get("sub_questions", [query])
            
            print(f"✅ Generated {len(sub_questions)} sub-questions")
            for i, sq in enumerate(sub_questions, 1):
                print(f"   {i}. {sq}")
                
            return sub_questions
        except Exception as e:
            print(f"⚠️ Planner error: {e}. Falling back to original query.")
            return [query]

if __name__ == "__main__":
    planner = Planner()
    test_query = "Compare the implementation of multi-head attention in Transformers with standard attention."
    questions = planner.decompose(test_query)
    print(questions)
