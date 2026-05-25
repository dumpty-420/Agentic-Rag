"""
Critic Module - Handles verification and reflection on generated answers
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict
import os
import json
from dotenv import load_dotenv

load_dotenv()

class Critic:
    def __init__(self, model_name="gemini-2.0-flash", temperature=0.2):
        """
        Initialize the critic
        
        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature for reflection
        """
        self.model_name = model_name
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature
        )
        
        self.critic_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a meticulous fact-checker and quality assurance agent for a RAG system. 
Your goal is to evaluate if the answer provided is accurate, grounded in the context, and fully answers the user's question.

Evaluation Criteria:
1. Grounding: Is every fact in the answer supported by the provided context?
2. Completeness: Does the answer fully address the user question?
3. Clarity: Is the answer clear and not contradictory?

Returns a JSON object:
- 'is_verified': boolean (true if everything is fine)
- 'feedback': string (reason for failure or 'OK' if verified)
- 'missing_info': list of strings (what else is needed if any)
- 'suggestion': string (how to improve or 'none' if fine)"""),
            ("user", "Question: {query}\n\nContext:\n{context}\n\nGenerated Answer:\n{answer}")
        ])
        
        print(f"✅ Critic initialized with model: {model_name}")

    def verify(self, query: str, context: str, answer: str) -> Dict:
        """
        Reflect on the quality of the answer
        """
        try:
            print("🧐 Critic is evaluating the answer...")
            chain = self.critic_prompt | self.llm
            response = chain.invoke({"query": query, "context": context, "answer": answer})
            
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            data = json.loads(content)
            
            if data.get('is_verified', False):
                print("✅ Answer verified by Critic")
            else:
                print(f"❌ Answer failed verification: {data.get('feedback', 'No feedback provided')}")
                
            return data
        except Exception as e:
            print(f"⚠️ Critic evaluation error: {e}")
            return {
                'is_verified': True, 
                'feedback': f'Critic Error: {e}', 
                'missing_info': [], 
                'suggestion': 'Proceed with caution'
            }

if __name__ == "__main__":
    critic = Critic()
    test_query = "What is the capital of France?"
    test_context = "Paris is the capital of France."
    test_answer = "The capital of France is Paris."
    
    result = critic.verify(test_query, test_context, test_answer)
    print(result)
