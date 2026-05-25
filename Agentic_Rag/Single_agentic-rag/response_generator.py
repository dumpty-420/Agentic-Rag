"""
Response Generator Module - Handles prompt creation and LLM generation
"""
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from typing import Dict

load_dotenv()

class ResponseGenerator:
    def __init__(self, model_name="gemini-2.5-flash", temperature=0.7):
        """
        Initialize the response generator
        
        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature for response generation
        """
        self.model_name = model_name
        self.temperature = temperature
        
        self.llm = GoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature
        )
        
        # Define the RAG prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant. Based on the following context, please answer the question accurately.

Context:
{context}

Question: {question}

Instructions:
- Answer based only on the provided context
- If the context doesn't contain enough information, say so
- Be concise and accurate
- Cite specific details from the context when relevant

Answer:"""
        )
        
        print(f"✅ Response generator initialized with model: {model_name}")
    
    def generate_response(self, query: str, context_chunks: list) -> str:
        """
        Generate a response using the LLM
        
        Args:
            query: User's question
            context_chunks: List of retrieved context chunks
            
        Returns:
            Generated response text
        """
        try:
            # Extract text from context chunks
            context_texts = [chunk.get('text', chunk) if isinstance(chunk, dict) else chunk for chunk in context_chunks]
            context = "\n\n".join(context_texts)
            
            # Create prompt
            prompt = self.prompt_template.format(context=context, question=query)
            
            # Generate response
            response = self.llm.invoke(prompt)
            
            return response
            
        except Exception as e:
            return f"Error generating response: {e}"
    
    def generate_response_with_scores(self, query: str, context_chunks_with_scores: list) -> Dict:
        """
        Generate a response and include score information
        
        Args:
            query: User's question
            context_chunks_with_scores: List of retrieved chunks with scores
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            # Extract just the texts for context
            context_texts = [chunk.get('text', '') for chunk in context_chunks_with_scores]
            
            # Generate response
            response = self.generate_response(query, context_texts)
            
            # Extract scores
            scores = [chunk.get('score', 0) for chunk in context_chunks_with_scores]
            
            return {
                'response': response,
                'query': query,
                'num_sources': len(context_chunks_with_scores),
                'max_score': max(scores) if scores else 0,
                'min_score': min(scores) if scores else 0,
                'avg_score': sum(scores) / len(scores) if scores else 0
            }
            
        except Exception as e:
            return {
                'response': f"Error: {e}",
                'query': query,
                'error': str(e)
            }
    
    def generate_custom_prompt_response(self, query: str, context_chunks: list, custom_instructions: str = "") -> str:
        """
        Generate response with custom instructions
        
        Args:
            query: User's question
            context_chunks: List of retrieved context chunks
            custom_instructions: Additional instructions to include in prompt
            
        Returns:
            Generated response text
        """
        try:
            # Extract text from context chunks
            context_texts = [chunk.get('text', chunk) if isinstance(chunk, dict) else chunk for chunk in context_chunks]
            context = "\n\n".join(context_texts)
            
            # Create custom prompt
            custom_prompt = f"""You are a helpful AI assistant. Based on the following context, please answer the question.

Context:
{context}

Question: {query}

{custom_instructions}

Answer:"""
            
            # Generate response
            response = self.llm.invoke(custom_prompt)
            
            return response
            
        except Exception as e:
            return f"Error generating response: {e}"
