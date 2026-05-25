"""
Reranker Module - Handles scoring and filtering of retrieved chunks
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict
import os
import json
from dotenv import load_dotenv

load_dotenv()

class Reranker:
    def __init__(self, model_name="gemini-2.0-flash", temperature=0.1, threshold=0.6):
        """
        Initialize the reranker
        
        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature for LLM scoring
            threshold: Minimum score to keep a chunk
        """
        self.model_name = model_name
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature
        )
        self.threshold = threshold
        
        self.rerank_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a highly capable AI specialized in information relevance. 
Your task is to rank the relevance of a retrieved context chunk to a user's question.

Guidelines:
- Score the relevance from 0 to 1, where 1 is highly relevant and 0 is completely irrelevant.
- Provide a brief justification for the score.
- Return only a JSON object with the keys: 'relevance_score' (float) and 'justification' (string)."""),
            ("user", "Question: {query}\n\nContext Chunk:\n{chunk}")
        ])
        
        print(f"✅ Reranker initialized with model: {model_name} (Threshold: {threshold})")

    def score_chunk(self, query: str, chunk_text: str) -> Dict:
        """
        Score a single chunk for relevance
        """
        try:
            chain = self.rerank_prompt | self.llm
            response = chain.invoke({"query": query, "chunk": chunk_text})
            
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            data = json.loads(content)
            return data
        except Exception as e:
            print(f"⚠️ Scoring error: {e}")
            return {"relevance_score": 0.5, "justification": "Error in scoring, default value assigned."}

    def rerank(self, query: str, chunks: List[Dict], top_n=3) -> List[Dict]:
        """
        Rerank a list of chunks based on LLM-based relevance scoring
        
        Args:
            query: The question for relevance check
            chunks: List of dictionaries with 'text' and 'score' (from retriever)
            top_n: Number of chunks to return
            
        Returns:
            Reranked list of chunks
        """
        print(f"📊 Reranking {len(chunks)} chunks for query: '{query}'")
        
        reranked_chunks = []
        for chunk in chunks:
            chunk_text = chunk.get('text', '')
            result = self.score_chunk(query, chunk_text)
            
            llm_score = result.get('relevance_score', 0)
            
            # Combine scores (retrieval score + LLM score)
            # Retrieval score is usually normalized in Pinecone or similar
            # For simplicity, we just use LLM score for reranking
            chunk['llm_score'] = llm_score
            chunk['rerank_justification'] = result.get('justification', '')
            
            if llm_score >= self.threshold:
                reranked_chunks.append(chunk)
        
        # Sort by llm_score descending
        reranked_chunks.sort(key=lambda x: x.get('llm_score', 0), reverse=True)
        
        # Keep only top_n
        final_chunks = reranked_chunks[:top_n]
        print(f"✅ Rerank completed. Kept {len(final_chunks)} chunks above {self.threshold} threshold.")
        
        for i, chunk in enumerate(final_chunks, 1):
            score = chunk.get('llm_score', 0)
            print(f"   {i}. Score: {score:.4f}")
            
        return final_chunks

if __name__ == "__main__":
    reranker = Reranker()
    test_query = "What is masked attention?"
    test_chunks = [
        {"text": "Attention mechanisms are used in Transformers.", "score": 0.8},
        {"text": "Masked attention ensures that the prediction for position i can depend only on the known outputs at positions less than i.", "score": 0.9}
    ]
    results = reranker.rerank(test_query, test_chunks)
    print(results)
