"""
Memory Module - Handles conversation history and session management
"""
from typing import List, Dict, Any

class Memory:
    def __init__(self, window_size=5):
        """
        Initialize the memory buffer
        
        Args:
            window_size: Number of turns to keep in memory
        """
        self.history = []
        self.window_size = window_size
        print(f"🧠 Memory initialized (Window: {window_size})")

    def add_interaction(self, user_query: str, ai_response: str):
        """
        Add a new turn to the memory history
        """
        self.history.append({
            "user": user_query,
            "assistant": ai_response
        })
        
        # Enforce window size
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def get_context_for_llm(self) -> str:
        """
        Format the history for inclusion in LLM prompts
        """
        if not self.history:
            return "No previous conversation history."
            
        formatted_history = []
        for turn in self.history:
            formatted_history.append(f"User: {turn['user']}")
            formatted_history.append(f"AI: {turn['assistant']}")
            
        return "\n".join(formatted_history)

    def clear(self):
        """
        Clear the conversation memory
        """
        self.history = []
        print("🧠 Memory cleared.")

    def get_last_n_turns(self, n=1) -> List[Dict]:
        """
        Return the last n turns
        """
        return self.history[-n:]

if __name__ == "__main__":
    mem = Memory()
    mem.add_interaction("Hello", "Hi there!")
    mem.add_interaction("How are you?", "I am a helpful assistant.")
    print(mem.get_context_for_llm())
