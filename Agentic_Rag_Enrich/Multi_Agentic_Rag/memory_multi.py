"""
Memory Module - Multi-Domain Session Management with Pydantic Models
"""
from typing import List, Optional
from datetime import datetime

from schemas import ConversationTurn, DomainEnum


class MultiMemory:
    def __init__(self, window_size: int = 5):
        """
        Initialize the global multi-index memory buffer.

        Args:
            window_size: Maximum number of conversation turns to retain.
        """
        self.history: List[ConversationTurn] = []
        self.window_size: int = window_size
        print(f"🧠 Global Multi-Memory initialized (Window: {window_size})")

    def add_interaction(
        self,
        user_query: str,
        ai_answer: str,
        domains_used: List[str],
    ) -> ConversationTurn:
        """
        Add a new multi-domain turn to history.

        Args:
            user_query: The user's original question.
            ai_answer: The AI-generated response.
            domains_used: List of domain name strings that were consulted.

        Returns:
            The validated ConversationTurn that was stored.
        """
        # Convert string domain names to DomainEnum, skipping invalid ones
        validated_domains: List[DomainEnum] = []
        for d in domains_used:
            try:
                validated_domains.append(DomainEnum(d))
            except ValueError:
                pass  # Skip unrecognized domain names gracefully

        turn = ConversationTurn(
            user=user_query,
            assistant=ai_answer,
            domains=validated_domains,
            timestamp=datetime.utcnow(),
        )
        self.history.append(turn)

        # Sliding window eviction
        if len(self.history) > self.window_size:
            self.history.pop(0)

        return turn

    def get_context_for_llm(self) -> str:
        """
        Format the conversation context for domain planners.

        Returns:
            A formatted string of recent conversation history.
        """
        if not self.history:
            return "No previous interaction history."

        formatted_lines: List[str] = []
        for turn in self.history:
            domain_names = ", ".join(d.value for d in turn.domains)
            formatted_lines.append(f"Query: {turn.user}")
            formatted_lines.append(f"Domains: {domain_names}")
            formatted_lines.append(f"AI Answer: {turn.assistant}")
            formatted_lines.append(f"Timestamp: {turn.timestamp.isoformat()}")
            formatted_lines.append("")  # blank line separator

        return "\n".join(formatted_lines)

    def get_recent_turns(self, n: Optional[int] = None) -> List[ConversationTurn]:
        """
        Get the N most recent conversation turns.

        Args:
            n: Number of turns to return. Defaults to all turns in the window.

        Returns:
            List of ConversationTurn objects.
        """
        if n is None:
            return list(self.history)
        return list(self.history[-n:])

    def clear(self) -> None:
        """Clear session memory."""
        self.history = []
        print("🧠 Global Memory cleared.")

    def serialize(self) -> List[dict]:
        """Serialize entire history to dicts for persistence."""
        return [turn.model_dump(mode="json") for turn in self.history]

    @classmethod
    def deserialize(cls, data: List[dict], window_size: int = 5) -> "MultiMemory":
        """Restore memory from serialized dicts."""
        mem = cls(window_size=window_size)
        for item in data:
            mem.history.append(ConversationTurn.model_validate(item))
        return mem


if __name__ == "__main__":
    # Quick smoke test
    mem = MultiMemory(window_size=3)
    t1 = mem.add_interaction("Where is order #1?", "Shipped yesterday.", ["orders"])
    t2 = mem.add_interaction("Product ABC price?", "$99.99", ["products"])
    print(mem.get_context_for_llm())
    print(f"Serialized: {mem.serialize()}")
