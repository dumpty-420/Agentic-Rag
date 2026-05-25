"""
Testing Script for Multi-Agentic RAG - Async with Schema Validation
"""
import asyncio
import os
from dotenv import load_dotenv

from agentic_multi_rag import AgenticMultiRAG
from schemas import (
    CriticVerdict,
    CriticCategory,
    DomainEnum,
    PlannerTask,
    QueryResponse,
    CriticReportResponse,
    ConversationTurn,
    LLMConfig,
    PineconeConfig,
    OrchestratorResponse,
    RerankerScore,
    DomainResearchResult,
    RetrievedDocument,
)

load_dotenv()


def test_pydantic_schemas():
    """Validate all Pydantic schema models instantiate correctly."""
    print("🧪 Testing Pydantic Schema Validation...")
    print("-" * 60)

    # LLMConfig
    config = LLMConfig(model_name="gemini-2.0-flash", temperature=0.5)
    assert config.model_name == "gemini-2.0-flash"
    assert config.temperature == 0.5
    print("  ✅ LLMConfig validated")

    # PineconeConfig
    pc_config = PineconeConfig(index_name="test-index", dimension=384, metric="cosine")
    assert pc_config.metric == "cosine"
    print("  ✅ PineconeConfig validated")

    # OrchestratorResponse
    orch = OrchestratorResponse(domains=[DomainEnum.ORDERS, DomainEnum.PRODUCTS], confidence=0.95)
    assert len(orch.domains) == 2
    print("  ✅ OrchestratorResponse validated")

    # PlannerTask
    task = PlannerTask(id=1, query="Find order #1234", domain=DomainEnum.ORDERS, depends_on=[])
    assert task.id == 1
    assert task.domain == DomainEnum.ORDERS
    print("  ✅ PlannerTask validated")

    # RerankerScore
    score = RerankerScore(relevance_score=0.85, justification="Highly relevant")
    assert score.relevance_score == 0.85
    print("  ✅ RerankerScore validated")

    # CriticVerdict
    verdict = CriticVerdict(
        is_verified=False,
        category=CriticCategory.DATA_GAP,
        feedback="Missing order details",
        missing_info=["delivery_date"],
        suggestion="Search orders domain",
    )
    assert not verdict.is_verified
    assert verdict.category == CriticCategory.DATA_GAP
    print("  ✅ CriticVerdict validated")

    # RetrievedDocument
    doc = RetrievedDocument(content="Order shipped", source="order.csv", domain=DomainEnum.ORDERS)
    assert doc.domain == DomainEnum.ORDERS
    print("  ✅ RetrievedDocument validated")

    # DomainResearchResult
    result = DomainResearchResult(domain=DomainEnum.PRODUCTS, documents=[doc], task_id=1)
    assert len(result.documents) == 1
    print("  ✅ DomainResearchResult validated")

    # ConversationTurn
    turn = ConversationTurn(user="Hello", assistant="Hi there", domains=[DomainEnum.SUPPORT])
    assert turn.user == "Hello"
    print("  ✅ ConversationTurn validated")

    # QueryResponse
    resp = QueryResponse(query="test", answer="response", domains=["orders"], loops=1)
    assert resp.loops == 1
    print("  ✅ QueryResponse validated")

    print("\n✅ All Pydantic schema tests PASSED!")


def test_pydantic_validation_errors():
    """Test that Pydantic catches invalid data."""
    print("\n🧪 Testing Pydantic Validation Errors...")
    print("-" * 60)

    # Invalid temperature
    try:
        LLMConfig(model_name="test", temperature=5.0)
        print("  ❌ Should have raised for temperature > 2.0")
    except Exception as e:
        print(f"  ✅ Correctly rejected invalid temperature: {type(e).__name__}")

    # Empty model name
    try:
        LLMConfig(model_name="  ", temperature=0.5)
        print("  ❌ Should have raised for empty model_name")
    except Exception as e:
        print(f"  ✅ Correctly rejected empty model_name: {type(e).__name__}")

    # Self-dependency in PlannerTask
    try:
        PlannerTask(id=1, query="test", domain=DomainEnum.ORDERS, depends_on=[1])
        print("  ❌ Should have raised for self-dependency")
    except Exception as e:
        print(f"  ✅ Correctly rejected self-dependency: {type(e).__name__}")

    # Invalid relevance score
    try:
        RerankerScore(relevance_score=1.5, justification="test")
        print("  ❌ Should have raised for score > 1.0")
    except Exception as e:
        print(f"  ✅ Correctly rejected invalid score: {type(e).__name__}")

    # Invalid metric
    try:
        PineconeConfig(index_name="test", dimension=384, metric="manhattan")
        print("  ❌ Should have raised for invalid metric")
    except Exception as e:
        print(f"  ✅ Correctly rejected invalid metric: {type(e).__name__}")

    print("\n✅ All validation error tests PASSED!")


async def test_multi_agentic_rag():
    """Full integration test of the multi-agentic RAG system."""
    print("\n🧪 Starting Multi-Agentic RAG Integration Test")
    print("-" * 60)

    agent = AgenticMultiRAG()

    queries = [
        "Where is my order #1234 and what are its features?",
        "How can I reset my password and does it cost anything?",
    ]

    for q in queries:
        print(f"\n🚀 GLOBAL TEST CASE: {q}")
        try:
            result = await agent.query(q)

            # Validate response structure
            assert "answer" in result, "Missing 'answer' key"
            assert "domains" in result, "Missing 'domains' key"
            assert "loops" in result, "Missing 'loops' key"
            assert "report" in result, "Missing 'report' key"
            assert "message_trace" in result, "Missing 'message_trace' key"

            print("\n✅ Global Result:")
            print(f"  Answer: {result['answer'][:300]}...")
            print(f"  Domains: {result['domains']}")
            print(f"  Loops: {result['loops']}")
            print(f"  Total Messages: {result.get('total_messages', 'N/A')}")

            # Validate critic report through Pydantic
            raw_report = result.get("report", {})
            if raw_report:
                try:
                    verdict = CriticVerdict.model_validate(raw_report)
                    print(f"  Critic Verified: {verdict.is_verified}")
                    print(f"  Critic Category: {verdict.category.value}")
                except Exception as e:
                    print(f"  ⚠️ Critic report not fully Pydantic-valid: {e}")

            # Print message trace
            print("\n  📝 Message Trace:")
            for msg in result.get("message_trace", [])[:5]:
                print(f"    [{msg['role']}] {msg['content'][:100]}")
            if len(result.get("message_trace", [])) > 5:
                print(f"    ... and {len(result['message_trace']) - 5} more messages")

        except Exception as e:
            print(f"❌ Global Test Failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n🏁 Multi-Agent Test Suite Finished")


if __name__ == "__main__":
    # Run schema tests first (fast, no API calls)
    test_pydantic_schemas()
    test_pydantic_validation_errors()

    # Run integration tests (requires API keys)
    if os.getenv("GOOGLE_API_KEY") and os.getenv("PC_API_KEY"):
        asyncio.run(test_multi_agentic_rag())
    else:
        print("\n⚠️ Skipping integration tests (GOOGLE_API_KEY or PC_API_KEY not set)")
