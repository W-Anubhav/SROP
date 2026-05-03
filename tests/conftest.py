"""
Test fixtures.

Key fixtures:
- `client`: async test client with in-memory SQLite DB
- `mock_adk`: patches the ADK root agent so tests don't hit the real LLM
- `seeded_db`: DB with a test user and session pre-created
"""
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base
from app.database import get_db
from main.main2 import app


TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def setup_test_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db() -> AsyncSession:
    async with TestSessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def client(db):
    """Async test client with DB overridden to in-memory SQLite."""
    app.dependency_overrides[get_db] = lambda: db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def mock_adk(monkeypatch):
    async def mock_run_async(self, session_id, user_message, state_dict=None):
        from main.agents import ADKEvent
        if "rotate" in user_message.lower():
            yield ADKEvent("tool_call", tool_name="search_docs", tool_args={"query": "rotate deploy key"})
            yield ADKEvent("retrieved_chunks", chunks=["chunk_abc123"])
            yield ADKEvent("tool_result", tool_name="search_docs", result="To rotate a deploy key...")
            class MockContent:
                parts = [type('Part', (), {'text': 'To rotate a deploy key...'})]
            yield ADKEvent("final_response", author="knowledge", content=MockContent())
        else:
            class MockContent:
                parts = [type('Part', (), {'text': f'Your plan tier is {state_dict.get("plan_tier")}'})]
            yield ADKEvent("final_response", author="srop_root", content=MockContent())

    monkeypatch.setattr("main.agents.InMemoryRunner.run_async", mock_run_async)
