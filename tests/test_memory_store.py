from __future__ import annotations

import pytest

from swarm.memory.pgvector_store import PGVectorStore, EmbeddingProvider


class StubProvider(EmbeddingProvider):
    def __init__(self):
        self.calls = []
    async def embed(self, texts):
        self.calls.append(list(texts))
        # map text to simple scalar vector [len(text) % 10] repeated
        return [[float(len(t) % 10)] * 384 for t in texts]


@pytest.mark.asyncio
@pytest.mark.integration
async def test_upsert_and_search(monkeypatch):
    store = PGVectorStore(dsn='postgresql+asyncpg://postgres:agentforge@localhost:5432/vector', migrate=False)
    # patch provider
    store.provider = StubProvider()

    # Mock Session to avoid DB; simulate insert and search
    class DummySession:
        def __init__(self):
            self.executed = []
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            return False
        async def execute(self, stmt, params=None):
            self.executed.append((str(stmt), params))
            class R:
                def fetchall(self_inner):
                    class Row:
                        def __init__(self, i):
                            self.id = f"id-{i}"
                            self.scope = "s"
                            self.content = f"doc-{i}"
                            self.meta = '{}'
                            self.cosine = 0.9 - i*0.1
                            self.lexical = 0.2 + i*0.05
                    return [Row(0), Row(1), Row(2)]
            return R()
        async def commit(self):
            return None

    class DummyMaker:
        def __call__(self, *a, **k):
            return DummySession()

    store.Session = DummyMaker()

    # upsert
    n = await store.upsert_batch("ns", [("s","hello",{}), ("s","world",{})])
    assert n == 2

    # search
    res = await store.search("ns", "hello world", top_k=2)
    assert len(res) == 2
    assert all("score" in r for r in res)
