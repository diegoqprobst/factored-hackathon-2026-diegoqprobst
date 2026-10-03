from src.agent.factory import build_agent
from src.router.classifier import tfidf_router
from tests.router.test_classifier import TRAIN


def test_hybrid_agent_loads_router_from_router_path(bank, frozen, tmp_path, monkeypatch):
    tfidf_router().fit(TRAIN).save(tmp_path, {"model": "tfidf"})
    monkeypatch.setenv("ROUTER_PATH", str(tmp_path))

    class NoLLM:
        model = "none"

        def complete(self, *a, **k):
            raise RuntimeError("unused")
    agent = build_agent(bank, "hybrid", llm=NoLLM())
    assert agent.router.version == "tfidf_v1"
