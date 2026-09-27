"""Build the agent in either mode. Baseline: keyword router + regex extraction, no LLM. Hybrid: learned router +
LLM extraction (rules as fallback). Everything else — FSM, tools, policy, templates — is shared, so a baseline vs
hybrid comparison isolates the learned and LLM components."""
import os

from src.agent.llm import DEFAULT_MODEL, OpenRouterLLM
from src.agent.nlu import LLMExtractor, RuleExtractor
from src.agent.orchestrator import Agent
from src.router.keyword import KeywordRouter


def build_agent(conn, mode: str = "hybrid", *, router=None, llm=None, faults: dict | None = None,
                llm_budget_usd: float | None = None) -> Agent:
    if mode == "baseline":
        return Agent(conn, router or KeywordRouter(), RuleExtractor(), mode="baseline", faults=faults)
    if mode != "hybrid":
        raise ValueError(f"unknown mode {mode!r}")
    if router is None:
        from src.router.classifier import load_router
        router = load_router()
    llm = llm or OpenRouterLLM(os.environ.get("AGENT_LLM_MODEL", DEFAULT_MODEL))
    if llm_budget_usd is not None:
        from src.agent.budget import BudgetedLLM
        llm = BudgetedLLM(llm, llm_budget_usd)
    return Agent(conn, router, LLMExtractor(llm), mode="hybrid", faults=faults)
