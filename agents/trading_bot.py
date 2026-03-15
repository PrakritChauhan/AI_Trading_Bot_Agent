from typing import TypedDict, Optional
from datetime import datetime, UTC
from langgraph.graph import StateGraph, END

from agents.fundamental_analysis.schemas import (
    CompanyContext,
    CompanyNewsAnalysis,
    MarketNewsAnalysis,
    FundamentalAnalysis
)
from agents.fundamental_analysis.company_context_agent import CompanyContextAgent
from agents.fundamental_analysis.company_news_agent import CompanyNewsAgent
from agents.fundamental_analysis.market_news_agent import MarketNewsAgent
from agents.fundamental_analysis.fundamental_analysis_agent import FundamentalAnalysisAgent


# ------------------ STATE ------------------

class AnalysisState(TypedDict):
    run_id: str
    timestamp: str
    ticker: str

    # Produced nodes
    company_context: Optional[CompanyContext]
    company_news_analysis: Optional[CompanyNewsAnalysis]
    market_news_analysis: Optional[MarketNewsAnalysis]
    fundamental_analysis: Optional[FundamentalAnalysis]

# ------------------ AGENT INIT ------------------

def initialize_agents(
    article_analysis_model: str,
    fundamental_analysis_model: str
):
    return {
        "company_context": CompanyContextAgent(),
        "company_news": CompanyNewsAgent(model=article_analysis_model),
        "market_news": MarketNewsAgent(model=article_analysis_model),
        "fundamental": FundamentalAnalysisAgent(model=fundamental_analysis_model),
    }
# ------------------ NODES ------------------

def company_context_node(agent: CompanyContextAgent):
    def _node(state: AnalysisState) -> dict:
        result = agent.run_company_context_agent(state)
        return result  # {"company_context": CompanyContext}
    return _node


def company_news_node(agent: CompanyNewsAgent):
    async def _node(state: AnalysisState) -> dict:
        # Expand state for this agent
        enriched_state = {
            **state,
            "company_name": state["company_context"].company_name,
        }
        result = await agent.run_company_news_agent(enriched_state)
        return result  # {"company_news_analysis": CompanyNewsAnalysis}
    return _node


def market_news_node(agent: MarketNewsAgent):
    async def _node(state: AnalysisState) -> dict:
        enriched_state = {
            **state,
            "company_name": state["company_context"].company_name,
            "sector": state["company_context"].sector,
            "country": state["company_context"].country,
        }
        result = await agent.run_market_news_agent(enriched_state)
        return result  # {"market_news_analysis": MarketNewsAnalysis}
    return _node


def fundamental_analysis_node(agent: FundamentalAnalysisAgent):
    async def _node(state: AnalysisState) -> dict:
        analysis = await agent.run_fundamental_analysis_agent(state)
        return {"fundamental_analysis": analysis}
    return _node


def risk_guard_node():
    def _node(state: AnalysisState) -> dict:
        analysis = state["fundamental_analysis"]
        company_news = state["company_news_analysis"]
        market_news = state["market_news_analysis"]

        if company_news.veto_flag or market_news.veto_flag:
            analysis.action_signal = "hold"
            analysis.market_bias = "neutral"
            analysis.confidence = min(analysis.confidence, 0.5)

        return {"fundamental_analysis": analysis}
    return _node

# ------------------ GRAPH ------------------

def build_fundamental_graph(agents: dict):
    graph = StateGraph(AnalysisState)

    graph.add_node("company_context", company_context_node(agents["company_context"]))
    graph.add_node("company_news", company_news_node(agents["company_news"]))
    graph.add_node("market_news", market_news_node(agents["market_news"]))
    graph.add_node("fundamental_analysis", fundamental_analysis_node(agents["fundamental"]))
    graph.add_node("risk_guard", risk_guard_node())

    graph.set_entry_point("company_context")

    graph.add_edge("company_context", "company_news")
    graph.add_edge("company_news", "market_news")
    graph.add_edge("market_news", "fundamental_analysis")
    graph.add_edge("fundamental_analysis", "risk_guard")
    graph.add_edge("risk_guard", END)

    return graph.compile()


# ------------------ RUNNER ------------------

async def run_fundamental_pipeline(
    run_id: str,
    ticker: str,
    article_model: str = "meta-llama/llama-guard-4-12b",
    fundamental_model: str = "gemini-1.5-pro",
) -> FundamentalAnalysis:

    agents = initialize_agents(
        article_analysis_model=article_model,
        fundamental_analysis_model=fundamental_model,
    )

    graph = build_fundamental_graph(agents)

    initial_state: AnalysisState = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "ticker": ticker,
        "company_context": None,
        "company_news_analysis": None,
        "market_news_analysis": None,
        "fundamental_analysis": None,
    }

    final_state = await graph.ainvoke(
        initial_state,
        config={"run_name": f"fundamental_run_{run_id}"}
    )

    return final_state["fundamental_analysis"]