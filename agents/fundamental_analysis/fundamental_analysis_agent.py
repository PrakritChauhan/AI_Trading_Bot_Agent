#Function: This agent is responsible for taking in all the aggreagated data from the other agents and determine a final decision of buy/sell/hold + bearish/neutral/bullish through analysis of data.

from schemas import FundamentalAnalysis, CompanyContext
import sys, os, json
from dotenv import load_dotenv
from datetime import date, datetime, UTC
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(override=True)

# Calculates the path to the project root (one level up from /agents/)
root_path = str(Path(__file__).resolve().parents[2])
if root_path not in sys.path:
    sys.path.append(root_path)

from utils.logger import setup_logger
from utils.helper_functions import retrieve_prompt

class FundamentalAnalysisAgent:

    def __init__(self, model):
        self.model = model
        self.logger = setup_logger(self.__class__.__name__, "fundamental.log")
        self.llm = ChatGoogleGenerativeAI(model=self.model, temperature=0.3)

    def _extract_company_context(
        self,
        context: CompanyContext
    ) -> dict:
        return {
            "identity": {
                "ticker": context.ticker,
                "company_name": context.company_name,
                "industry": context.industry,
                "sector": context.sector,
                "country": context.country,
                "exchange": context.exchange,
                "market_cap": context.market_cap,
            },
            "business": {
                "summary": context.business_summary,
                "type": context.business_type,
            },
            "financials": {
                "pe_ratio": context.pe_ratio,
                "forward_pe": context.forward_pe,
                "debt_to_equity": context.debt_to_equity,
                "profit_margin": context.profit_margin,
                "operating_margin": context.operating_margin,
                "revenue_growth": context.revenue_growth,
                "earnings_growth": context.earnings_growth,
            },
            "risk": {
                "beta": context.beta,
                "cash_on_hand": context.cash_on_hand,
                "total_debt": context.total_debt,
            },
        }
    
    def _extract_article_summaries(
        self,
        articles: list
    ) -> list[dict]:    
        summaries = []

        for a in articles:
            summaries.append({
                "title": a.title,
                "timeframe": a.timeframe,
                "relevance": getattr(a, "relevance", None),
                "macro_theme": getattr(a, "macro_theme", None),
                "summary": a.raw_summary,
                "sentiment": a.sentiment_label,
                "directional_bias": a.directional_bias,
                "action_signal": a.action_signal,
                "impact_horizon": a.impact_horizon,
                "confidence": a.confidence,
            })

        return summaries

    def _extract_news_aggregates(
        self,
        analysis
    ) -> dict:
        payload = {
            "total_articles": analysis.total_articles,
            "daily_articles": analysis.daily_articles,
            "monthly_articles": analysis.monthly_articles,
            "avg_sentiment_score": analysis.avg_sentiment_score,
            "sentiment_distribution": analysis.sentiment_distribution,
            "action_distribution": analysis.action_distribution,
            "horizon_distribution": analysis.horizon_distribution,
            "avg_confidence": analysis.avg_confidence,
            "veto_flag": analysis.veto_flag,
            "veto_articles": analysis.veto_articles,
        }

        # market news only
        if hasattr(analysis, "macro_theme_distribution"):
            payload["macro_theme_distribution"] = analysis.macro_theme_distribution

        return payload
    
    def _build_fundamental_prompt(
        self,
        company_context: dict,
        company_news: dict,
        company_articles: list,
        market_news: dict,
        market_articles: list
    ) -> str:
        return f"""
        Analyze the following structured investment data and produce a final fundamental decision.

        ====================
        COMPANY CONTEXT
        ====================
        {json.dumps(company_context, indent=2)}

        ====================
        COMPANY NEWS — AGGREGATES
        ====================
        {json.dumps(company_news, indent=2)}

        ====================
        COMPANY NEWS — ARTICLE SUMMARIES
        ====================
        {json.dumps(company_articles, indent=2)}

        ====================
        MARKET & MACRO NEWS — AGGREGATES
        ====================
        {json.dumps(market_news, indent=2)}

        ====================
        MARKET & MACRO NEWS — ARTICLE SUMMARIES
        ====================
        {json.dumps(market_articles, indent=2)}

        Return a single JSON object matching the required schema.
        """

    async def run_fundamental_analysis_agent(
        self,
        state: dict
    ) -> FundamentalAnalysis:
        """
            Final decision-making agent.
            Uses STRICT structured output for safety.
        """

        run_id = state["run_id"]

        company_context = state["company_context"]
        company_news = state["company_news_analysis"]
        market_news = state["market_news_analysis"]

        self.logger.info(f"[{run_id}] Running Fundamental Analysis LLM")
        #Extract and aggregate necessary info from the provided schemas:
        context_payload = self._extract_company_context(company_context)

        company_article_summaries = self._extract_article_summaries(
            company_news.business_news_summaries
        )

        market_article_summaries = self._extract_article_summaries(
            market_news.market_news_summaries
        )

        company_aggregates = self._extract_news_aggregates(company_news)
        market_aggregates = self._extract_news_aggregates(market_news)

        user_prompt = self._build_fundamental_prompt(
            company_context=context_payload,
            company_news=company_aggregates,
            company_articles=company_article_summaries,
            market_news=market_aggregates,
            market_articles=market_article_summaries,
        )

        messages = [
            {
                "role": "system",
                "content": retrieve_prompt("fundamental_analysis_agent")
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        llm = self.llm.with_structured_output(FundamentalAnalysis)

        analysis: FundamentalAnalysis = await llm.ainvoke(messages)

        
        analysis.ticker = company_context.ticker
        analysis.company_name = company_context.company_name
        analysis.generated_at = datetime.now(UTC)

        if (
            analysis.action_signal == "buy"
            and analysis.market_bias in {"bearish", "very_bearish"}
        ):
            self.logger.warning(
                f"[{run_id}] BUY with bearish bias detected — confidence={analysis.confidence}"
            )

        if analysis.confidence > 0.85:
            self.logger.info(
                f"[{run_id}] High confidence decision ({analysis.confidence:.2f})"
            )

        self.logger.info(
            f"[{run_id}] Fundamental decision | "
            f"bias={analysis.market_bias} "
            f"action={analysis.action_signal} "
            f"confidence={analysis.confidence:.2f}"
        )

        return analysis