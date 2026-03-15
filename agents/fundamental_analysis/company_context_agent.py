# Function: This agent is responsible for fetching the company context from yahoo finance based on the ticker symbol and return a pydantic object will all the company data and information.
from schemas import CompanyContext
import sys, pprint, json
import yfinance as yf
from pathlib import Path

# Calculates the path to the project root (one level up from /agents/)
root_path = str(Path(__file__).resolve().parents[2])
if root_path not in sys.path:
    sys.path.append(root_path)

from utils.logger import setup_logger

class CompanyContextAgent:
    
    def __init__(self):
        #self.logger = setup_logger(self.__class__.__name__, "fundamental.log")
        pass

    def run_company_context_agent(self, state: dict) -> dict:
        ticker = state["ticker"]
        run_id = state["run_id"]
        timestamp = state["timestamp"]

        # self.logger.info(f"[{run_id}] Fetching company context for {ticker}")
        
        ticker_object = yf.Ticker(ticker)
        company_info = ticker_object.info
        
        context = CompanyContext(
            ticker=ticker,
            company_name=company_info.get("longName"),
            industry=company_info.get("industry"),
            sector=company_info.get("sector"),
            country=company_info.get("country"),
            exchange=company_info.get("exchange"),
            market_cap=company_info.get("marketCap"),
            business_summary=company_info.get("longBusinessSummary"),
            business_type=company_info.get("quoteType", "Unknown"),  # optional heuristic later
            pe_ratio=company_info.get("trailingPE"),
            forward_pe=company_info.get("forwardPE"),
            debt_to_equity=company_info.get("debtToEquity"),
            profit_margin=company_info.get("profitMargins"),
            operating_margin=company_info.get("operatingMargins"),
            revenue_growth=company_info.get("revenueGrowth"),
            earnings_growth=company_info.get("earningsGrowth"),
            beta=company_info.get("beta"),
            cash_on_hand=company_info.get("totalCash"),
            total_debt=company_info.get("totalDebt"),

            data_source="yfinance",
            last_updated=timestamp
        )
        print(*context, sep="\n")
        # self.logger.info(f"[{run_id}] Company context completed")
        return {
            **state,
            "company_context": context.model_dump()
        }

if __name__ == "__main__":
    agent = CompanyContextAgent()
    agent.run_company_context_agent({"ticker": "AAPL", "run_id": "123", "timestamp": "2026-01-12 10:00:00"})