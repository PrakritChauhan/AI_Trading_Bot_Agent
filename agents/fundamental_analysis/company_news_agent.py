#Function : This agent is responsible for fetching news articles from yfinance api and then asynchronously scrape each news article and return the sentimental analysis and summary of each article as a pydantic object. 
from pdb import run
from schemas import BusinessNewsArticle, CompanyNewsAnalysis
import sys, os, json
import yfinance as yf
import asyncio, aiohttp
from dotenv import load_dotenv
from datetime import date, datetime, UTC
from dateutil.relativedelta import relativedelta
from newsapi import NewsApiClient
from collections import Counter
from pathlib import Path
from newspaper import Article
from groq import Groq

# Calculates the path to the project root (one level up from /agents/)
root_path = str(Path(__file__).resolve().parents[2])
if root_path not in sys.path:
    sys.path.append(root_path)

from utils.logger import setup_logger
from utils.helper_functions import retrieve_prompt

load_dotenv(override=True)

class CompanyNewsAgent: 
    
    def __init__(self, model):
        self.model = model
        self.logger = setup_logger(self.__class__.__name__, "fundamental.log")
        self.llm = Groq(api_key=os.environ["GROQ_API_KEY"])

    def _find_prev_one_month_date(self) -> str:
        today = date.today()
        one_month_ago = today - relativedelta(months=1) + relativedelta(days=1)
        iso_date = one_month_ago.strftime('%Y-%m-%d')
        return iso_date

    def _find_monthly_news_articles(self, company_name: str) -> list:
        news_api_key = os.environ["NEWS_API_KEY"]
        newsapi = NewsApiClient(api_key=news_api_key)
        from_date = self._find_prev_one_month_date()
        monthly_news_articles = newsapi.get_everything(
            q=company_name,
            from_param=from_date,
            language='en',
            sort_by="relevancy",
            page_size=2
        )
        return monthly_news_articles["articles"]

    def _find_recent_business_news_articles(self, company_name: str) -> list:
        news_api_key = os.environ["NEWS_API_KEY"]
        newsapi = NewsApiClient(api_key=news_api_key)
        business_news_articles = newsapi.get_top_headlines(
            q=company_name,
            category="business",
            language='en',
            page_size=2
        )
        if not business_news_articles["articles"]:
            business_news_articles = newsapi.get_everything(
            q=company_name,
            sort_by="relevancy",
            from_param=date.today().strftime('%Y-%m-%d'),
            language='en',
            page_size=2
        )
        #Using yfinance
        # ticker_object = yf.Ticker(ticker)
        # business_news_articles = ticker_object.get_news(count=2, tab="news")

        # print(business_news_articles["articles"])
        return business_news_articles["articles"]

    def _scrape_article_blocking(url: str) -> str:
        article = Article(url)
        article.download()
        article.parse()
        return article.text

    async def _scrape_article_async(
        self,
        url: str,
        meta: dict,
        run_id: str,
        timeout: int = 9
    ) -> dict | None:
        try:
            self.logger.info(f"[{run_id}] Scraping article: {url}")

            text = await asyncio.wait_for(
                asyncio.to_thread(self.scrape_article_blocking, url),
                timeout=timeout
            )

            self.logger.info(f"[{run_id}] Scraped article successfully")

            return {
                "url": url,
                "text": text,
                **meta
            }

        except asyncio.TimeoutError:
            self.logger.warning(f"[{run_id}] Scrape timeout: {url}")
            return None

        except Exception as e:
            self.logger.error(f"[{run_id}] Scrape failed {url}: {e}")
            return None
    
    async def _analyze_article_async(
        self,
        article: dict,
        run_id: str,
        semaphore: asyncio.Semaphore
    ):
        async with semaphore:
            try:
                self.logger.info(f"[{run_id}] LLM analyzing article: {article['url']}")
                # LLM summary and sentimental analysis for the article
                user_prompt = f"""
                    Analyze the following financial news article.

                    Metadata:
                    - timeframe: {article["timeframe"]}
                    - publisher: {article["publisher"]}
                    - published_at: {article["published_at"]}

                    Article text:
                    --------------------
                    {article["text"]}
                    --------------------
                """
                messages = [
                    {
                        "role": "system",
                        "content": retrieve_prompt("company_news_agent")
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ]

                llm_response = await asyncio.to_thread(
                self.llm.chat.completions.create,
                model=self.model,
                messages=messages,
            )
                parsed_response = json.loads(llm_response.choices[0].message.content)
                self.logger.info(f"[{run_id}] LLM analysis completed")
                self.logger.info(
                    f"[{run_id}] Article decision | "
                    f"title='{article['title']}' "
                    f"timeframe={article['timeframe']} "
                    f"sentiment={parsed_response['sentiment_label']} "
                    f"direction={parsed_response['directional_bias']} "
                    f"action={parsed_response['action_signal']} "
                    f"confidence={parsed_response['confidence']:.2f}"
                )

                business_news_analysis = BusinessNewsArticle(
                    title=article["title"],
                    publisher=article["publisher"],
                    url=article["url"],
                    published_at=article["published_at"],
                    timeframe=article["timeframe"],
                    relevance="company",
                    **parsed_response
                )
                return business_news_analysis

            except Exception as e:
                self.logger.error(
                f"[{run_id}] LLM analysis failed for {article['url']}: {e}"
            )
                return None
            

    def _aggregate_company_news(
        self,
        ticker: str,
        run_id: str,
        articles: list[BusinessNewsArticle]
    ) -> CompanyNewsAnalysis:

        # ---------- Counts ----------
        daily_articles = [a for a in articles if a.timeframe == "daily"]
        monthly_articles = [a for a in articles if a.timeframe == "monthly"]

        # ---------- Sentiment ----------

        sentiment_scores = [
            a.sentiment_score for a in articles if a.sentiment_score is not None
        ]

        avg_sentiment = (
            sum(sentiment_scores) / len(sentiment_scores)
            if sentiment_scores else None
        )

        # ---------- Distributions ----------
        sentiment_distribution = dict(
            Counter(a.sentiment_label for a in articles)
        )

        action_distribution = dict(
            Counter(a.action_signal for a in articles)
        )

        horizon_distribution = dict(
            Counter(a.impact_horizon for a in articles)
        )

        # ---------- Confidence ----------
        confidences = [a.confidence for a in articles]

        avg_confidence = (
            sum(confidences) / len(confidences)
            if confidences else 0.0
        )

        # ---------- Mechanical Veto Logic ----------
        veto_articles = [
            a.title
            for a in articles
            if (
                a.sentiment_label == "very_bearish"
                and a.impact_horizon in ("medium_term", "long_term")
            )
        ]

        veto_flag = len(veto_articles) > 0
            
        company_news_analysis = CompanyNewsAnalysis(
            ticker=ticker,
            business_news_summaries=articles,
            total_articles=len(articles),
            daily_articles=len(daily_articles),
            monthly_articles=len(monthly_articles),
            avg_sentiment_score=avg_sentiment,
            sentiment_distribution=sentiment_distribution,
            action_distribution=action_distribution,
            horizon_distribution=horizon_distribution,
            avg_confidence=avg_confidence,
            veto_flag=veto_flag,
            veto_articles=veto_articles,
            data_source="newsapi",
            generated_at= datetime.now(UTC)
        )

        self.logger.info(
            f"[{run_id}] Aggregated news | "
            f"articles={len(articles)} "
            f"avg_sentiment={avg_sentiment} "
            f"avg_confidence={avg_confidence:.2f} "
            f"veto={veto_flag}"
         )
        return company_news_analysis

    
    async def run_company_news_agent(self, state: dict) -> dict:
        run_id = state["run_id"]
        timestamp = state["timestamp"]
        ticker = state["ticker"]
        company_name = state["company_name"]


        self.logger.info(f"[{run_id}] CompanyNewsAgent started for {ticker}")

        # 1. Fetch metadata
        monthly_articles = self._find_monthly_news_articles(company_name)[:2]
        daily_articles = self._find_recent_business_news_articles(company_name)

        self.logger.info(
            f"[{run_id}] Retrieved {len(daily_articles)} daily and "
            f"{len(monthly_articles)} monthly articles"
        )

        articles_meta = []

        for article in daily_articles:
            articles_meta.append({
            "url": article.get("url"),
            "title": article.get("title"),
            "publisher": article.get("source", {}).get("name"),
            "published_at": article.get("publishedAt"),
            "timeframe": "daily",
        })

        for article in monthly_articles:
            articles_meta.append({
            "url": article.get("url"),
            "title": article.get("title"),
            "publisher": article.get("source", {}).get("name"),
            "published_at": article.get("publishedAt"),
            "timeframe": "monthly",
        })

        # 2. Scrape articles (async)
        scrape_tasks = [
            self._scrape_article_async(
                item["url"],
                meta=item,
                run_id=run_id
            )
            for item in articles_meta if item["url"]
        ]

        scraped_articles = await asyncio.gather(*scrape_tasks)
        scraped_articles = [a for a in scraped_articles if a]


        self.logger.info(f"[{run_id}] Scraped {len(scraped_articles)} articles")

        semaphore = asyncio.Semaphore(5)

        analysis_tasks = [
            self._analyze_article_async(article, run_id, semaphore)
            for article in scraped_articles
        ]

        analyzed = await asyncio.gather(*analysis_tasks)
        analyzed = [a for a in analyzed if a]

        self.logger.info(f"[{run_id}] Analyzed {len(analyzed)} articles")

        company_news_analysis = self._aggregate_company_news(
            ticker=ticker,
            articles=analyzed, 
            run_id=run_id
        )
        self.logger.info(f"[{run_id}] CompanyNewsAgent completed")

        return {
            "company_news_analysis": company_news_analysis
        }

if __name__ == "__main__":
    cnagent = CompanyNewsAgent(model="meta-llama/llama-guard-4-12b")