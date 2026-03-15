# V1 Trading Agent Architecture

## Overview

This document describes the **V1 architecture** of a modular, agent-based stock analysis and trading decision system. The goal of V1 is **decision intelligence**, not execution speed or high-frequency trading. The system analyzes **fundamental and technical signals**, logs every meaningful step, and produces a **clear, explainable trade recommendation**.

V1 intentionally avoids:

* Live tick-by-tick streaming
* Automated trade execution
* Over-optimization or premature scaling

Instead, it focuses on **correctness, interpretability, and learnability**, so the entire system can be built and understood by a single developer.

---

## Core Design Principles

1. **Batch-Based Intelligence**

   * Data is pulled at decision time (daily / scheduled)
   * No candle-by-candle live replay
   * Reduces API cost, system complexity, and noise

2. **Agent Decomposition**

   * Each agent does one job
   * Agents communicate through structured state
   * Failure in one agent is observable and logged

3. **Explainability First**

   * Every decision must be traceable to inputs
   * Logging is a first-class citizen

4. **V1 ≠ Production Trading System**

   * V1 is an *analysis engine*
   * Execution and portfolio management come later

---

## High-Level System Architecture

```
User / Scheduler
      ↓
Orchestrator (LangGraph)
      ↓
┌─────────────────────────────┐
│ Fundamental Analysis Agent  │
└─────────────────────────────┘
      ↓
┌─────────────────────────────┐
│ Technical Analysis Agent    │
└─────────────────────────────┘
      ↓
Decision Aggregator
      ↓
Final Output + Logging
```

---

## Fundamental Analysis Agent (Primary Focus for V1)

The Fundamental Analysis Agent is itself a **multi-agent system**, inspired directly by the provided diagram.

### Purpose

To understand **what is happening with the company and its environment**, not just its price.

---

### News Categories (Critical Insight)

Fundamental analysis explicitly separates news into two scopes:

1. **Company-Specific News**

   * Earnings
   * Leadership changes
   * Product launches
   * Lawsuits / regulatory actions

2. **Industry / Economy News**

   * Sector-wide trends
   * Macroeconomic signals
   * Interest rates, inflation, supply chains

This separation prevents narrow sentiment bias and enables **context-aware reasoning**.

---

### Fundamental Analysis Internal Architecture

```
START
  ↓
Company News Agent ───┐
                      ├─→ Sentiment Aggregation
Industry/Economy Agent┘
        ↓
   Combined Context Prompt
        ↓
   Fundamental Verdict
        ↓
END
```

---

### Company News Agent

**Inputs**:

* Ticker
* Date range (daily / monthly)

**Process**:

1. Fetch 3–4 relevant company articles (API or scraping)
2. Batch scrape in parallel
3. Summarize each article
4. Perform sentiment analysis per article
5. Aggregate results into a structured summary

**Output**:

* Aggregated company sentiment
* Key bullet-point risks and positives

---

### Industry / Economy News Agent

Same structure as the Company News Agent, but focused on:

* Industry trends
* Market-wide signals
* Economic indicators

This agent answers:

> “Is the environment helping or hurting this company?”

---

### Sentiment Aggregation Node

Combines:

* Company sentiment
* Industry sentiment

Produces:

* Weighted sentiment score
* Confidence level
* Natural language explanation

---

### Fundamental Verdict Node

Uses a **single large prompt** that includes:

* Company summary
* Industry summary
* Aggregated sentiment

Produces:

* Bullish / Neutral / Bearish stance
* Reasoning trace

---

## Technical Analysis Agent (V1 Scope)

### Role

Analyze price-based signals only.

### V1 Indicators

* Moving averages
* RSI
* MACD
* Support / resistance (basic)

### Design Decision

For V1:

* **Single technical agent**
* Financial ratios (P/E, margins) are deferred or lightly integrated

Splitting into sub-agents can happen in V2.

---

## Decision Aggregator

Combines:

* Fundamental verdict
* Technical signals

Produces:

* Final decision: BUY / HOLD / SELL
* Confidence score
* Explanation

No portfolio logic in V1.

---

## Backtesting Philosophy (V1)

Backtesting is **discrete replay**, not live streaming.

### How It Works

For each historical date:

1. Load data *up to that date*
2. Run full analysis pipeline
3. Record decision
4. Compare against future outcome

No real-time candle ingestion.

---

## Logging Architecture (V1)

### Why Logging Exists

* Debug decisions
* Diagnose failures
* Create explainability

### Tooling

* Python `logging`
* Single `.log` or `.jsonl` file

### Where Logging Happens

1. Start of fundamental analysis
2. End of fundamental analysis
3. After each major agent completes
4. On any error
5. Final decision output

### What Is Logged

* run_id
* agent_name
* timestamp
* status
* sentiment scores
* final decision

No raw article text is logged.

---

## Data Storage (V1)

* No database required
* Logs stored as files
* DataFrames used in-memory

DB integration comes later.

---

## Final Output of the System

For each run:

* Ticker
* Date
* BUY / HOLD / SELL
* Confidence score
* Fundamental reasoning
* Technical reasoning

This output is **human-readable and auditable**.

---

## Explicit Non-Goals of V1

* Automated trading
* Portfolio optimization
* Latency optimization
* Multi-asset correlation

---

## Summary

V1 is an **intelligent analysis engine**, not a trading bot.

If V1 produces:

* Consistent
* Explainable
* Logged
* Reasonable decisions

Then V2 earns the right to scale.
