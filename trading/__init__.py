"""
Multi-Agent Trading System

Agents:
  - ScanAgent: Filters 300+ markets by liquidity, volume, spread, price moves
  - ResearchAgent: Parallel scraping of Twitter, Reddit, RSS + sentiment analysis
  - PredictionAgent: XGBoost + LLM probability calibration
  - RiskManager: Bet sizing, account protection, trade blocking
  - LossAnalyzer: Post-loss root cause analysis + system self-improvement
  - Orchestrator: Coordinates all agents end-to-end
"""

__version__ = "1.0.0"
