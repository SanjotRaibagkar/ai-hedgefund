#!/usr/bin/env python3
"""
Data Models for FNO RAG System
Defines the core data structures used throughout the system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, date
from enum import Enum


class HorizonType(Enum):
    """Trading horizons for probability prediction."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class DirectionType(Enum):
    """Price movement directions."""
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


@dataclass
class FNOData:
    """FNO market data structure."""
    symbol: str
    date: date
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    open_interest: Optional[int] = None
    put_call_ratio: Optional[float] = None
    implied_volatility: Optional[float] = None
    
    # Technical indicators
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    atr: Optional[float] = None
    
    # Returns
    daily_return: Optional[float] = None
    weekly_return: Optional[float] = None
    monthly_return: Optional[float] = None
    
    # Labels
    daily_label: Optional[int] = None  # +1, -1, 0
    weekly_label: Optional[int] = None
    monthly_label: Optional[int] = None


@dataclass
class ProbabilityResult:
    """Probability prediction result."""
    symbol: str
    horizon: HorizonType
    up_probability: float
    down_probability: float
    neutral_probability: float
    confidence_score: float
    ml_probability: Optional[Dict[str, float]] = None
    rag_probability: Optional[Dict[str, float]] = None
    similar_cases: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PredictionRequest:
    """Request for probability prediction."""
    symbol: str
    horizon: HorizonType
    include_explanations: bool = True
    top_k_similar: int = 5


@dataclass
class MarketCondition:
    """Market condition representation for RAG."""
    symbol: str
    date: date
    condition_text: str
    features: Dict[str, float]
    outcome: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class RAGResult:
    """RAG retrieval result."""
    query_condition: MarketCondition
    similar_cases: List[MarketCondition]
    empirical_probabilities: Dict[str, float]
    confidence_score: float


@dataclass
class MLResult:
    """ML prediction result."""
    symbol: str
    horizon: HorizonType
    probabilities: Dict[str, float]
    confidence_score: float
    feature_importance: Optional[Dict[str, float]] = None


@dataclass
class HybridResult:
    """Combined ML + RAG result."""
    symbol: str
    horizon: HorizonType
    final_probabilities: Dict[str, float]
    ml_result: MLResult
    rag_result: RAGResult
    alpha_weight: float
    confidence_score: float


@dataclass
class FNOSearchQuery:
    """Search query for FNO analysis."""
    query_text: str
    symbols: Optional[List[str]] = None
    horizon: Optional[HorizonType] = None
    min_probability: Optional[float] = None
    max_results: int = 10


@dataclass
class FNOSearchResult:
    """Search result for FNO analysis."""
    query: FNOSearchQuery
    results: List[ProbabilityResult]
    total_found: int
    search_time: float
    timestamp: datetime = field(default_factory=datetime.now)

