#!/usr/bin/env python3
"""
FNO RAG System - Futures & Options RAG with ML Integration
Enhanced version with semantic embeddings and advanced vector store.
"""

from .core.enhanced_fno_engine import EnhancedFNOEngine, FNOEngine
from .models.data_models import (
    HorizonType, ProbabilityResult, PredictionRequest,
    FNOSearchQuery, FNOSearchResult, MarketCondition, RAGResult
)
from .core.probability_predictor import FNOProbabilityPredictor
from .core.data_processor import FNODataProcessor
from .ml.probability_models import FNOProbabilityModels
from .api.chat_interface import FNOChatInterface
from .utils.embedding_utils import EmbeddingUtils

# Import enhanced vector store if available
try:
    from build_enhanced_vector_store import EnhancedFNOVectorStore
    ENHANCED_VECTOR_STORE_AVAILABLE = True
except ImportError:
    ENHANCED_VECTOR_STORE_AVAILABLE = False
    from .rag.vector_store import FNOVectorStore

__version__ = "2.0.0"
__author__ = "MokshTechandInvestment"
__description__ = "Enhanced FNO RAG System with Semantic Embeddings"

__all__ = [
    'FNOEngine',
    'EnhancedFNOEngine',
    'FNOProbabilityPredictor',
    'FNODataProcessor',
    'FNOProbabilityModels',
    'FNOChatInterface',
    'EmbeddingUtils',
    'EnhancedFNOVectorStore' if ENHANCED_VECTOR_STORE_AVAILABLE else 'FNOVectorStore',
    'HorizonType',
    'ProbabilityResult',
    'PredictionRequest',
    'FNOSearchQuery',
    'FNOSearchResult',
    'MarketCondition',
    'RAGResult',
    'ENHANCED_VECTOR_STORE_AVAILABLE'
]
