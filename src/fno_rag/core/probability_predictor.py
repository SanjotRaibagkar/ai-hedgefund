#!/usr/bin/env python3
"""
FNO Probability Predictor
Combines ML and RAG approaches for unified probability prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from loguru import logger
from datetime import datetime

from ..models.data_models import (
    HorizonType, ProbabilityResult, PredictionRequest, 
    MarketCondition, HybridResult, MLResult, RAGResult
)
from ..ml.probability_models import FNOProbabilityModels
from ..rag.vector_store import FNOVectorStore
from ..core.data_processor import FNODataProcessor
from ..utils.embedding_utils import EmbeddingUtils


class FNOProbabilityPredictor:
    """Unified probability predictor combining ML and RAG."""
    
    def __init__(self):
        """Initialize the probability predictor."""
        self.logger = logger
        
        # Initialize components
        self.data_processor = FNODataProcessor()
        self.ml_models = FNOProbabilityModels()
        self.vector_store = FNOVectorStore()
        self.embedding_utils = EmbeddingUtils()
        
        # Load models and vector store
        self._initialize_components()
        
        # Alpha weights for different horizons
        self.alpha_weights = {
            HorizonType.DAILY: 0.7,    # ML heavier for short-term
            HorizonType.WEEKLY: 0.6,   # Balanced
            HorizonType.MONTHLY: 0.5   # Historical analogs equally important
        }
    
    def _initialize_components(self):
        """Initialize ML models and vector store."""
        try:
            # Load ML models
            self.ml_models.load_models()
            self.logger.info("ML models loaded")
            
            # Load vector store
            self.vector_store.load_vector_store()
            self.logger.info("Vector store loaded")
            
        except Exception as e:
            self.logger.warning(f"Some components failed to load: {e}")
    
    def predict_probability(self, request: PredictionRequest) -> ProbabilityResult:
        """Predict probability for a specific symbol and horizon."""
        try:
            symbol = request.symbol
            horizon = request.horizon
            
            self.logger.info(f"Predicting probability for {symbol} ({horizon.value})")
            
            # Get ML prediction
            ml_result = self._get_ml_prediction(symbol, horizon)
            
            # Get RAG prediction
            rag_result = self._get_rag_prediction(symbol, horizon, request.top_k_similar)
            
            # Combine predictions
            hybrid_result = self._combine_predictions(ml_result, rag_result, horizon)
            
            # Create final result
            result = ProbabilityResult(
                symbol=symbol,
                horizon=horizon,
                up_probability=hybrid_result.final_probabilities['up'],
                down_probability=hybrid_result.final_probabilities['down'],
                neutral_probability=hybrid_result.final_probabilities['neutral'],
                confidence_score=hybrid_result.confidence_score,
                ml_probability=ml_result.probabilities if ml_result else None,
                rag_probability=rag_result.empirical_probabilities if rag_result else None,
                similar_cases=self._format_similar_cases(rag_result) if rag_result and request.include_explanations else None
            )
            
            self.logger.info(f"✅ Prediction completed for {symbol}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to predict probability for {request.symbol}: {e}")
            raise
    
    def predict_batch(self, requests: List[PredictionRequest]) -> List[ProbabilityResult]:
        """Predict probabilities for multiple symbols."""
        try:
            results = []
            
            for request in requests:
                try:
                    result = self.predict_probability(request)
                    results.append(result)
                except Exception as e:
                    self.logger.warning(f"Failed to predict for {request.symbol}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to predict batch: {e}")
            raise
    
    def search_stocks_by_probability(self, query: str, horizon: HorizonType, 
                                   min_probability: float = 0.1,
                                   max_results: int = 10) -> List[ProbabilityResult]:
        """Search for stocks based on probability criteria."""
        try:
            self.logger.info(f"Searching stocks for: {query} ({horizon.value})")
            
            # Get all available symbols
            symbols = self._get_available_symbols()
            
            # Create prediction requests
            requests = [
                PredictionRequest(symbol=symbol, horizon=horizon, include_explanations=False)
                for symbol in symbols
            ]
            
            # Get predictions
            results = self.predict_batch(requests)
            
            # Filter based on query
            filtered_results = self._filter_results_by_query(results, query, min_probability)
            
            # Sort by relevance
            sorted_results = self._sort_results_by_relevance(filtered_results, query)
            
            # Limit results
            final_results = sorted_results[:max_results]
            
            self.logger.info(f"Found {len(final_results)} stocks matching criteria")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Failed to search stocks: {e}")
            raise
    
    def _get_ml_prediction(self, symbol: str, horizon: HorizonType) -> Optional[MLResult]:
        """Get ML prediction for a symbol."""
        try:
            # Get latest data for the symbol
            df = self.data_processor.get_latest_data([symbol])
            
            if df.empty:
                self.logger.warning(f"No data available for {symbol}")
                return None
            
            # Prepare features
            X, _ = self.data_processor.prepare_features(df, horizon)
            
            if X.empty:
                self.logger.warning(f"Could not prepare features for {symbol}")
                return None
            
            # Get model and scaler
            model = self.ml_models.models.get(horizon)
            scaler = self.ml_models.scalers.get(horizon)
            
            if model is None or scaler is None:
                self.logger.warning(f"Model not trained for {horizon.value} horizon")
                return None
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Get predictions
            probabilities = model.predict_proba(X_scaled)[0]
            
            # Map to directions
            prob_dict = {
                'down': probabilities[0],  # 0
                'neutral': probabilities[1],  # 1
                'up': probabilities[2]  # 2
            }
            
            # Calculate confidence score
            confidence_score = np.max(probabilities)
            
            # Get feature importance if available
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(self.ml_models.feature_names, model.feature_importances_))
            
            result = MLResult(
                symbol=symbol,
                horizon=horizon,
                probabilities=prob_dict,
                confidence_score=confidence_score,
                feature_importance=feature_importance
            )
            
            return result
            
        except Exception as e:
            self.logger.warning(f"ML prediction failed for {symbol}: {e}")
            return None
    
    def _get_rag_prediction(self, symbol: str, horizon: HorizonType, 
                           top_k: int = 10) -> Optional[RAGResult]:
        """Get RAG prediction for a symbol."""
        try:
            # Get latest data for the symbol
            df = self.data_processor.get_latest_data([symbol])
            
            if df.empty:
                return None
            
            # Create market condition
            row = df.iloc[0]
            condition_text = self._create_condition_text(row)
            features = self._extract_features(row)
            
            market_condition = MarketCondition(
                symbol=symbol,
                date=row['date'],
                condition_text=condition_text,
                features=features,
                outcome={},
                embedding=None
            )
            
            # Search similar conditions
            rag_result = self.vector_store.search_similar_conditions(market_condition, top_k)
            return rag_result
            
        except Exception as e:
            self.logger.warning(f"RAG prediction failed for {symbol}: {e}")
            return None
    
    def _combine_predictions(self, ml_result: Optional[MLResult], 
                           rag_result: Optional[RAGResult],
                           horizon: HorizonType) -> HybridResult:
        """Combine ML and RAG predictions."""
        try:
            alpha = self.alpha_weights[horizon]
            
            # Default probabilities
            default_probs = {'up': 0.33, 'down': 0.33, 'neutral': 0.34}
            
            # Get ML probabilities
            ml_probs = ml_result.probabilities if ml_result else default_probs
            
            # Get RAG probabilities
            rag_probs = rag_result.empirical_probabilities if rag_result else default_probs
            
            # Combine with alpha weight
            final_probs = {}
            for direction in ['up', 'down', 'neutral']:
                ml_prob = ml_probs.get(direction, 0.0)
                rag_prob = rag_probs.get(direction, 0.0)
                final_probs[direction] = alpha * ml_prob + (1 - alpha) * rag_prob
            
            # Normalize probabilities
            total = sum(final_probs.values())
            if total > 0:
                final_probs = {k: v/total for k, v in final_probs.items()}
            
            # Calculate confidence score
            ml_confidence = ml_result.confidence_score if ml_result else 0.5
            rag_confidence = rag_result.confidence_score if rag_result else 0.5
            final_confidence = alpha * ml_confidence + (1 - alpha) * rag_confidence
            
            result = HybridResult(
                symbol=ml_result.symbol if ml_result else "unknown",
                horizon=horizon,
                final_probabilities=final_probs,
                ml_result=ml_result,
                rag_result=rag_result,
                alpha_weight=alpha,
                confidence_score=final_confidence
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to combine predictions: {e}")
            raise
    
    def _create_condition_text(self, row: pd.Series) -> str:
        """Create text representation of market condition."""
        try:
            text_parts = [
                f"Stock={row['symbol']}",
                f"Date={row['date']}",
                f"Close={row['close_price']:.2f}",
                f"Volume={row['volume']:,}",
                f"Daily_Return={row.get('daily_return', 0):.3f}"
            ]
            
            # Add technical indicators
            if 'rsi_14' in row and not pd.isna(row['rsi_14']):
                text_parts.append(f"RSI={row['rsi_14']:.1f}")
            
            if 'macd' in row and not pd.isna(row['macd']):
                text_parts.append(f"MACD={row['macd']:.4f}")
            
            if 'volume_spike_ratio' in row and not pd.isna(row['volume_spike_ratio']):
                text_parts.append(f"Volume_Spike={row['volume_spike_ratio']:.2f}")
            
            return ", ".join(text_parts)
            
        except Exception as e:
            self.logger.error(f"Failed to create condition text: {e}")
            return f"Stock={row['symbol']}, Date={row['date']}"
    
    def _extract_features(self, row: pd.Series) -> Dict[str, float]:
        """Extract features from data row."""
        try:
            features = {}
            
            # Basic features
            features['close_price'] = float(row['close_price'])
            features['volume'] = float(row['volume'])
            features['daily_return'] = float(row.get('daily_return', 0))
            
            # Technical indicators
            if 'rsi_14' in row and not pd.isna(row['rsi_14']):
                features['rsi'] = float(row['rsi_14'])
            
            if 'macd' in row and not pd.isna(row['macd']):
                features['macd'] = float(row['macd'])
            
            if 'volume_spike_ratio' in row and not pd.isna(row['volume_spike_ratio']):
                features['volume_spike'] = float(row['volume_spike_ratio'])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to extract features: {e}")
            return {}
    
    def _format_similar_cases(self, rag_result: RAGResult) -> List[Dict[str, Any]]:
        """Format similar cases for output."""
        try:
            cases = []
            
            for i, case in enumerate(rag_result.similar_cases[:5]):  # Top 5 cases
                case_info = {
                    'rank': i + 1,
                    'symbol': case.symbol,
                    'date': str(case.date),
                    'condition': case.condition_text,
                    'outcome': case.outcome
                }
                cases.append(case_info)
            
            return cases
            
        except Exception as e:
            self.logger.error(f"Failed to format similar cases: {e}")
            return []
    
    def _get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        try:
            # Get unique symbols from database
            query = "SELECT DISTINCT TckrSymb as symbol FROM fno_bhav_copy ORDER BY TckrSymb"
            result = self.data_processor.db_manager.connection.execute(query).fetchdf()
            return result['symbol'].tolist()
            
        except Exception as e:
            self.logger.error(f"Failed to get available symbols: {e}")
            return []
    
    def _filter_results_by_query(self, results: List[ProbabilityResult], 
                                query: str, min_probability: float) -> List[ProbabilityResult]:
        """Filter results based on query and minimum probability."""
        try:
            filtered = []
            
            for result in results:
                # Check minimum probability
                max_prob = max(result.up_probability, result.down_probability, result.neutral_probability)
                if max_prob < min_probability:
                    continue
                
                # Check if query matches symbol or conditions
                if self._matches_query(result, query):
                    filtered.append(result)
            
            return filtered
            
        except Exception as e:
            self.logger.error(f"Failed to filter results: {e}")
            return results
    
    def _matches_query(self, result: ProbabilityResult, query: str) -> bool:
        """Check if result matches the query."""
        try:
            query_lower = query.lower()
            
            # Check symbol
            if query_lower in result.symbol.lower():
                return True
            
            # Check for specific keywords
            if 'up' in query_lower and result.up_probability > 0.3:
                return True
            
            if 'down' in query_lower and result.down_probability > 0.3:
                return True
            
            if 'high' in query_lower and max(result.up_probability, result.down_probability) > 0.4:
                return True
            
            return True  # Default to include all
            
        except Exception as e:
            self.logger.error(f"Failed to check query match: {e}")
            return True
    
    def _sort_results_by_relevance(self, results: List[ProbabilityResult], 
                                  query: str) -> List[ProbabilityResult]:
        """Sort results by relevance to query."""
        try:
            def relevance_score(result):
                score = 0
                
                # Base score on highest probability
                max_prob = max(result.up_probability, result.down_probability, result.neutral_probability)
                score += max_prob * 10
                
                # Boost for confidence
                score += result.confidence_score * 5
                
                # Boost for specific query matches
                query_lower = query.lower()
                if 'up' in query_lower:
                    score += result.up_probability * 3
                elif 'down' in query_lower:
                    score += result.down_probability * 3
                
                return score
            
            return sorted(results, key=relevance_score, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Failed to sort results: {e}")
            return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all system components."""
        try:
            status = {
                'ml_models': {},
                'vector_store': {},
                'embedding_service': {},
                'data_processor': {}
            }
            
            # Check ML models
            for horizon in HorizonType:
                perf = self.ml_models.get_model_performance(horizon)
                status['ml_models'][horizon.value] = {
                    'loaded': horizon in self.ml_models.models,
                    'performance': perf
                }
            
            # Check vector store
            stats = self.vector_store.get_stats()
            status['vector_store'] = {
                'loaded': self.vector_store.index is not None,
                'stats': stats
            }
            
            # Check embedding service
            embedding_working = self.embedding_utils.test_embedding_service()
            status['embedding_service'] = {
                'working': embedding_working,
                'cache_info': self.embedding_utils.get_cache_info()
            }
            
            # Check data processor
            try:
                test_data = self.data_processor.get_fno_data()
                status['data_processor'] = {
                    'working': True,
                    'sample_count': len(test_data) if test_data is not None else 0
                }
            except Exception as e:
                status['data_processor'] = {
                    'working': False,
                    'error': str(e)
                }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}
