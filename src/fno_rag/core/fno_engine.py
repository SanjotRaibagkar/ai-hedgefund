#!/usr/bin/env python3
"""
FNO Engine - Main Orchestrator
Coordinates all FNO RAG system components for unified probability prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from loguru import logger
from datetime import datetime, timedelta
import os

from ..models.data_models import (
    HorizonType, ProbabilityResult, PredictionRequest, 
    FNOSearchQuery, FNOSearchResult
)
from ..core.probability_predictor import FNOProbabilityPredictor
from ..core.data_processor import FNODataProcessor
from ..ml.probability_models import FNOProbabilityModels
from ..rag.vector_store import FNOVectorStore
from ..api.chat_interface import FNOChatInterface
from ..utils.embedding_utils import EmbeddingUtils


class FNOEngine:
    """Main FNO RAG system orchestrator."""
    
    def __init__(self, groq_api_key: Optional[str] = None):
        """Initialize the FNO engine."""
        self.logger = logger
        
        # Initialize components
        self.data_processor = FNODataProcessor()
        self.ml_models = FNOProbabilityModels()
        self.vector_store = FNOVectorStore()
        self.embedding_utils = EmbeddingUtils()
        self.predictor = FNOProbabilityPredictor()
        self.chat_interface = FNOChatInterface(groq_api_key)
        
        # System status
        self.initialized = False
        self.last_update = None
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components."""
        try:
            self.logger.info("Initializing FNO RAG System...")
            
            # Load ML models
            self.ml_models.load_models()
            self.logger.info("✅ ML models loaded")
            
            # Load vector store
            self.vector_store.load_vector_store()
            self.logger.info("✅ Vector store loaded")
            
            # Test embedding service
            if self.embedding_utils.test_embedding_service():
                self.logger.info("✅ Embedding service working")
            else:
                self.logger.warning("⚠️ Embedding service not available")
            
            # Test data processor
            test_data = self.data_processor.get_fno_data()
            if test_data is not None and len(test_data) > 0:
                self.logger.info("✅ Data processor working")
            else:
                self.logger.warning("⚠️ Data processor issues")
            
            self.initialized = True
            self.last_update = datetime.now()
            
            self.logger.info("🎉 FNO RAG System initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            self.initialized = False
    
    def train_models(self, symbols: Optional[List[str]] = None,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Train ML models for all horizons."""
        try:
            self.logger.info("Training ML models...")
            
            results = self.ml_models.train_models(symbols, start_date, end_date, df)
            
            self.logger.info("✅ ML models trained successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to train models: {e}")
            raise
    
    def build_vector_store(self, symbols: Optional[List[str]] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> Dict[str, Any]:
        """Build vector store from FNO data."""
        try:
            self.logger.info("Building vector store...")
            
            stats = self.vector_store.build_vector_store(symbols, start_date, end_date)
            
            self.logger.info("✅ Vector store built successfully")
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to build vector store: {e}")
            raise
    
    def predict_probability(self, symbol: str, horizon: HorizonType = HorizonType.DAILY,
                          include_explanations: bool = True) -> ProbabilityResult:
        """Predict probability for a specific symbol and horizon."""
        try:
            if not self.initialized:
                raise ValueError("System not initialized")
            
            request = PredictionRequest(
                symbol=symbol,
                horizon=horizon,
                include_explanations=include_explanations,
                top_k_similar=5
            )
            
            result = self.predictor.predict_probability(request)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to predict probability for {symbol}: {e}")
            raise
    
    def predict_batch(self, symbols: List[str], horizon: HorizonType = HorizonType.DAILY) -> List[ProbabilityResult]:
        """Predict probabilities for multiple symbols."""
        try:
            if not self.initialized:
                raise ValueError("System not initialized")
            
            requests = [
                PredictionRequest(symbol=symbol, horizon=horizon, include_explanations=False)
                for symbol in symbols
            ]
            
            results = self.predictor.predict_batch(requests)
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to predict batch: {e}")
            raise
    
    def search_stocks(self, query: str, horizon: HorizonType = HorizonType.DAILY,
                     min_probability: float = 0.1, max_results: int = 10) -> List[ProbabilityResult]:
        """Search for stocks based on probability criteria."""
        try:
            if not self.initialized:
                raise ValueError("System not initialized")
            
            results = self.predictor.search_stocks_by_probability(
                query=query,
                horizon=horizon,
                min_probability=min_probability,
                max_results=max_results
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search stocks: {e}")
            raise
    
    def chat_query(self, query: str) -> Dict[str, Any]:
        """Process natural language query."""
        try:
            if not self.initialized:
                return {
                    'error': True,
                    'message': "System not initialized. Please wait for initialization to complete."
                }
            
            response = self.chat_interface.process_query(query)
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to process chat query: {e}")
            return {
                'error': True,
                'message': f"Sorry, I encountered an error: {str(e)}"
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            status = {
                'initialized': self.initialized,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'components': self.predictor.get_system_status(),
                'data_info': self._get_data_info(),
                'model_info': self._get_model_info(),
                'vector_store_info': self._get_vector_store_info()
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}
    
    def _get_data_info(self) -> Dict[str, Any]:
        """Get information about available data."""
        try:
            # Get latest data date
            query = "SELECT MAX(TRADE_DATE) as latest_date, COUNT(DISTINCT TckrSymb) as symbol_count FROM fno_bhav_copy"
            result = self.data_processor.db_manager.connection.execute(query).fetchdf()
            
            if not result.empty:
                row = result.iloc[0]
                return {
                    'latest_date': str(row['latest_date']),
                    'symbol_count': int(row['symbol_count']),
                    'data_available': True
                }
            else:
                return {'data_available': False}
                
        except Exception as e:
            self.logger.error(f"Failed to get data info: {e}")
            return {'data_available': False, 'error': str(e)}
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get information about ML models."""
        try:
            model_info = {}
            
            for horizon in HorizonType:
                perf = self.ml_models.get_model_performance(horizon)
                model_info[horizon.value] = {
                    'loaded': horizon in self.ml_models.models,
                    'performance': perf
                }
            
            return model_info
            
        except Exception as e:
            self.logger.error(f"Failed to get model info: {e}")
            return {'error': str(e)}
    
    def _get_vector_store_info(self) -> Dict[str, Any]:
        """Get information about vector store."""
        try:
            stats = self.vector_store.get_stats()
            return {
                'loaded': self.vector_store.index is not None,
                'stats': stats
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get vector store info: {e}")
            return {'error': str(e)}
    
    def retrain_models(self, horizon: Optional[HorizonType] = None) -> Dict[str, Any]:
        """Retrain ML models."""
        try:
            if horizon:
                self.logger.info(f"Retraining {horizon.value} model...")
                result = self.ml_models.retrain_model(horizon)
                self.logger.info(f"✅ {horizon.value} model retrained")
                return {horizon.value: result}
            else:
                self.logger.info("Retraining all models...")
                results = {}
                for h in HorizonType:
                    try:
                        result = self.ml_models.retrain_model(h)
                        results[h.value] = result
                        self.logger.info(f"✅ {h.value} model retrained")
                    except Exception as e:
                        self.logger.error(f"Failed to retrain {h.value} model: {e}")
                        results[h.value] = {'error': str(e)}
                
                return results
                
        except Exception as e:
            self.logger.error(f"Failed to retrain models: {e}")
            raise
    
    def update_system(self) -> Dict[str, Any]:
        """Update the entire system with latest data."""
        try:
            self.logger.info("Updating FNO RAG system...")
            
            # Get latest data date
            query = "SELECT MAX(TRADE_DATE) as latest_date FROM fno_bhav_copy"
            result = self.data_processor.db_manager.connection.execute(query).fetchdf()
            
            if result.empty:
                raise ValueError("No data available")
            
            latest_date = result.iloc[0]['latest_date']
            
            # Check if update is needed
            if self.last_update and latest_date <= self.last_update.date():
                return {
                    'message': 'System is already up to date',
                    'latest_data_date': str(latest_date),
                    'last_update': self.last_update.isoformat()
                }
            
            # Retrain models with latest data
            model_results = self.retrain_models()
            
            # Rebuild vector store with latest data
            vector_results = self.build_vector_store()
            
            self.last_update = datetime.now()
            
            return {
                'message': 'System updated successfully',
                'latest_data_date': str(latest_date),
                'model_results': model_results,
                'vector_results': vector_results,
                'last_update': self.last_update.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to update system: {e}")
            raise
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available FNO symbols."""
        try:
            query = "SELECT DISTINCT TckrSymb as symbol FROM fno_bhav_copy ORDER BY TckrSymb"
            result = self.data_processor.db_manager.connection.execute(query).fetchdf()
            return result['symbol'].tolist()
            
        except Exception as e:
            self.logger.error(f"Failed to get available symbols: {e}")
            return []
    
    def get_symbol_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical data for a specific symbol."""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            df = self.data_processor.get_fno_data([symbol], start_date, end_date)
            df = self.data_processor.calculate_technical_indicators(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get data for {symbol}: {e}")
            raise
    
    def export_results(self, results: List[ProbabilityResult], 
                      format: str = 'json') -> str:
        """Export results to various formats."""
        try:
            if format.lower() == 'json':
                data = [self._result_to_dict(result) for result in results]
                return json.dumps(data, indent=2, default=str)
            
            elif format.lower() == 'csv':
                data = [self._result_to_dict(result) for result in results]
                df = pd.DataFrame(data)
                return df.to_csv(index=False)
            
            elif format.lower() == 'excel':
                data = [self._result_to_dict(result) for result in results]
                df = pd.DataFrame(data)
                filename = f"fno_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                df.to_excel(filename, index=False)
                return filename
            
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            self.logger.error(f"Failed to export results: {e}")
            raise
    
    def _result_to_dict(self, result: ProbabilityResult) -> Dict[str, Any]:
        """Convert ProbabilityResult to dictionary."""
        try:
            return {
                'symbol': result.symbol,
                'horizon': result.horizon.value,
                'up_probability': result.up_probability,
                'down_probability': result.down_probability,
                'neutral_probability': result.neutral_probability,
                'confidence_score': result.confidence_score,
                'timestamp': result.timestamp.isoformat() if result.timestamp else None
            }
        except Exception as e:
            self.logger.error(f"Failed to convert result to dict: {e}")
            return {}
