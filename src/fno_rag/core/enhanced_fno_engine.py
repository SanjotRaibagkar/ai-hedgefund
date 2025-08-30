#!/usr/bin/env python3
"""
Enhanced FNO Engine - Main Orchestrator with New Vector Store
Coordinates all FNO RAG system components for unified probability prediction.
Integrates the new enhanced vector store with semantic embeddings.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from loguru import logger
from datetime import datetime, timedelta
import os
import sys

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..', '..')
sys.path.insert(0, project_root)

from ..models.data_models import (
    HorizonType, ProbabilityResult, PredictionRequest, 
    FNOSearchQuery, FNOSearchResult
)
from ..core.probability_predictor import FNOProbabilityPredictor
from ..core.data_processor import FNODataProcessor
from ..ml.probability_models import FNOProbabilityModels
from ..api.chat_interface import FNOChatInterface
from ..utils.embedding_utils import EmbeddingUtils

# Import the new enhanced vector store
try:
    from build_enhanced_vector_store import EnhancedFNOVectorStore
    ENHANCED_VECTOR_STORE_AVAILABLE = True
except ImportError:
    logger.warning("Enhanced vector store not available, falling back to original")
    ENHANCED_VECTOR_STORE_AVAILABLE = False
    from ..rag.vector_store import FNOVectorStore


class EnhancedFNOEngine:
    """Enhanced FNO RAG system orchestrator with new vector store."""
    
    def __init__(self, groq_api_key: Optional[str] = None):
        """Initialize the enhanced FNO engine."""
        self.logger = logger
        
        # Initialize components
        self.data_processor = FNODataProcessor()
        self.ml_models = FNOProbabilityModels()
        self.embedding_utils = EmbeddingUtils()
        self.predictor = FNOProbabilityPredictor()
        self.chat_interface = FNOChatInterface(groq_api_key)
        
        # Initialize vector store (enhanced or fallback)
        if ENHANCED_VECTOR_STORE_AVAILABLE:
            self.vector_store = EnhancedFNOVectorStore()
            self.logger.info("✅ Using Enhanced Vector Store")
        else:
            from ..rag.vector_store import FNOVectorStore
            self.vector_store = FNOVectorStore()
            self.logger.info("⚠️ Using Original Vector Store (enhanced not available)")
        
        # System status
        self.initialized = False
        self.last_update = None
        self.enhanced_mode = ENHANCED_VECTOR_STORE_AVAILABLE
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components."""
        try:
            self.logger.info("Initializing Enhanced FNO RAG System...")
            
            # Load ML models
            self.ml_models.load_models()
            self.logger.info("✅ ML models loaded")
            
            # Load vector store
            if self.enhanced_mode:
                # Try to load enhanced vector store
                if self.vector_store.load_vector_store():
                    self.logger.info("✅ Enhanced vector store loaded")
                else:
                    self.logger.warning("⚠️ Enhanced vector store not found, building new one...")
                    self._build_enhanced_vector_store()
            else:
                # Load original vector store
                self.vector_store.load_vector_store()
                self.logger.info("✅ Original vector store loaded")
            
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
            
            self.logger.info("🎉 Enhanced FNO RAG System initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            self.initialized = False
    
    def _build_enhanced_vector_store(self):
        """Build the enhanced vector store if not available."""
        try:
            self.logger.info("Building enhanced vector store...")
            
            # Build for last 6 months
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            
            self.vector_store.build_vector_store(start_date, end_date)
            self.logger.info("✅ Enhanced vector store built successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to build enhanced vector store: {e}")
            raise
    
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
    
    def predict_probability(self, symbol: str, horizon: HorizonType = HorizonType.DAILY) -> ProbabilityResult:
        """Get probability prediction for a symbol and horizon."""
        try:
            if not self.initialized:
                raise ValueError("System not initialized")
            
            self.logger.info(f"Predicting probability for {symbol} ({horizon.value})")
            
            # Use the predictor to get unified probability
            result = self.predictor.predict_probability(symbol, horizon)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to predict probability: {e}")
            raise
    
    def search_similar_cases(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar historical cases using the enhanced vector store."""
        try:
            if not self.initialized:
                raise ValueError("System not initialized")
            
            if self.enhanced_mode:
                # Use enhanced vector store for semantic search
                similar_cases = self.vector_store.search_similar_cases(query, top_k)
                return similar_cases
            else:
                # Fallback to original vector store
                self.logger.warning("Enhanced search not available, using basic search")
                return []
                
        except Exception as e:
            self.logger.error(f"Failed to search similar cases: {e}")
            return []
    
    def get_rag_analysis(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Get comprehensive RAG analysis including similar cases and LLM response."""
        try:
            if not self.initialized:
                raise ValueError("System not initialized")
            
            if self.enhanced_mode:
                # Use enhanced RAG system
                result = self.vector_store.query_rag_system(query, top_k)
                return result
            else:
                # Fallback response
                return {
                    'query': query,
                    'similar_cases': [],
                    'llm_response': "Enhanced RAG system not available. Please build the enhanced vector store."
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get RAG analysis: {e}")
            return {
                'query': query,
                'similar_cases': [],
                'llm_response': f"Error getting RAG analysis: {str(e)}"
            }
    
    def chat(self, query: str) -> str:
        """Process natural language queries using enhanced chat interface."""
        try:
            if not self.initialized:
                return "❌ Enhanced FNO RAG System not initialized. Please initialize the system first."
            
            # First try enhanced RAG analysis
            if self.enhanced_mode:
                try:
                    rag_result = self.get_rag_analysis(query, top_k=3)
                    
                    # Format the response
                    response_parts = []
                    
                    # Add similar cases if available
                    if rag_result.get('similar_cases'):
                        response_parts.append("📊 Similar Historical Cases:")
                        for i, case in enumerate(rag_result['similar_cases'][:3], 1):
                            response_parts.append(
                                f"  {i}. {case['symbol']} ({case['date']}): "
                                f"{case['daily_return']:+.2f}% → Next: {case['next_day_return']:+.2f}% "
                                f"(PCR: {case['pcr']:.2f})"
                            )
                        response_parts.append("")
                    
                    # Add LLM analysis
                    llm_response = rag_result.get('llm_response', '')
                    if llm_response and not llm_response.startswith("LLM service not available"):
                        response_parts.append("🤖 AI Analysis:")
                        response_parts.append(llm_response)
                    else:
                        # Fallback to basic analysis
                        response_parts.append("📈 Based on similar historical patterns:")
                        if rag_result.get('similar_cases'):
                            avg_next_return = np.mean([case['next_day_return'] for case in rag_result['similar_cases']])
                            response_parts.append(f"Average next-day return: {avg_next_return:+.2f}%")
                        else:
                            response_parts.append("No similar historical cases found.")
                    
                    return "\n".join(response_parts)
                    
                except Exception as e:
                    self.logger.warning(f"Enhanced RAG failed, falling back to basic chat: {e}")
            
            # Fallback to original chat interface
            try:
                response = self.chat_interface.process_query(query)
                if isinstance(response, dict):
                    return response.get('message', 'No response generated')
                else:
                    return str(response)
            except Exception as e:
                return f"❌ Error processing query: {str(e)}"
                
        except Exception as e:
            self.logger.error(f"Failed to process chat query: {e}")
            return f"❌ Error processing query: {str(e)}"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and component information."""
        try:
            status = {
                'initialized': self.initialized,
                'enhanced_mode': self.enhanced_mode,
                'last_update': self.last_update.isoformat() if self.last_update else None,
                'components': {
                    'ml_models': self.ml_models is not None,
                    'vector_store': self.vector_store is not None,
                    'data_processor': self.data_processor is not None,
                    'chat_interface': self.chat_interface is not None
                }
            }
            
            if self.enhanced_mode and hasattr(self.vector_store, 'metadata'):
                status['vector_store_stats'] = {
                    'total_snapshots': len(self.vector_store.metadata) if self.vector_store.metadata else 0,
                    'embedding_dimension': getattr(self.vector_store, 'embedding_dim', 'Unknown')
                }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}
    
    def rebuild_vector_store(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
        """Rebuild the vector store with new data."""
        try:
            if not self.enhanced_mode:
                return {'error': 'Enhanced vector store not available'}
            
            self.logger.info("Rebuilding enhanced vector store...")
            
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            
            self.vector_store.build_vector_store(start_date, end_date)
            
            return {
                'success': True,
                'message': 'Enhanced vector store rebuilt successfully',
                'date_range': f"{start_date} to {end_date}"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to rebuild vector store: {e}")
            return {'error': str(e)}


# Backward compatibility - create an alias for the original FNOEngine
FNOEngine = EnhancedFNOEngine
