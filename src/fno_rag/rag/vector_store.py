#!/usr/bin/env python3
"""
FNO Vector Store for RAG
Handles embedding storage and similarity search for historical market conditions.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import pickle
import os
from pathlib import Path
import logging
from loguru import logger
from datetime import datetime, date
import faiss
import json

from ..models.data_models import MarketCondition, RAGResult
from ..core.data_processor import FNODataProcessor
from ..utils.embedding_utils import EmbeddingUtils


class FNOVectorStore:
    """Vector store for FNO market conditions."""
    
    def __init__(self, vector_dir: str = "data/fno_vectors"):
        """Initialize the vector store."""
        self.vector_dir = Path(vector_dir)
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_processor = FNODataProcessor()
        self.embedding_utils = EmbeddingUtils()
        self.logger = logger
        
        # Vector store components
        self.index = None
        self.market_conditions = []
        self.condition_texts = []
        self.symbols = []
        self.dates = []
        
        # Index file paths
        self.index_file = self.vector_dir / "fno_index.faiss"
        self.metadata_file = self.vector_dir / "fno_metadata.pkl"
        
    def build_vector_store(self, symbols: Optional[List[str]] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          batch_size: int = 1000) -> Dict[str, Any]:
        """Build vector store from FNO data."""
        try:
            self.logger.info("Building FNO vector store...")
            
            # Get FNO data
            df = self.data_processor.get_fno_data(symbols, start_date, end_date)
            df = self.data_processor.calculate_technical_indicators(df)
            df = self.data_processor.create_labels(df)
            
            if df.empty:
                raise ValueError("No data available for vector store")
            
            # Process data in batches
            total_records = len(df)
            processed_records = 0
            
            for i in range(0, total_records, batch_size):
                batch_df = df.iloc[i:i+batch_size]
                self._process_batch(batch_df)
                processed_records += len(batch_df)
                self.logger.info(f"Processed {processed_records}/{total_records} records")
            
            # Build FAISS index
            self._build_faiss_index()
            
            # Save vector store
            self._save_vector_store()
            
            stats = {
                'total_conditions': len(self.market_conditions),
                'unique_symbols': len(set(self.symbols)),
                'date_range': f"{min(self.dates)} to {max(self.dates)}",
                'index_size': self.index.ntotal if self.index else 0
            }
            
            self.logger.info(f"✅ Vector store built successfully: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to build vector store: {e}")
            raise
    
    def _process_batch(self, df: pd.DataFrame):
        """Process a batch of data into market conditions."""
        try:
            for _, row in df.iterrows():
                # Create market condition text
                condition_text = self._create_condition_text(row)
                
                # Get embedding
                embedding = self.embedding_utils.get_embedding(condition_text)
                
                if embedding is not None:
                    # Create market condition
                    market_condition = MarketCondition(
                        symbol=row['symbol'],
                        date=row['date'],
                        condition_text=condition_text,
                        features=self._extract_features(row),
                        outcome=self._extract_outcome(row),
                        embedding=embedding
                    )
                    
                    self.market_conditions.append(market_condition)
                    self.condition_texts.append(condition_text)
                    self.symbols.append(row['symbol'])
                    self.dates.append(row['date'])
                    
        except Exception as e:
            self.logger.error(f"Failed to process batch: {e}")
            raise
    
    def _create_condition_text(self, row: pd.Series) -> str:
        """Create text representation of market condition."""
        try:
            text_parts = [
                f"Stock={row['symbol']}",
                f"Date={row['date']}",
                f"Close={row['close_price']:.2f}",
                f"Volume={row['volume']:,}",
                f"Daily_Return={row.get('daily_return', 0):.3f}",
                f"Weekly_Return={row.get('weekly_return', 0):.3f}",
                f"Monthly_Return={row.get('monthly_return', 0):.3f}"
            ]
            
            # Add technical indicators
            if 'rsi_14' in row and not pd.isna(row['rsi_14']):
                text_parts.append(f"RSI={row['rsi_14']:.1f}")
            
            if 'macd' in row and not pd.isna(row['macd']):
                text_parts.append(f"MACD={row['macd']:.4f}")
            
            if 'atr_14' in row and not pd.isna(row['atr_14']):
                text_parts.append(f"ATR={row['atr_14']:.3f}")
            
            if 'volume_spike_ratio' in row and not pd.isna(row['volume_spike_ratio']):
                text_parts.append(f"Volume_Spike={row['volume_spike_ratio']:.2f}")
            
            # Add Open Interest if available
            if 'open_interest' in row and not pd.isna(row['open_interest']):
                text_parts.append(f"OI={row['open_interest']:,}")
            
            if 'oi_change_pct' in row and not pd.isna(row['oi_change_pct']):
                text_parts.append(f"OI_Change={row['oi_change_pct']:.2f}%")
            
            # Add Put-Call ratio if available
            if 'put_call_ratio' in row and not pd.isna(row['put_call_ratio']):
                text_parts.append(f"Put_Call_Ratio={row['put_call_ratio']:.2f}")
            
            # Add Implied Volatility if available
            if 'implied_volatility' in row and not pd.isna(row['implied_volatility']):
                text_parts.append(f"IV={row['implied_volatility']:.2f}")
            
            return ", ".join(text_parts)
            
        except Exception as e:
            self.logger.error(f"Failed to create condition text: {e}")
            return f"Stock={row['symbol']}, Date={row['date']}"
    
    def _extract_features(self, row: pd.Series) -> Dict[str, float]:
        """Extract numerical features from row."""
        try:
            features = {}
            
            # Basic price features
            features['close_price'] = float(row['close_price'])
            features['volume'] = float(row['volume'])
            features['daily_return'] = float(row.get('daily_return', 0))
            features['weekly_return'] = float(row.get('weekly_return', 0))
            features['monthly_return'] = float(row.get('monthly_return', 0))
            
            # Technical indicators
            if 'rsi_14' in row and not pd.isna(row['rsi_14']):
                features['rsi'] = float(row['rsi_14'])
            
            if 'macd' in row and not pd.isna(row['macd']):
                features['macd'] = float(row['macd'])
            
            if 'atr_14' in row and not pd.isna(row['atr_14']):
                features['atr'] = float(row['atr_14'])
            
            if 'volume_spike_ratio' in row and not pd.isna(row['volume_spike_ratio']):
                features['volume_spike'] = float(row['volume_spike_ratio'])
            
            # Open Interest features
            if 'open_interest' in row and not pd.isna(row['open_interest']):
                features['open_interest'] = float(row['open_interest'])
            
            if 'oi_change_pct' in row and not pd.isna(row['oi_change_pct']):
                features['oi_change'] = float(row['oi_change_pct'])
            
            # Put-Call ratio
            if 'put_call_ratio' in row and not pd.isna(row['put_call_ratio']):
                features['put_call_ratio'] = float(row['put_call_ratio'])
            
            # Implied Volatility
            if 'implied_volatility' in row and not pd.isna(row['implied_volatility']):
                features['implied_volatility'] = float(row['implied_volatility'])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to extract features: {e}")
            return {}
    
    def _extract_outcome(self, row: pd.Series) -> Dict[str, Any]:
        """Extract outcome information from row."""
        try:
            outcome = {
                'daily_label': int(row.get('daily_label', 0)),
                'weekly_label': int(row.get('weekly_label', 0)),
                'monthly_label': int(row.get('monthly_label', 0)),
                'daily_return': float(row.get('daily_return', 0)),
                'weekly_return': float(row.get('weekly_return', 0)),
                'monthly_return': float(row.get('monthly_return', 0))
            }
            
            return outcome
            
        except Exception as e:
            self.logger.error(f"Failed to extract outcome: {e}")
            return {}
    
    def _build_faiss_index(self):
        """Build FAISS index from embeddings."""
        try:
            if not self.market_conditions:
                raise ValueError("No market conditions available")
            
            # Get embeddings
            embeddings = [mc.embedding for mc in self.market_conditions]
            embeddings = np.array(embeddings, dtype=np.float32)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to index
            self.index.add(embeddings)
            
            self.logger.info(f"FAISS index built with {self.index.ntotal} vectors")
            
        except Exception as e:
            self.logger.error(f"Failed to build FAISS index: {e}")
            raise
    
    def search_similar_conditions(self, query_condition: MarketCondition, 
                                top_k: int = 10) -> RAGResult:
        """Search for similar market conditions."""
        try:
            if self.index is None:
                raise ValueError("Vector store not initialized")
            
            # Get query embedding
            query_embedding = query_condition.embedding
            if query_embedding is None:
                query_embedding = self.embedding_utils.get_embedding(query_condition.condition_text)
                query_condition.embedding = query_embedding
            
            if query_embedding is None:
                raise ValueError("Could not generate embedding for query")
            
            # Prepare query vector
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)
            
            # Search
            similarities, indices = self.index.search(query_vector, top_k)
            
            # Get similar conditions
            similar_conditions = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.market_conditions):
                    similar_conditions.append(self.market_conditions[idx])
            
            # Calculate empirical probabilities
            empirical_probs = self._calculate_empirical_probabilities(similar_conditions)
            
            # Calculate confidence score
            confidence_score = np.mean(similarities[0]) if len(similarities[0]) > 0 else 0.0
            
            result = RAGResult(
                query_condition=query_condition,
                similar_cases=similar_conditions,
                empirical_probabilities=empirical_probs,
                confidence_score=confidence_score
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to search similar conditions: {e}")
            raise
    
    def _calculate_empirical_probabilities(self, similar_conditions: List[MarketCondition]) -> Dict[str, float]:
        """Calculate empirical probabilities from similar cases."""
        try:
            if not similar_conditions:
                return {'up': 0.0, 'down': 0.0, 'neutral': 1.0}
            
            daily_labels = [mc.outcome.get('daily_label', 0) for mc in similar_conditions]
            weekly_labels = [mc.outcome.get('weekly_label', 0) for mc in similar_conditions]
            monthly_labels = [mc.outcome.get('monthly_label', 0) for mc in similar_conditions]
            
            # Calculate probabilities for each horizon
            daily_probs = self._calculate_label_probabilities(daily_labels)
            weekly_probs = self._calculate_label_probabilities(weekly_labels)
            monthly_probs = self._calculate_label_probabilities(monthly_labels)
            
            # Return daily probabilities as default (can be extended for multi-horizon)
            return daily_probs
            
        except Exception as e:
            self.logger.error(f"Failed to calculate empirical probabilities: {e}")
            return {'up': 0.0, 'down': 0.0, 'neutral': 1.0}
    
    def _calculate_label_probabilities(self, labels: List[int]) -> Dict[str, float]:
        """Calculate probabilities from labels."""
        try:
            total = len(labels)
            if total == 0:
                return {'up': 0.0, 'down': 0.0, 'neutral': 1.0}
            
            up_count = sum(1 for label in labels if label == 1)
            down_count = sum(1 for label in labels if label == -1)
            neutral_count = sum(1 for label in labels if label == 0)
            
            return {
                'up': up_count / total,
                'down': down_count / total,
                'neutral': neutral_count / total
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate label probabilities: {e}")
            return {'up': 0.0, 'down': 0.0, 'neutral': 1.0}
    
    def _save_vector_store(self):
        """Save vector store to disk."""
        try:
            # Save FAISS index
            if self.index is not None:
                faiss.write_index(self.index, str(self.index_file))
            
            # Save metadata
            metadata = {
                'market_conditions': self.market_conditions,
                'condition_texts': self.condition_texts,
                'symbols': self.symbols,
                'dates': self.dates
            }
            
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            self.logger.info("Vector store saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save vector store: {e}")
            raise
    
    def load_vector_store(self):
        """Load vector store from disk."""
        try:
            # Load FAISS index
            if self.index_file.exists():
                self.index = faiss.read_index(str(self.index_file))
                self.logger.info(f"FAISS index loaded with {self.index.ntotal} vectors")
            
            # Load metadata
            if self.metadata_file.exists():
                with open(self.metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                
                self.market_conditions = metadata['market_conditions']
                self.condition_texts = metadata['condition_texts']
                self.symbols = metadata['symbols']
                self.dates = metadata['dates']
                
                self.logger.info(f"Metadata loaded: {len(self.market_conditions)} conditions")
            
        except Exception as e:
            self.logger.error(f"Failed to load vector store: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            stats = {
                'total_conditions': len(self.market_conditions),
                'unique_symbols': len(set(self.symbols)) if self.symbols else 0,
                'index_size': self.index.ntotal if self.index else 0,
                'date_range': None
            }
            
            if self.dates:
                stats['date_range'] = f"{min(self.dates)} to {max(self.dates)}"
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {}

