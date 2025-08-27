#!/usr/bin/env python3
"""
Embedding Utilities for FNO RAG
Handles text embedding generation using hash-based approach.
"""

import os
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import logging
from loguru import logger
from functools import lru_cache
# from sentence_transformers import SentenceTransformer
import hashlib


class EmbeddingUtils:
    """Utilities for generating and managing embeddings using hash-based approach."""
    
    def __init__(self, model_name: str = "simple-hash"):
        """Initialize embedding utilities."""
        self.model_name = model_name
        self.logger = logger
        self.embedding_dim = 384  # Standard dimension for compatibility
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for a text string using simple hash-based approach."""
        try:
            # Use cached embedding if available
            return self._get_embedding_cached(text)
            
        except Exception as e:
            self.logger.error(f"Failed to get embedding: {e}")
            return None
    
    @lru_cache(maxsize=10000)
    def _get_embedding_cached(self, text: str) -> Optional[List[float]]:
        """Cached version of embedding generation using hash-based approach."""
        try:
            # Create a hash of the text
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            
            # Convert hash to a list of floats
            embedding = []
            for i in range(0, len(text_hash), 2):
                if i + 1 < len(text_hash):
                    # Convert hex pair to float between 0 and 1
                    hex_val = text_hash[i:i+2]
                    float_val = int(hex_val, 16) / 255.0
                    embedding.append(float_val)
            
            # Pad or truncate to standard dimension
            while len(embedding) < self.embedding_dim:
                embedding.extend(embedding[:min(len(embedding), self.embedding_dim - len(embedding))])
            
            embedding = embedding[:self.embedding_dim]
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to get embedding from hash: {e}")
            return None
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[Optional[List[float]]]:
        """Get embeddings for multiple texts in batches."""
        try:
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = [self.get_embedding(text) for text in batch_texts]
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to get batch embeddings: {e}")
            return [None] * len(texts)
    
    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            if not embedding1 or not embedding2:
                return 0.0
            
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Normalize vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate cosine similarity: {e}")
            return 0.0
    
    def find_most_similar(self, query_embedding: List[float], 
                         candidate_embeddings: List[List[float]]) -> Tuple[int, float]:
        """Find the most similar embedding from a list of candidates."""
        try:
            if not query_embedding or not candidate_embeddings:
                return -1, 0.0
            
            similarities = []
            for candidate in candidate_embeddings:
                if candidate:
                    similarity = self.cosine_similarity(query_embedding, candidate)
                    similarities.append(similarity)
                else:
                    similarities.append(0.0)
            
            if not similarities:
                return -1, 0.0
            
            max_idx = np.argmax(similarities)
            max_similarity = similarities[max_idx]
            
            return int(max_idx), float(max_similarity)
            
        except Exception as e:
            self.logger.error(f"Failed to find most similar embedding: {e}")
            return -1, 0.0
    
    def create_market_condition_embedding(self, symbol: str, features: Dict[str, float]) -> Optional[List[float]]:
        """Create embedding for market condition from features."""
        try:
            # Create structured text representation
            text_parts = [f"Stock={symbol}"]
            
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    text_parts.append(f"{key}={value:.3f}")
                else:
                    text_parts.append(f"{key}={value}")
            
            condition_text = ", ".join(text_parts)
            return self.get_embedding(condition_text)
            
        except Exception as e:
            self.logger.error(f"Failed to create market condition embedding: {e}")
            return None
    
    def normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding vector to unit length."""
        try:
            if not embedding:
                return embedding
            
            vec = np.array(embedding)
            norm = np.linalg.norm(vec)
            
            if norm == 0:
                return embedding
            
            normalized = vec / norm
            return normalized.tolist()
            
        except Exception as e:
            self.logger.error(f"Failed to normalize embedding: {e}")
            return embedding
    
    def calculate_similarity_matrix(self, embeddings: List[List[float]]) -> np.ndarray:
        """Calculate pairwise similarity matrix for a list of embeddings."""
        try:
            if not embeddings:
                return np.array([])
            
            n = len(embeddings)
            similarity_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(i, n):
                    if embeddings[i] and embeddings[j]:
                        similarity = self.cosine_similarity(embeddings[i], embeddings[j])
                        similarity_matrix[i, j] = similarity
                        similarity_matrix[j, i] = similarity
            
            return similarity_matrix
            
        except Exception as e:
            self.logger.error(f"Failed to calculate similarity matrix: {e}")
            return np.array([])
    
    def find_top_k_similar(self, query_embedding: List[float], 
                          candidate_embeddings: List[List[float]], 
                          k: int = 5) -> List[Tuple[int, float]]:
        """Find top-k most similar embeddings."""
        try:
            if not query_embedding or not candidate_embeddings:
                return []
            
            similarities = []
            for i, candidate in enumerate(candidate_embeddings):
                if candidate:
                    similarity = self.cosine_similarity(query_embedding, candidate)
                    similarities.append((i, similarity))
                else:
                    similarities.append((i, 0.0))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:k]
            
        except Exception as e:
            self.logger.error(f"Failed to find top-k similar embeddings: {e}")
            return []
    
    def calculate_embedding_statistics(self, embeddings: List[List[float]]) -> Dict[str, float]:
        """Calculate statistics for a collection of embeddings."""
        try:
            if not embeddings:
                return {}
            
            # Filter out None embeddings
            valid_embeddings = [emb for emb in embeddings if emb is not None]
            
            if not valid_embeddings:
                return {}
            
            # Convert to numpy array
            emb_array = np.array(valid_embeddings)
            
            stats = {
                'count': len(valid_embeddings),
                'dimension': emb_array.shape[1] if len(emb_array.shape) > 1 else 0,
                'mean_norm': float(np.mean([np.linalg.norm(emb) for emb in valid_embeddings])),
                'std_norm': float(np.std([np.linalg.norm(emb) for emb in valid_embeddings])),
                'min_norm': float(np.min([np.linalg.norm(emb) for emb in valid_embeddings])),
                'max_norm': float(np.max([np.linalg.norm(emb) for emb in valid_embeddings])),
                'mean_values': float(np.mean(emb_array)),
                'std_values': float(np.std(emb_array)),
                'min_values': float(np.min(emb_array)),
                'max_values': float(np.max(emb_array))
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to calculate embedding statistics: {e}")
            return {}
    
    def validate_embedding(self, embedding: List[float]) -> bool:
        """Validate if an embedding is properly formatted."""
        try:
            if not embedding:
                return False
            
            if not isinstance(embedding, list):
                return False
            
            if len(embedding) != self.embedding_dim:
                return False
            
            if not all(isinstance(x, (int, float)) for x in embedding):
                return False
            
            # Check for NaN or infinite values
            if any(np.isnan(x) or np.isinf(x) for x in embedding):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to validate embedding: {e}")
            return False
    
    def create_embedding_fingerprint(self, embedding: List[float]) -> str:
        """Create a fingerprint/hash for an embedding for quick comparison."""
        try:
            if not embedding:
                return ""
            
            # Normalize and round to reduce noise
            normalized = self.normalize_embedding(embedding)
            rounded = [round(x, 4) for x in normalized]
            
            # Create hash
            fingerprint = hashlib.md5(str(rounded).encode()).hexdigest()
            return fingerprint
            
        except Exception as e:
            self.logger.error(f"Failed to create embedding fingerprint: {e}")
            return ""
    
    def test_embedding_service(self) -> bool:
        """Test if the embedding service is working."""
        try:
            test_text = "Test embedding for FNO market condition"
            embedding = self.get_embedding(test_text)
            
            if embedding and len(embedding) > 0:
                self.logger.info("✅ Hash-based embedding service is working")
                return True
            else:
                self.logger.error("❌ Hash-based embedding service is not working")
                return False
                
        except Exception as e:
            self.logger.error(f"Embedding service test failed: {e}")
            return False
    
    def get_embedding_dimension(self) -> Optional[int]:
        """Get the dimension of embeddings."""
        try:
            test_text = "Test"
            embedding = self.get_embedding(test_text)
            
            if embedding:
                return len(embedding)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get embedding dimension: {e}")
            return None
    
    def clear_cache(self):
        """Clear the embedding cache."""
        try:
            self._get_embedding_cached.cache_clear()
            self.logger.info("Embedding cache cleared")
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the embedding cache."""
        try:
            cache_info = self._get_embedding_cached.cache_info()
            return {
                'hits': cache_info.hits,
                'misses': cache_info.misses,
                'maxsize': cache_info.maxsize,
                'currsize': cache_info.currsize
            }
        except Exception as e:
            self.logger.error(f"Failed to get cache info: {e}")
            return {}
    
    def optimize_cache_size(self, target_hit_rate: float = 0.8):
        """Optimize cache size based on hit rate."""
        try:
            cache_info = self.get_cache_info()
            
            if cache_info.get('hits', 0) + cache_info.get('misses', 0) == 0:
                return
            
            current_hit_rate = cache_info['hits'] / (cache_info['hits'] + cache_info['misses'])
            
            if current_hit_rate < target_hit_rate:
                # Increase cache size
                new_maxsize = int(cache_info['maxsize'] * 1.5)
                self._get_embedding_cached.cache_clear()
                self._get_embedding_cached = lru_cache(maxsize=new_maxsize)(self._get_embedding_cached.__wrapped__)
                self.logger.info(f"Cache size increased to {new_maxsize}")
            elif current_hit_rate > 0.95:
                # Decrease cache size if hit rate is very high
                new_maxsize = int(cache_info['maxsize'] * 0.8)
                self._get_embedding_cached.cache_clear()
                self._get_embedding_cached = lru_cache(maxsize=new_maxsize)(self._get_embedding_cached.__wrapped__)
                self.logger.info(f"Cache size decreased to {new_maxsize}")
                
        except Exception as e:
            self.logger.error(f"Failed to optimize cache size: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        try:
            cache_info = self.get_cache_info()
            total_requests = cache_info.get('hits', 0) + cache_info.get('misses', 0)
            
            metrics = {
                'cache_hit_rate': cache_info.get('hits', 0) / total_requests if total_requests > 0 else 0,
                'cache_size': cache_info.get('currsize', 0),
                'cache_max_size': cache_info.get('maxsize', 0),
                'total_requests': total_requests,
                'embedding_dimension': self.embedding_dim,
                'model_name': self.model_name
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return {}
