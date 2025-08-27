# 📊 RAG System Documentation

## Overview

The RAG (Retrieval-Augmented Generation) System enhances ML predictions by retrieving similar historical market conditions and providing context-aware trading insights.

## Architecture

### Core Components

1. **Vector Store**: FAISS-based similarity search
2. **Embedding Engine**: Hash-based text embeddings
3. **Historical Data**: 77K+ market conditions
4. **Similarity Search**: K-nearest neighbors retrieval
5. **Context Generation**: Enhanced prediction context

### Data Flow

```
Historical FNO Data → Market Conditions → Embeddings → Vector Store
                                                           ↓
User Query → Embedding → Similarity Search → Context → Enhanced Prediction
```

## Vector Store

### FNOVectorStore Class

```python
class FNOVectorStore:
    def __init__(self, vector_dir: str = "data/fno_vectors"):
        """Initialize vector store with FAISS index."""
    
    def build_vector_store(self, symbols: List[str], 
                          start_date: str, end_date: str, 
                          batch_size: int = 1000) -> Dict[str, Any]:
        """Build vector store from FNO data."""
    
    def search_similar_conditions(self, query_text: str, 
                                k: int = 5) -> List[MarketCondition]:
        """Search for similar historical market conditions."""
    
    def get_empirical_probabilities(self, symbol: str, 
                                  horizon: HorizonType) -> Dict[str, float]:
        """Get empirical probabilities from historical data."""
```

### Market Conditions

```python
@dataclass
class MarketCondition:
    symbol: str
    date: date
    condition: str  # "bullish", "bearish", "neutral"
    direction: str  # "up", "down", "sideways"
    daily_return: float
    volume_change: float
    oi_change: float
    text: str
    features: Dict[str, float]
```

### Example Market Condition

```python
{
    'symbol': 'NIFTY',
    'date': '2024-01-15',
    'condition': 'bullish',
    'direction': 'up',
    'daily_return': 0.045,
    'volume_change': 0.12,
    'oi_change': 0.08,
    'text': 'NIFTY bullish market on 2024-01-15: up 4.5% return, volume change 12.0%, OI change 8.0%',
    'features': {
        'daily_return': 0.045,
        'volume_change': 0.12,
        'oi_change': 0.08,
        'high_low_ratio': 0.023
    }
}
```

## Embedding System

### Hash-Based Embeddings

```python
class EmbeddingUtils:
    def __init__(self, embedding_dim: int = 128):
        """Initialize hash-based embedding system."""
    
    def create_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Create embeddings from text using hash functions."""
    
    def create_single_embedding(self, text: str) -> np.ndarray:
        """Create embedding for single text."""
```

### Embedding Process

```python
def create_embedding(text: str) -> np.ndarray:
    """Create hash-based embedding for text."""
    
    # 1. Preprocess text
    text = text.lower().strip()
    
    # 2. Create hash features
    features = []
    for i in range(embedding_dim):
        # Use different hash functions for diversity
        hash_value = hash(f"{text}_{i}") % 1000
        features.append(hash_value / 1000.0)  # Normalize to [0, 1]
    
    return np.array(features, dtype=np.float32)
```

## Building Vector Store

### Simple Approach

```python
#!/usr/bin/env python3
"""
Build FNO Vector Store (Simple Approach)
"""

from src.fno_rag.rag.vector_store import FNOVectorStore

def build_vector_store():
    """Build vector store with limited data."""
    
    # Initialize vector store
    vector_store = FNOVectorStore()
    
    # Top FNO symbols
    symbols = ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS', 'INFY']
    
    # Date range (last 30 days)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Build vector store
    stats = vector_store.build_vector_store(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        batch_size=100
    )
    
    print(f"Vector store built: {stats}")
    return stats
```

### Chunked Approach (Memory Efficient)

```python
def build_vector_store_chunked():
    """Build vector store using chunked processing."""
    
    # Process symbols in chunks
    chunk_size = 10
    symbols = get_all_symbols()
    
    for i in range(0, len(symbols), chunk_size):
        chunk_symbols = symbols[i:i + chunk_size]
        
        # Process chunk
        process_symbol_chunk(chunk_symbols)
        
        # Clear memory
        gc.collect()
```

## Similarity Search

### Basic Search

```python
def search_similar_conditions(query_text: str, k: int = 5):
    """Search for similar historical conditions."""
    
    # Create query embedding
    query_embedding = embedding_utils.create_single_embedding(query_text)
    
    # Search vector store
    distances, indices = vector_store.index.search(
        query_embedding.reshape(1, -1), k
    )
    
    # Get similar conditions
    similar_conditions = []
    for idx in indices[0]:
        if idx < len(vector_store.market_conditions):
            condition = vector_store.market_conditions[idx]
            similar_conditions.append(condition)
    
    return similar_conditions
```

### Advanced Search

```python
def search_with_filters(query_text: str, symbol: str = None, 
                       date_range: Tuple[date, date] = None, k: int = 5):
    """Search with additional filters."""
    
    # Get similar conditions
    similar_conditions = search_similar_conditions(query_text, k * 2)
    
    # Apply filters
    filtered_conditions = []
    for condition in similar_conditions:
        # Symbol filter
        if symbol and condition.symbol != symbol:
            continue
        
        # Date range filter
        if date_range and not (date_range[0] <= condition.date <= date_range[1]):
            continue
        
        filtered_conditions.append(condition)
    
    return filtered_conditions[:k]
```

## Context Generation

### RAG-Enhanced Predictions

```python
def get_rag_enhanced_prediction(symbol: str, horizon: HorizonType):
    """Get prediction enhanced with RAG context."""
    
    # 1. Get ML prediction
    ml_result = get_ml_prediction(symbol, horizon)
    
    # 2. Create query text
    query_text = f"{symbol} {horizon.value} prediction"
    
    # 3. Search similar conditions
    similar_conditions = search_similar_conditions(query_text, k=5)
    
    # 4. Generate context
    context = generate_context(similar_conditions)
    
    # 5. Combine with ML prediction
    enhanced_result = combine_predictions(ml_result, context)
    
    return enhanced_result
```

### Context Generation

```python
def generate_context(similar_conditions: List[MarketCondition]) -> str:
    """Generate context from similar conditions."""
    
    if not similar_conditions:
        return "No similar historical conditions found."
    
    # Analyze patterns
    up_count = sum(1 for c in similar_conditions if c.direction == 'up')
    down_count = sum(1 for c in similar_conditions if c.direction == 'down')
    neutral_count = sum(1 for c in similar_conditions if c.direction == 'sideways')
    
    total = len(similar_conditions)
    
    # Calculate empirical probabilities
    up_prob = up_count / total
    down_prob = down_count / total
    neutral_prob = neutral_count / total
    
    # Generate context text
    context = f"Based on {total} similar historical conditions:\n"
    context += f"- Up probability: {up_prob:.1%}\n"
    context += f"- Down probability: {down_prob:.1%}\n"
    context += f"- Neutral probability: {neutral_prob:.1%}\n"
    
    # Add specific examples
    context += "\nRecent similar conditions:\n"
    for i, condition in enumerate(similar_conditions[:3]):
        context += f"{i+1}. {condition.text}\n"
    
    return context
```

## Empirical Probabilities

### Historical Analysis

```python
def get_empirical_probabilities(symbol: str, horizon: HorizonType):
    """Get empirical probabilities from historical data."""
    
    # Get historical data for symbol
    historical_data = get_historical_data(symbol)
    
    # Calculate returns for horizon
    if horizon == HorizonType.DAILY:
        returns = historical_data['daily_return']
        threshold = 0.03  # ±3%
    elif horizon == HorizonType.WEEKLY:
        returns = historical_data['weekly_return']
        threshold = 0.05  # ±5%
    else:  # Monthly
        returns = historical_data['monthly_return']
        threshold = 0.10  # ±10%
    
    # Calculate probabilities
    up_count = sum(1 for r in returns if r >= threshold)
    down_count = sum(1 for r in returns if r <= -threshold)
    neutral_count = sum(1 for r in returns if -threshold < r < threshold)
    
    total = len(returns)
    
    return {
        'up': up_count / total,
        'down': down_count / total,
        'neutral': neutral_count / total,
        'total_samples': total
    }
```

## Integration with ML Models

### Combined Predictions

```python
def combine_ml_and_rag(ml_result: ProbabilityResult, 
                      rag_context: str) -> ProbabilityResult:
    """Combine ML and RAG predictions."""
    
    # Extract empirical probabilities from RAG context
    empirical_probs = extract_empirical_probabilities(rag_context)
    
    # Weight combination (70% ML, 30% RAG)
    ml_weight = 0.7
    rag_weight = 0.3
    
    # Combine probabilities
    combined_up = (ml_result.up_probability * ml_weight + 
                  empirical_probs['up'] * rag_weight)
    combined_down = (ml_result.down_probability * ml_weight + 
                    empirical_probs['down'] * rag_weight)
    combined_neutral = (ml_result.neutral_probability * ml_weight + 
                       empirical_probs['neutral'] * rag_weight)
    
    # Normalize
    total = combined_up + combined_down + combined_neutral
    combined_up /= total
    combined_down /= total
    combined_neutral /= total
    
    # Create enhanced result
    enhanced_result = ProbabilityResult(
        symbol=ml_result.symbol,
        horizon=ml_result.horizon,
        up_probability=combined_up,
        down_probability=combined_down,
        neutral_probability=combined_neutral,
        confidence_score=ml_result.confidence_score,
        rag_context=rag_context,
        timestamp=datetime.now()
    )
    
    return enhanced_result
```

## Performance Optimization

### Memory Management

```python
def optimize_memory_usage():
    """Optimize memory usage for large datasets."""
    
    # 1. Use chunked processing
    chunk_size = 1000
    
    # 2. Clear variables after use
    del large_dataframe
    gc.collect()
    
    # 3. Use memory-efficient data types
    df = df.astype({
        'price': 'float32',
        'volume': 'int32',
        'date': 'datetime64[ns]'
    })
    
    # 4. Process in batches
    for batch in data_generator(batch_size):
        process_batch(batch)
```

### Search Optimization

```python
def optimize_search_performance():
    """Optimize similarity search performance."""
    
    # 1. Use approximate search
    vector_store.index = faiss.IndexIVFFlat(
        quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT
    )
    
    # 2. Pre-filter by symbol
    symbol_conditions = filter_by_symbol(conditions, symbol)
    
    # 3. Use smaller embedding dimensions
    embedding_dim = 64  # Instead of 128
    
    # 4. Cache frequent queries
    query_cache = {}
```

## Monitoring and Maintenance

### Vector Store Health

```python
def check_vector_store_health():
    """Check vector store health and performance."""
    
    # Check index integrity
    if vector_store.index is None:
        return "Index not initialized"
    
    # Check data consistency
    if len(vector_store.market_conditions) != vector_store.index.ntotal:
        return "Data inconsistency detected"
    
    # Check search performance
    test_query = "NIFTY daily prediction"
    start_time = time.time()
    results = search_similar_conditions(test_query, k=5)
    search_time = time.time() - start_time
    
    if search_time > 1.0:  # More than 1 second
        return f"Slow search performance: {search_time:.2f}s"
    
    return "Healthy"
```

### Data Quality Checks

```python
def validate_market_conditions(conditions: List[MarketCondition]):
    """Validate market condition data quality."""
    
    issues = []
    
    for condition in conditions:
        # Check for missing data
        if not condition.text or not condition.symbol:
            issues.append(f"Missing data in condition: {condition}")
        
        # Check for invalid returns
        if abs(condition.daily_return) > 0.5:  # 50% daily return
            issues.append(f"Unrealistic return: {condition.daily_return}")
        
        # Check for future dates
        if condition.date > datetime.now().date():
            issues.append(f"Future date: {condition.date}")
    
    return issues
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Use chunked processing and clear variables
2. **Slow Search**: Optimize index parameters and use approximate search
3. **Data Inconsistency**: Rebuild vector store with clean data
4. **Embedding Errors**: Check text preprocessing and hash functions

### Debug Mode

```python
def debug_vector_store():
    """Debug vector store issues."""
    
    # Check index status
    print(f"Index size: {vector_store.index.ntotal}")
    print(f"Conditions count: {len(vector_store.market_conditions)}")
    
    # Test search
    test_query = "NIFTY bullish market"
    results = search_similar_conditions(test_query, k=3)
    
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result.text}")
        print(f"  Distance: {result.distance}")
        print(f"  Date: {result.date}")
```

## Best Practices

1. **Data Quality**: Ensure clean, consistent historical data
2. **Memory Management**: Use chunked processing for large datasets
3. **Search Optimization**: Use appropriate index types and parameters
4. **Context Generation**: Provide meaningful, actionable insights
5. **Integration**: Balance ML and RAG predictions appropriately
6. **Monitoring**: Regular health checks and performance monitoring

## Future Enhancements

1. **Advanced Embeddings**: Use transformer-based embeddings
2. **Dynamic Updates**: Real-time vector store updates
3. **Multi-Modal Search**: Combine text and numerical features
4. **Explainable RAG**: Provide reasoning for retrieved conditions
5. **Temporal Context**: Consider time-based similarity
