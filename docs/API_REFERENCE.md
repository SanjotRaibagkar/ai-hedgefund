# 📚 API Reference Documentation

## Overview

This document provides a comprehensive API reference for the FNO RAG system, including all classes, methods, and data structures.

## Core Classes

### FNOEngine

The main orchestrator class for the FNO RAG system.

```python
class FNOEngine:
    """Main orchestrator for the FNO RAG system."""
    
    def __init__(self):
        """Initialize the FNO RAG system."""
    
    def predict_probability(self, symbol: str, horizon: HorizonType, 
                          use_rag: bool = False) -> Optional[ProbabilityResult]:
        """Get probability prediction for a symbol and timeframe."""
    
    def chat(self, query: str) -> str:
        """Process natural language query and return response."""
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics."""
    
    def train_models(self, training_period_months: int = 6) -> Dict[str, Any]:
        """Train ML models with historical data."""
```

#### Methods

##### `predict_probability(symbol, horizon, use_rag=False)`

Get probability prediction for a symbol and timeframe.

**Parameters:**
- `symbol` (str): Stock symbol (e.g., "NIFTY", "RELIANCE")
- `horizon` (HorizonType): Prediction timeframe (DAILY, WEEKLY, MONTHLY)
- `use_rag` (bool): Whether to use RAG enhancement (default: False)

**Returns:**
- `ProbabilityResult` or `None`: Prediction result or None if failed

**Example:**
```python
from src.fno_rag import FNOEngine, HorizonType

fno_engine = FNOEngine()
result = fno_engine.predict_probability("NIFTY", HorizonType.DAILY)
if result:
    print(f"Up: {result.up_probability:.1%}")
    print(f"Down: {result.down_probability:.1%}")
```

##### `chat(query)`

Process natural language query and return response.

**Parameters:**
- `query` (str): Natural language query

**Returns:**
- `str`: Formatted response

**Example:**
```python
response = fno_engine.chat("What's the probability of NIFTY moving up tomorrow?")
print(response)
```

##### `get_system_status()`

Get system status and statistics.

**Returns:**
- `Dict[str, Any]`: System status information

**Example:**
```python
status = fno_engine.get_system_status()
print(f"ML Models: {status['ml_models_loaded']}")
print(f"Vector Store: {status['vector_store_ready']}")
```

##### `train_models(training_period_months=6)`

Train ML models with historical data.

**Parameters:**
- `training_period_months` (int): Number of months of historical data to use

**Returns:**
- `Dict[str, Any]`: Training results

**Example:**
```python
results = fno_engine.train_models(training_period_months=3)
for horizon, metrics in results.items():
    print(f"{horizon}: Accuracy={metrics['accuracy']:.3f}")
```

## Data Models

### ProbabilityResult

Result object containing prediction probabilities.

```python
@dataclass
class ProbabilityResult:
    symbol: str
    horizon: HorizonType
    up_probability: float
    down_probability: float
    neutral_probability: float
    confidence_score: float
    timestamp: datetime
    rag_context: Optional[str] = None
```

**Attributes:**
- `symbol` (str): Stock symbol
- `horizon` (HorizonType): Prediction timeframe
- `up_probability` (float): Probability of upward movement (0.0-1.0)
- `down_probability` (float): Probability of downward movement (0.0-1.0)
- `neutral_probability` (float): Probability of sideways movement (0.0-1.0)
- `confidence_score` (float): Model confidence (0.0-1.0)
- `timestamp` (datetime): Prediction timestamp
- `rag_context` (Optional[str]): RAG context if used

**Example:**
```python
result = ProbabilityResult(
    symbol="NIFTY",
    horizon=HorizonType.DAILY,
    up_probability=0.65,
    down_probability=0.25,
    neutral_probability=0.10,
    confidence_score=0.85,
    timestamp=datetime.now()
)
```

### HorizonType

Enumeration for prediction timeframes.

```python
class HorizonType(Enum):
    DAILY = "daily"      # ±3-5% move
    WEEKLY = "weekly"    # ±5% move
    MONTHLY = "monthly"  # ±10% move
```

**Values:**
- `DAILY`: Daily predictions (±3-5% movement)
- `WEEKLY`: Weekly predictions (±5% movement)
- `MONTHLY`: Monthly predictions (±10% movement)

### MarketCondition

Historical market condition for RAG.

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

**Attributes:**
- `symbol` (str): Stock symbol
- `date` (date): Market date
- `condition` (str): Market condition ("bullish", "bearish", "neutral")
- `direction` (str): Price direction ("up", "down", "sideways")
- `daily_return` (float): Daily return percentage
- `volume_change` (float): Volume change percentage
- `oi_change` (float): Open interest change percentage
- `text` (str): Descriptive text
- `features` (Dict[str, float]): Feature dictionary

## ML Models

### FNOProbabilityModels

Machine learning models for probability prediction.

```python
class FNOProbabilityModels:
    """ML models for FNO probability prediction."""
    
    def __init__(self):
        """Initialize ML models."""
    
    def train_models(self, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Train all ML models."""
    
    def predict_probability(self, features: Union[str, Dict[str, float]], 
                          horizon: HorizonType) -> Optional[MLResult]:
        """Get probability prediction from ML models."""
    
    def save_models(self) -> bool:
        """Save trained models to disk."""
    
    def load_models(self) -> bool:
        """Load trained models from disk."""
```

#### Methods

##### `train_models(df=None)`

Train all ML models with historical data.

**Parameters:**
- `df` (Optional[pd.DataFrame]): Training data (if None, fetches from database)

**Returns:**
- `Dict[str, Any]`: Training results for each horizon

**Example:**
```python
from src.fno_rag.ml.probability_models import FNOProbabilityModels

models = FNOProbabilityModels()
results = models.train_models()
print(f"Daily model accuracy: {results['daily']['accuracy']:.3f}")
```

##### `predict_probability(features, horizon)`

Get probability prediction from ML models.

**Parameters:**
- `features` (Union[str, Dict[str, float]]): Features or symbol
- `horizon` (HorizonType): Prediction timeframe

**Returns:**
- `MLResult` or `None`: ML prediction result

**Example:**
```python
# Using feature dictionary
features = {
    'daily_return': 0.02,
    'volume_spike_ratio': 1.5,
    'rsi_14': 65.0
}
result = models.predict_probability(features, HorizonType.DAILY)

# Using symbol
result = models.predict_probability("NIFTY", HorizonType.DAILY)
```

### MLResult

ML model prediction result.

```python
@dataclass
class MLResult:
    up_probability: float
    down_probability: float
    neutral_probability: float
    confidence_score: float
    model_type: str
    feature_importance: Optional[Dict[str, float]] = None
```

**Attributes:**
- `up_probability` (float): Upward movement probability
- `down_probability` (float): Downward movement probability
- `neutral_probability` (float): Sideways movement probability
- `confidence_score` (float): Model confidence
- `model_type` (str): Type of ML model used
- `feature_importance` (Optional[Dict[str, float]]): Feature importance scores

## RAG System

### FNOVectorStore

Vector store for historical market conditions.

```python
class FNOVectorStore:
    """FAISS-based vector store for market conditions."""
    
    def __init__(self, vector_dir: str = "data/fno_vectors"):
        """Initialize vector store."""
    
    def build_vector_store(self, symbols: List[str], start_date: str, 
                          end_date: str, batch_size: int = 1000) -> Dict[str, Any]:
        """Build vector store from FNO data."""
    
    def search_similar_conditions(self, query_text: str, 
                                k: int = 5) -> List[MarketCondition]:
        """Search for similar historical conditions."""
    
    def get_empirical_probabilities(self, symbol: str, 
                                  horizon: HorizonType) -> Dict[str, float]:
        """Get empirical probabilities from historical data."""
    
    def save(self) -> bool:
        """Save vector store to disk."""
    
    def load(self) -> bool:
        """Load vector store from disk."""
```

#### Methods

##### `build_vector_store(symbols, start_date, end_date, batch_size=1000)`

Build vector store from FNO data.

**Parameters:**
- `symbols` (List[str]): List of stock symbols
- `start_date` (str): Start date (YYYY-MM-DD)
- `end_date` (str): End date (YYYY-MM-DD)
- `batch_size` (int): Batch size for processing

**Returns:**
- `Dict[str, Any]`: Build statistics

**Example:**
```python
from src.fno_rag.rag.vector_store import FNOVectorStore

vector_store = FNOVectorStore()
stats = vector_store.build_vector_store(
    symbols=['NIFTY', 'BANKNIFTY', 'RELIANCE'],
    start_date='2024-01-01',
    end_date='2024-12-31'
)
print(f"Built vector store with {stats['total_conditions']} conditions")
```

##### `search_similar_conditions(query_text, k=5)`

Search for similar historical conditions.

**Parameters:**
- `query_text` (str): Query text
- `k` (int): Number of similar conditions to return

**Returns:**
- `List[MarketCondition]`: List of similar conditions

**Example:**
```python
similar_conditions = vector_store.search_similar_conditions(
    "NIFTY bullish market with high volume",
    k=3
)
for condition in similar_conditions:
    print(f"{condition.date}: {condition.text}")
```

### EmbeddingUtils

Text embedding utilities.

```python
class EmbeddingUtils:
    """Hash-based text embedding utilities."""
    
    def __init__(self, embedding_dim: int = 128):
        """Initialize embedding utilities."""
    
    def create_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Create embeddings for multiple texts."""
    
    def create_single_embedding(self, text: str) -> np.ndarray:
        """Create embedding for single text."""
```

#### Methods

##### `create_embeddings(texts)`

Create embeddings for multiple texts.

**Parameters:**
- `texts` (List[str]): List of text strings

**Returns:**
- `List[np.ndarray]`: List of embeddings

**Example:**
```python
from src.fno_rag.utils.embedding_utils import EmbeddingUtils

embedding_utils = EmbeddingUtils()
texts = ["NIFTY bullish market", "RELIANCE bearish trend"]
embeddings = embedding_utils.create_embeddings(texts)
print(f"Created {len(embeddings)} embeddings")
```

## Data Processing

### FNODataProcessor

Data processing for FNO data.

```python
class FNODataProcessor:
    """Data processor for FNO data."""
    
    def __init__(self):
        """Initialize data processor."""
    
    def get_fno_data(self, symbols: Optional[List[str]] = None, 
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> pd.DataFrame:
        """Get FNO data from database."""
    
    def prepare_features(self, df: pd.DataFrame, 
                        horizon: HorizonType) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for ML training."""
    
    def get_latest_data(self, symbols: List[str]) -> pd.DataFrame:
        """Get latest data for symbols."""
    
    def get_all_fno_symbols(self) -> List[str]:
        """Get all available FNO symbols."""
    
    def get_available_dates(self, start_date: datetime, 
                          end_date: datetime) -> List[datetime]:
        """Get available dates in range."""
```

#### Methods

##### `get_fno_data(symbols=None, start_date=None, end_date=None)`

Get FNO data from database.

**Parameters:**
- `symbols` (Optional[List[str]]): List of symbols (if None, gets all)
- `start_date` (Optional[str]): Start date (YYYY-MM-DD)
- `end_date` (Optional[str]): End date (YYYY-MM-DD)

**Returns:**
- `pd.DataFrame`: FNO data

**Example:**
```python
from src.fno_rag.core.data_processor import FNODataProcessor

data_processor = FNODataProcessor()
df = data_processor.get_fno_data(
    symbols=['NIFTY', 'RELIANCE'],
    start_date='2024-01-01',
    end_date='2024-12-31'
)
print(f"Retrieved {len(df)} records")
```

##### `prepare_features(df, horizon)`

Prepare features and labels for ML training.

**Parameters:**
- `df` (pd.DataFrame): Input data
- `horizon` (HorizonType): Prediction timeframe

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: Features and labels

**Example:**
```python
X, y = data_processor.prepare_features(df, HorizonType.DAILY)
print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")
```

## Natural Language Interface

### ChatInterface

Natural language query processing.

```python
class ChatInterface:
    """Natural language query interface."""
    
    def __init__(self):
        """Initialize chat interface."""
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse natural language query."""
    
    def extract_symbols(self, query: str) -> List[str]:
        """Extract stock symbols from query."""
    
    def extract_timeframe(self, query: str) -> HorizonType:
        """Extract timeframe from query."""
    
    def extract_direction(self, query: str) -> str:
        """Extract direction from query."""
```

#### Methods

##### `parse_query(query)`

Parse natural language query into structured format.

**Parameters:**
- `query` (str): Natural language query

**Returns:**
- `Dict[str, Any]`: Parsed query with intent and entities

**Example:**
```python
from src.fno_rag.api.chat_interface import ChatInterface

chat_interface = ChatInterface()
parsed = chat_interface.parse_query("What's the probability of NIFTY moving up tomorrow?")
print(f"Intent: {parsed['intent']}")
print(f"Symbols: {parsed['symbols']}")
print(f"Horizon: {parsed['horizon']}")
```

## Database Management

### DuckDBManager

Database management for DuckDB.

```python
class DuckDBManager:
    """DuckDB database manager."""
    
    def __init__(self, db_path: str = "data/comprehensive_equity.duckdb"):
        """Initialize database manager."""
    
    def initialize_database(self) -> bool:
        """Initialize database tables."""
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query."""
    
    def insert_data(self, table_name: str, data: pd.DataFrame) -> bool:
        """Insert data into table."""
    
    def get_table_info(self, table_name: str) -> pd.DataFrame:
        """Get table information."""
```

#### Methods

##### `execute_query(query)`

Execute SQL query.

**Parameters:**
- `query` (str): SQL query

**Returns:**
- `pd.DataFrame`: Query results

**Example:**
```python
from src.data.database.duckdb_manager import DuckDBManager

db_manager = DuckDBManager()
df = db_manager.execute_query("SELECT * FROM options_chain_data LIMIT 10")
print(f"Retrieved {len(df)} records")
```

## Configuration

### Model Configuration

ML model parameters can be configured in `src/fno_rag/ml/probability_models.py`:

```python
MODEL_CONFIGS = {
    HorizonType.DAILY: {
        'model_type': 'xgboost',
        'params': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
    },
    HorizonType.WEEKLY: {
        'model_type': 'random_forest',
        'params': {
            'n_estimators': 200,
            'max_depth': 10,
            'random_state': 42
        }
    },
    HorizonType.MONTHLY: {
        'model_type': 'gradient_boosting',
        'params': {
            'n_estimators': 150,
            'max_depth': 8,
            'learning_rate': 0.05,
            'random_state': 42
        }
    }
}
```

### Environment Variables

```bash
# Required
export PYTHONPATH=./src

# Optional
export GROQ_API_KEY=your_groq_api_key
export NSE_API_KEY=your_nse_api_key
export LOG_LEVEL=INFO
```

## Error Handling

### Common Exceptions

```python
class FNOEngineError(Exception):
    """Base exception for FNO RAG system."""
    pass

class ModelNotTrainedError(FNOEngineError):
    """Raised when models are not trained."""
    pass

class VectorStoreNotInitializedError(FNOEngineError):
    """Raised when vector store is not initialized."""
    pass

class DataNotFoundError(FNOEngineError):
    """Raised when required data is not found."""
    pass
```

### Error Handling Example

```python
from src.fno_rag import FNOEngine, FNOEngineError

try:
    fno_engine = FNOEngine()
    result = fno_engine.predict_probability("NIFTY", HorizonType.DAILY)
    if result:
        print(f"Prediction: {result.up_probability:.1%}")
    else:
        print("No prediction available")
except FNOEngineError as e:
    print(f"FNO Engine Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Optimization

### Memory Management

```python
def optimize_memory_usage():
    """Optimize memory usage for large datasets."""
    
    # Use chunked processing
    chunk_size = 1000
    
    # Clear variables after use
    del large_dataframe
    gc.collect()
    
    # Use memory-efficient data types
    df = df.astype({
        'price': 'float32',
        'volume': 'int32',
        'date': 'datetime64[ns]'
    })
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_prediction(symbol: str, horizon: str):
    """Cache prediction results."""
    return fno_engine.predict_probability(symbol, HorizonType(horizon))
```

## Testing

### Unit Tests

```python
import pytest
from src.fno_rag import FNOEngine, HorizonType

def test_prediction():
    """Test probability prediction."""
    fno_engine = FNOEngine()
    result = fno_engine.predict_probability("NIFTY", HorizonType.DAILY)
    
    assert result is not None
    assert 0 <= result.up_probability <= 1
    assert 0 <= result.down_probability <= 1
    assert 0 <= result.confidence_score <= 1

def test_chat_interface():
    """Test chat interface."""
    fno_engine = FNOEngine()
    response = fno_engine.chat("What's the probability of NIFTY moving up tomorrow?")
    
    assert isinstance(response, str)
    assert len(response) > 0
```

### Integration Tests

```python
def test_end_to_end():
    """Test end-to-end functionality."""
    fno_engine = FNOEngine()
    
    # Test ML prediction
    ml_result = fno_engine.predict_probability("NIFTY", HorizonType.DAILY)
    assert ml_result is not None
    
    # Test RAG enhancement
    rag_result = fno_engine.predict_probability("NIFTY", HorizonType.DAILY, use_rag=True)
    assert rag_result is not None
    
    # Test chat interface
    chat_response = fno_engine.chat("What's the probability of NIFTY moving up tomorrow?")
    assert isinstance(chat_response, str)
```

## Best Practices

### Code Style

1. **Use type hints** for all function parameters and return values
2. **Add docstrings** for all classes and methods
3. **Follow PEP 8** guidelines
4. **Use meaningful variable names**
5. **Handle exceptions gracefully**

### Performance

1. **Use chunked processing** for large datasets
2. **Cache frequently used results**
3. **Optimize database queries**
4. **Use memory-efficient data types**
5. **Monitor memory usage**

### Error Handling

1. **Use specific exception types**
2. **Provide meaningful error messages**
3. **Log errors appropriately**
4. **Graceful degradation**
5. **Validate inputs**

## Future Enhancements

### Planned Features

1. **Real-time data streaming**
2. **Advanced risk management**
3. **Portfolio optimization**
4. **Backtesting framework**
5. **Web dashboard**
6. **Mobile app**
7. **API endpoints**
8. **Multi-market support**

### API Extensions

```python
# Future API methods
def get_portfolio_analysis(self, portfolio: Dict[str, float]) -> PortfolioAnalysis:
    """Analyze portfolio risk and performance."""

def get_backtest_results(self, strategy: str, start_date: str, 
                        end_date: str) -> BacktestResult:
    """Run backtesting for trading strategy."""

def get_real_time_signals(self) -> List[TradingSignal]:
    """Get real-time trading signals."""

def optimize_portfolio(self, constraints: Dict[str, Any]) -> OptimizedPortfolio:
    """Optimize portfolio allocation."""
```
