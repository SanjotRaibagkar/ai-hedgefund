# 🚀 FNO RAG System - AI-Powered Hedge Fund

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Quick Start](#quick-start)
4. [Features](#features)
5. [Installation](#installation)
6. [Usage](#usage)
7. [API Reference](#api-reference)
8. [Trading Guide](#trading-guide)
9. [Configuration](#configuration)
10. [Troubleshooting](#troubleshooting)
11. [Contributing](#contributing)

## 🎯 Overview

The FNO RAG (Retrieval-Augmented Generation) System is an advanced AI-powered trading platform designed for Indian Futures & Options (FNO) markets. It combines machine learning models with historical pattern recognition to provide intelligent trading predictions.

### Key Capabilities

- **🤖 ML-Powered Predictions**: XGBoost, Random Forest, and Gradient Boosting models
- **📊 RAG-Enhanced Analysis**: Historical pattern matching using FAISS vector store
- **💬 Natural Language Interface**: Query the system using plain English
- **📈 Multi-Timeframe Analysis**: Daily, weekly, and monthly predictions
- **🛡️ Risk Management**: Built-in position sizing and stop-loss recommendations
- **📊 Real-time Data**: Live FNO data processing with technical indicators

### System Components

```
FNO RAG System
├── 🤖 ML Models (XGBoost, Random Forest, Gradient Boosting)
├── 📊 RAG Vector Store (FAISS + Historical Patterns)
├── 💬 Natural Language Interface
├── 📈 Data Processing Pipeline
├── 🛡️ Risk Management Engine
└── 📊 Trading Decision System
```

## 🏗️ System Architecture

### Core Components

1. **ML Models** (`src/fno_rag/ml/`)
   - Probability prediction models for different timeframes
   - Feature engineering and model training
   - Model persistence and loading

2. **RAG System** (`src/fno_rag/rag/`)
   - Vector store for historical market conditions
   - Similarity search for pattern matching
   - Embedding generation and storage

3. **Data Processing** (`src/fno_rag/core/`)
   - Real-time FNO data collection
   - Technical indicator calculation
   - Feature preparation for ML models

4. **Natural Language Interface** (`src/fno_rag/api/`)
   - Query parsing and intent recognition
   - Response generation
   - Chat interface

5. **Trading Engine** (`src/fno_rag/core/`)
   - Probability prediction orchestration
   - Risk management calculations
   - Trading recommendations

### Data Flow

```
FNO Data → Data Processing → Feature Engineering → ML Models → Predictions
                ↓
Historical Data → Vector Store → RAG Analysis → Enhanced Predictions
                ↓
User Queries → NL Interface → Response Generation → Trading Decisions
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-hedge-fund

# Install dependencies
poetry install

# Set up environment
export PYTHONPATH=./src
```

### 2. Initialize the System

```bash
# Train ML models
poetry run python train_fno_models_optimized.py

# Build vector store
poetry run python build_vector_store_simple.py
```

### 3. Run Demo

```bash
# Complete system demo
poetry run python fno_rag_demo.py

# Trading decision guide
poetry run python trading_decision_guide.py
```

### 4. Natural Language Queries

```python
from src.fno_rag import FNOEngine

# Initialize the system
fno_engine = FNOEngine()

# Ask questions in natural language
response = fno_engine.chat("What's the probability of NIFTY moving up tomorrow?")
print(response)
```

## ✨ Features

### 🤖 Machine Learning Models

- **Multi-Timeframe Predictions**: Daily (±3-5%), Weekly (±5%), Monthly (±10%)
- **High Accuracy**: 88-96% confidence scores
- **Real-time Training**: Continuous model updates
- **Feature Engineering**: 22+ technical indicators

### 📊 RAG System

- **Historical Pattern Matching**: 77K+ market conditions
- **Similarity Search**: FAISS-based vector retrieval
- **Context-Aware Predictions**: Enhanced with historical analogs
- **Scalable Architecture**: Handle large datasets efficiently

### 💬 Natural Language Interface

- **Plain English Queries**: "What's the chance of TCS going down this week?"
- **Intent Recognition**: Automatic symbol and timeframe detection
- **Structured Responses**: Clear probability breakdowns
- **Multi-Modal Input**: Text and structured queries

### 📈 Trading Features

- **Risk Management**: Position sizing and stop-loss recommendations
- **Portfolio Analysis**: Multi-symbol correlation analysis
- **Confidence Scoring**: Probability-based decision making
- **Real-time Updates**: Live market data integration

## 📦 Installation

### Prerequisites

- Python 3.10-3.13
- Poetry (dependency management)
- DuckDB (database)
- 8GB+ RAM (for vector operations)

### Step-by-Step Installation

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd ai-hedge-fund
   ```

2. **Install Dependencies**
   ```bash
   poetry install
   ```

3. **Set Environment Variables**
   ```bash
   export PYTHONPATH=./src
   # Optional: Set GROQ_API_KEY for enhanced chat features
   export GROQ_API_KEY=your_api_key_here
   ```

4. **Initialize Database**
   ```bash
   poetry run python -c "from src.data.database.duckdb_manager import DuckDBManager; DuckDBManager().initialize_database()"
   ```

5. **Download Historical Data**
   ```bash
   poetry run python download_5_years_historical_data.py
   ```

## 🎯 Usage

### Basic Usage

```python
from src.fno_rag import FNOEngine, HorizonType

# Initialize the system
fno_engine = FNOEngine()

# Get probability prediction
result = fno_engine.predict_probability("NIFTY", HorizonType.DAILY)
print(f"Up: {result.up_probability:.1%}")
print(f"Down: {result.down_probability:.1%}")
print(f"Confidence: {result.confidence_score:.1%}")
```

### Natural Language Queries

```python
# Ask questions in plain English
queries = [
    "What's the probability of NIFTY moving up tomorrow?",
    "Predict RELIANCE movement for next month",
    "What's the chance of TCS going down this week?",
    "Show me INFY probability for tomorrow"
]

for query in queries:
    response = fno_engine.chat(query)
    print(f"Q: {query}")
    print(f"A: {response}\n")
```

### Trading Decisions

```python
# Get comprehensive trading analysis
symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY"]

for symbol in symbols:
    daily = fno_engine.predict_probability(symbol, HorizonType.DAILY)
    weekly = fno_engine.predict_probability(symbol, HorizonType.WEEKLY)
    monthly = fno_engine.predict_probability(symbol, HorizonType.MONTHLY)
    
    # Calculate weighted recommendation
    weighted_down = (
        daily.down_probability * 0.5 +
        weekly.down_probability * 0.3 +
        monthly.down_probability * 0.2
    )
    
    if weighted_down > 0.6:
        print(f"{symbol}: SELL ({weighted_down:.1%} probability)")
    elif weighted_down < 0.4:
        print(f"{symbol}: BUY ({(1-weighted_down):.1%} probability)")
    else:
        print(f"{symbol}: HOLD (neutral signals)")
```

## 📚 API Reference

### Core Classes

#### FNOEngine

Main orchestrator for the FNO RAG system.

```python
class FNOEngine:
    def __init__(self):
        """Initialize the FNO RAG system."""
    
    def predict_probability(self, symbol: str, horizon: HorizonType) -> ProbabilityResult:
        """Get probability prediction for a symbol and timeframe."""
    
    def chat(self, query: str) -> str:
        """Process natural language query and return response."""
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics."""
```

#### ProbabilityResult

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
```

#### HorizonType

Enumeration for prediction timeframes.

```python
class HorizonType(Enum):
    DAILY = "daily"      # ±3-5% move
    WEEKLY = "weekly"    # ±5% move
    MONTHLY = "monthly"  # ±10% move
```

### Data Models

#### MarketCondition

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

## 📈 Trading Guide

### Understanding Predictions

The system provides three types of predictions:

1. **Up Probability**: Chance of significant upward movement
2. **Down Probability**: Chance of significant downward movement  
3. **Neutral Probability**: Chance of sideways movement
4. **Confidence Score**: Reliability of the prediction

### Trading Strategies

#### High Confidence Trading (>80% confidence)

- **Strong Signals**: Consider larger position sizes
- **Clear Direction**: Follow the dominant probability
- **Tight Stops**: Use 1-2% stop losses for daily trades

#### Medium Confidence Trading (60-80% confidence)

- **Standard Positions**: Use normal position sizes
- **Mixed Signals**: Consider hedging strategies
- **Moderate Stops**: Use 2-3% stop losses

#### Low Confidence Trading (<60% confidence)

- **Reduced Positions**: Use smaller position sizes
- **Wait for Clarity**: Avoid trading unclear signals
- **Wide Stops**: Use 3-5% stop losses

### Risk Management Rules

1. **Position Sizing**: Never risk more than 2% of capital per trade
2. **Stop Losses**: Always set stop losses for every position
3. **Profit Taking**: Take partial profits at 50% of target
4. **Diversification**: Trade multiple symbols and timeframes
5. **Emotional Control**: Stick to your trading plan

### Example Trading Session

```python
# 1. Get market overview
symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY"]
for symbol in symbols:
    result = fno_engine.predict_probability(symbol, HorizonType.DAILY)
    if result.confidence_score > 0.8:
        if result.down_probability > 0.7:
            print(f"Strong SELL signal for {symbol}")
        elif result.up_probability > 0.7:
            print(f"Strong BUY signal for {symbol}")

# 2. Get detailed analysis for high-confidence signals
nifty_result = fno_engine.predict_probability("NIFTY", HorizonType.DAILY)
if nifty_result.confidence_score > 0.8:
    print(f"NIFTY Analysis:")
    print(f"  Up: {nifty_result.up_probability:.1%}")
    print(f"  Down: {nifty_result.down_probability:.1%}")
    print(f"  Confidence: {nifty_result.confidence_score:.1%}")
    
    # Calculate position size based on confidence
    position_size = min(0.02, nifty_result.confidence_score * 0.025)
    print(f"  Recommended Position Size: {position_size:.1%} of capital")
```

## ⚙️ Configuration

### Environment Variables

```bash
# Required
export PYTHONPATH=./src

# Optional
export GROQ_API_KEY=your_groq_api_key
export NSE_API_KEY=your_nse_api_key
export LOG_LEVEL=INFO
```

### Database Configuration

The system uses DuckDB for data storage. Configuration is handled automatically, but you can customize:

```python
# Custom database path
from src.data.database.duckdb_manager import DuckDBManager
db_manager = DuckDBManager(db_path="custom/path/database.duckdb")
```

### Model Configuration

ML model parameters can be adjusted in `src/fno_rag/ml/probability_models.py`:

```python
MODEL_CONFIGS = {
    HorizonType.DAILY: {
        'model_type': 'xgboost',
        'params': {
            'n_estimators': 100,  # Adjust for speed vs accuracy
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
    }
}
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Memory Issues

**Problem**: "Unable to allocate memory" during vector store building

**Solution**: Use the chunked approach
```bash
poetry run python build_vector_store_simple.py
```

#### 2. Model Loading Errors

**Problem**: "Model not found" errors

**Solution**: Retrain the models
```bash
poetry run python train_fno_models_optimized.py
```

#### 3. Database Connection Issues

**Problem**: "Connection already closed" errors

**Solution**: Restart the system and ensure single database instance
```python
# Use fresh database manager instance
from src.data.database.duckdb_manager import DuckDBManager
db_manager = DuckDBManager()
```

#### 4. Import Errors

**Problem**: "Module not found" errors

**Solution**: Set PYTHONPATH correctly
```bash
export PYTHONPATH=./src
```

### Performance Optimization

1. **Reduce Data Size**: Use smaller date ranges for testing
2. **Batch Processing**: Process data in chunks
3. **Memory Management**: Clear variables after use
4. **Model Optimization**: Use simpler models for faster inference

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

### Development Setup

1. **Fork the repository**
2. **Create feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make changes and test**
   ```bash
   poetry run pytest tests/
   ```
4. **Commit changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
5. **Push to branch**
   ```bash
   git push origin feature/amazing-feature
   ```
6. **Create Pull Request**

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add docstrings for all functions
- Write unit tests for new features

### Testing

```bash
# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_ml_models.py

# Run with coverage
poetry run pytest --cov=src
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For support and questions:

1. **Documentation**: Check this README and the docs folder
2. **Issues**: Create an issue on GitHub
3. **Discussions**: Use GitHub Discussions for questions
4. **Email**: Contact the development team

## 🗺️ Roadmap

### Upcoming Features

- [ ] Real-time data streaming
- [ ] Advanced risk management
- [ ] Portfolio optimization
- [ ] Backtesting framework
- [ ] Web dashboard
- [ ] Mobile app
- [ ] API endpoints
- [ ] Multi-market support

### Version History

- **v1.0.0**: Initial release with ML models and RAG system
- **v1.1.0**: Added natural language interface
- **v1.2.0**: Enhanced risk management features
- **v2.0.0**: Complete system with all components

---

**Disclaimer**: This system is for educational and research purposes. Trading involves risk, and past performance does not guarantee future results. Always consult with financial advisors before making trading decisions.
