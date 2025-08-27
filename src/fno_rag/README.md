# FNO RAG System - Futures & Options RAG with ML Integration

A comprehensive system for FNO (Futures & Options) probability prediction combining Machine Learning and Retrieval-Augmented Generation (RAG) approaches.

## 🎯 Overview

The FNO RAG System provides unified probability predictions for FNO stocks across three horizons:
- **Daily**: ±3-5% move probability
- **Weekly**: ±5% move probability  
- **Monthly**: ±10% move probability

Each prediction combines:
- **ML-based classifier** (statistical probability)
- **RAG-based retrieval** (historical analogs)
- **Hybrid scoring** (weighted combination)

## 🏗️ Architecture

```
src/fno_rag/
├── __init__.py                    # Main package
├── README.md                      # This file
├── core/                          # Core components
│   ├── __init__.py
│   ├── fno_engine.py             # Main orchestrator
│   ├── probability_predictor.py  # Unified predictor
│   └── data_processor.py         # Data processing
├── models/                        # Data models
│   ├── __init__.py
│   └── data_models.py            # Data structures
├── ml/                           # Machine Learning
│   ├── __init__.py
│   └── probability_models.py     # ML classifiers
├── rag/                          # RAG components
│   ├── __init__.py
│   └── vector_store.py           # Vector store & retrieval
├── utils/                        # Utilities
│   ├── __init__.py
│   └── embedding_utils.py        # Embedding generation
├── api/                          # API interfaces
│   ├── __init__.py
│   └── chat_interface.py         # Natural language interface
└── notebooks/                    # Demo & examples
    ├── fno_rag_demo.ipynb        # Jupyter notebook demo
    └── fno_rag_demo.py           # Python script demo
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
poetry install

# Set environment variables
export GROQ_API_KEY="your_groq_api_key_here"
```

### 2. Basic Usage

```python
from src.fno_rag import FNOEngine, HorizonType

# Initialize the system
fno_engine = FNOEngine()

# Predict probability for a stock
result = fno_engine.predict_probability("RELIANCE", HorizonType.DAILY)
print(f"Up: {result.up_probability:.1%}")
print(f"Down: {result.down_probability:.1%}")
print(f"Neutral: {result.neutral_probability:.1%}")

# Search for stocks
results = fno_engine.search_stocks("up tomorrow", HorizonType.DAILY)
for result in results:
    print(f"{result.symbol}: Up={result.up_probability:.1%}")

# Natural language query
response = fno_engine.chat_query("What's the probability of TCS moving up tomorrow?")
print(response['message'])
```

### 3. Training Models

```python
# Train ML models (one-time setup)
fno_engine.train_models()

# Build vector store
fno_engine.build_vector_store()

# Update system with latest data
fno_engine.update_system()
```

## 📊 Features

### 1. **Multi-Horizon Prediction**
- **Daily**: ±3-5% move probability
- **Weekly**: ±5% move probability
- **Monthly**: ±10% move probability

### 2. **Hybrid ML + RAG Approach**
- **ML Models**: XGBoost, Random Forest, Gradient Boosting
- **RAG Retrieval**: Historical analog search using FAISS
- **Hybrid Scoring**: Weighted combination (α * ML + (1-α) * RAG)

### 3. **Technical Indicators**
- RSI, MACD, Bollinger Bands
- ATR, Volume analysis
- Open Interest analysis
- Put-Call ratios

### 4. **Natural Language Interface**
- Query parsing and intent recognition
- Stock symbol extraction
- Horizon and direction detection
- GROQ API integration for general questions

### 5. **Stock Search & Filtering**
- Probability-based filtering
- Multi-criteria search
- Relevance scoring
- Batch processing

### 6. **Export & Integration**
- JSON, CSV, Excel export
- RESTful API ready
- Jupyter notebook support
- Service integration

## 🔧 Configuration

### Environment Variables

```bash
# Required for embeddings and chat
GROQ_API_KEY=your_groq_api_key

# Optional configurations
FNO_MODELS_DIR=models/fno_ml
FNO_VECTOR_DIR=data/fno_vectors
```

### Model Configuration

```python
# Alpha weights for different horizons
alpha_weights = {
    HorizonType.DAILY: 0.7,    # ML heavier for short-term
    HorizonType.WEEKLY: 0.6,   # Balanced
    HorizonType.MONTHLY: 0.5   # Historical analogs equally important
}
```

## 📈 Usage Examples

### Example 1: Single Stock Analysis

```python
# Get probability for RELIANCE tomorrow
result = fno_engine.predict_probability("RELIANCE", HorizonType.DAILY)

print(f"RELIANCE Daily Prediction:")
print(f"  Up: {result.up_probability:.1%}")
print(f"  Down: {result.down_probability:.1%}")
print(f"  Neutral: {result.neutral_probability:.1%}")
print(f"  Confidence: {result.confidence_score:.1%}")

# Get similar historical cases
if result.similar_cases:
    print(f"\nSimilar cases: {len(result.similar_cases)}")
    for case in result.similar_cases[:3]:
        print(f"  {case['symbol']} ({case['date']})")
```

### Example 2: Stock Search

```python
# Find stocks with high probability of moving up tomorrow
results = fno_engine.search_stocks(
    query="up tomorrow",
    horizon=HorizonType.DAILY,
    min_probability=0.3,
    max_results=10
)

print(f"Found {len(results)} stocks:")
for i, result in enumerate(results, 1):
    print(f"{i}. {result.symbol}: Up={result.up_probability:.1%}")
```

### Example 3: Natural Language Queries

```python
# Natural language queries
queries = [
    "What's the probability of RELIANCE moving up tomorrow?",
    "Find FNO stocks that can move 3% tomorrow",
    "Which stocks have high probability of moving down this week?",
    "What's the system status?"
]

for query in queries:
    response = fno_engine.chat_query(query)
    print(f"Q: {query}")
    print(f"A: {response['message']}\n")
```

### Example 4: Batch Processing

```python
# Process multiple stocks
symbols = ["RELIANCE", "TCS", "INFY", "HDFC", "ICICIBANK"]
results = fno_engine.predict_batch(symbols, HorizonType.WEEKLY)

# Export results
json_data = fno_engine.export_results(results, 'json')
csv_data = fno_engine.export_results(results, 'csv')
```

## 🧪 Demo & Testing

### Run Demo Script

```bash
cd src/fno_rag/notebooks
python fno_rag_demo.py
```

### Jupyter Notebook

```bash
cd src/fno_rag/notebooks
jupyter notebook fno_rag_demo.ipynb
```

## 🔍 System Status

```python
# Check system health
status = fno_engine.get_system_status()
print(json.dumps(status, indent=2))

# Check specific components
print(f"ML Models: {status['components']['ml_models']}")
print(f"Vector Store: {status['components']['vector_store']}")
print(f"Embedding Service: {status['components']['embedding_service']}")
```

## 📋 API Reference

### Core Classes

#### `FNOEngine`
Main orchestrator for the FNO RAG system.

```python
fno_engine = FNOEngine(groq_api_key="optional")

# Methods
fno_engine.predict_probability(symbol, horizon)
fno_engine.predict_batch(symbols, horizon)
fno_engine.search_stocks(query, horizon, min_probability, max_results)
fno_engine.chat_query(query)
fno_engine.train_models()
fno_engine.build_vector_store()
fno_engine.update_system()
fno_engine.get_system_status()
```

#### `ProbabilityResult`
Result object containing prediction probabilities.

```python
result = ProbabilityResult(
    symbol="RELIANCE",
    horizon=HorizonType.DAILY,
    up_probability=0.25,
    down_probability=0.15,
    neutral_probability=0.60,
    confidence_score=0.75
)
```

## 🛠️ Development

### Adding New Features

1. **New Technical Indicators**: Add to `data_processor.py`
2. **New ML Models**: Extend `probability_models.py`
3. **New RAG Features**: Enhance `vector_store.py`
4. **New API Endpoints**: Extend `chat_interface.py`

### Testing

```bash
# Run tests
poetry run pytest tests/

# Run specific test
poetry run pytest tests/test_fno_engine.py
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📚 Dependencies

### Core Dependencies
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning
- `xgboost`: Gradient boosting
- `faiss`: Vector similarity search
- `requests`: HTTP client
- `loguru`: Logging

### Optional Dependencies
- `jupyter`: Notebook support
- `openpyxl`: Excel export
- `plotly`: Visualization

## 🤝 Integration

### As a Service

```python
# Initialize as a service
fno_service = FNOEngine()

# REST API endpoints
@app.route('/predict/<symbol>')
def predict(symbol):
    result = fno_service.predict_probability(symbol)
    return jsonify(result.__dict__)

@app.route('/search')
def search():
    query = request.args.get('query')
    results = fno_service.search_stocks(query)
    return jsonify([r.__dict__ for r in results])
```

### In Jupyter Notebooks

```python
# Import and use in notebooks
from src.fno_rag import FNOEngine, HorizonType

# Initialize
fno = FNOEngine()

# Interactive analysis
import plotly.express as px
results = fno.predict_batch(["RELIANCE", "TCS", "INFY"])
df = pd.DataFrame([r.__dict__ for r in results])
px.bar(df, x='symbol', y='up_probability')
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NSE India for market data
- GROQ for LLM services
- FAISS for vector similarity search
- The open-source community for various libraries

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation

---

**Note**: This system is for educational and research purposes. Always perform your own analysis before making trading decisions.

