# Enhanced FNO RAG System

## Overview

The Enhanced FNO RAG (Retrieval-Augmented Generation) system is a comprehensive solution for analyzing F&O (Futures & Options) market data using advanced natural language processing and machine learning techniques.

## Features

### 🔍 Advanced Feature Engineering
- **PCR (Put/Call Ratio)**: Calculates put-call ratio for sentiment analysis
- **OI Trends**: Analyzes open interest patterns and trends
- **Buildup Classification**: Identifies long buildup and short covering patterns
- **Implied Volatility**: Calculates simplified implied volatility metrics
- **Price Normalization**: Removes symbol bias by normalizing price data

### 📝 Semantic Snapshot Generation
- Converts daily market data into natural language descriptions
- Example: *"On 27-Aug-2025, NIFTY closed at ₹19,750 (+0.8%). Options data showed high put-call ratio suggesting bearish sentiment with PCR at 1.25. Long buildup was observed with increasing open interest and rising prices. On the next day, NIFTY moved +1.1% (UP)."*

### 🤖 Intelligent RAG Pipeline
- **HuggingFace Embeddings**: Uses `all-MiniLM-L6-v2` for semantic similarity
- **FAISS Vector Store**: Fast similarity search with rich metadata
- **Groq LLM Integration**: Uses `llama-3-70b-chat` for intelligent analysis
- **Natural Language Queries**: Query the system in plain English

### 🎯 Use Cases
- Find similar historical market conditions
- Predict potential price movements based on patterns
- Analyze options flow and sentiment
- Get intelligent insights from historical data

## Installation

### Prerequisites
- Python 3.10+
- Poetry package manager
- GROQ API key (for LLM features)

### Setup
```bash
# Install dependencies
poetry install

# Set environment variables
export GROQ_API_KEY="your_groq_api_key_here"
```

## Usage

### 1. Build Vector Store

```bash
# Build with default settings (last 6 months)
poetry run python build_enhanced_vector_store.py

# Build with custom date range
poetry run python build_enhanced_vector_store.py --start-date 2024-01-01 --end-date 2024-06-30
```

### 2. CLI Interface

#### Interactive Mode
```bash
poetry run python fno_rag_cli.py --interactive
```

#### Single Query
```bash
poetry run python fno_rag_cli.py --query "How much can RELIANCE move tomorrow?"
```

#### Run Examples
```bash
poetry run python fno_rag_cli.py --examples
```

### 3. Example Queries

```python
# Find similar patterns
"Find similar cases where NIFTY rose with high Put OI"
"Show me cases where BANKNIFTY had low PCR and moved up"

# Predict movements
"How much can RELIANCE move tomorrow based on current FNO data?"
"What's the probability of CANBK moving up next week?"

# Analyze patterns
"What happens when there's long buildup in stock futures?"
"Find patterns where implied volatility was high and price moved down"
```

## System Architecture

### Data Flow
1. **Data Ingestion**: Reads from `fno_bhav_copy` table
2. **Feature Engineering**: Calculates PCR, OI trends, buildup patterns
3. **Snapshot Generation**: Creates natural language descriptions
4. **Embedding Generation**: Uses sentence-transformers
5. **Vector Storage**: FAISS index with metadata
6. **Query Processing**: Semantic search + LLM analysis

### Key Components

#### EnhancedFNOVectorStore
- Main class for vector store operations
- Handles data processing, embedding generation, and search

#### FNORAGCLI
- Command-line interface
- Interactive mode and batch processing

#### Feature Engineering
- `_calculate_pcr()`: Put/Call ratio calculation
- `_analyze_oi_trends()`: Open interest analysis
- `_calculate_implied_volatility()`: Volatility metrics
- `_get_next_day_outcome()`: Outcome tracking

## Configuration

### Environment Variables
```bash
GROQ_API_KEY=your_groq_api_key_here
```

### Database Configuration
- **Database**: `data/comprehensive_equity.duckdb`
- **Table**: `fno_bhav_copy`
- **Vector Store**: `data/fno_vectors_enhanced/`

### Model Configuration
- **Embedding Model**: `all-MiniLM-L6-v2`
- **LLM Model**: `llama-3-70b-chat`
- **FAISS Index**: `IndexFlatIP` for cosine similarity

## Output Format

### Similar Cases Response
```
📊 Similar Historical Cases (Top 5):
1. RELIANCE (2024-08-15)
   • Close Price: ₹2,450.50
   • Daily Return: +1.25%
   • PCR: 0.85
   • Implied Vol: 18.5%
   • Next Day: +0.75% (UP)
   • Similarity Score: 0.892
```

### LLM Analysis
```
🤖 AI Analysis:
Based on the similar historical patterns, RELIANCE shows characteristics 
similar to bullish setups with moderate PCR and increasing OI. In similar 
cases, the stock typically moved 0.5-1.5% in the following session. 
Consider support at ₹2,420 and resistance at ₹2,480.
```

## Performance

### Vector Store Statistics
- **Embedding Dimension**: 384 (sentence-transformers)
- **Search Speed**: ~1ms per query
- **Storage**: ~50MB for 6 months of data
- **Memory Usage**: ~200MB for loaded index

### Accuracy Metrics
- **Semantic Similarity**: Cosine similarity scores
- **Pattern Recognition**: Historical outcome tracking
- **LLM Response Quality**: Context-aware analysis

## Troubleshooting

### Common Issues

#### 1. GROQ API Key Not Set
```
Error: LLM service not available. Please install groq and set GROQ_API_KEY.
```
**Solution**: Set the `GROQ_API_KEY` environment variable

#### 2. Vector Store Not Found
```
Error: Vector store files not found
```
**Solution**: Run the build process first: `python build_enhanced_vector_store.py`

#### 3. Database Connection Issues
```
Error: Cannot open database file
```
**Solution**: Ensure the database file exists and has proper permissions

### Performance Optimization

#### For Large Datasets
- Use batch processing for feature calculation
- Implement parallel embedding generation
- Consider using GPU for sentence-transformers

#### Memory Management
- Load vector store on demand
- Use streaming for large result sets
- Implement result pagination

## Development

### Adding New Features

#### 1. New Technical Indicators
```python
def _calculate_new_indicator(self, data):
    # Add your indicator calculation
    return indicator_value
```

#### 2. Custom Embedding Models
```python
# In __init__
self.embedding_model = SentenceTransformer("your-model-name")
```

#### 3. Additional LLM Providers
```python
# Add new LLM client
if NEW_LLM_AVAILABLE:
    self.new_llm_client = NewLLMClient()
```

### Testing
```bash
# Run tests
poetry run pytest tests/test_enhanced_rag.py

# Test specific components
poetry run python -m pytest tests/test_vector_store.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `logs/` directory
3. Create an issue on GitHub

## Roadmap

### Phase 1: Core Features ✅
- [x] Basic vector store implementation
- [x] Feature engineering
- [x] CLI interface

### Phase 2: Advanced Features 🚧
- [ ] Real-time data integration
- [ ] Multi-timeframe analysis
- [ ] Advanced pattern recognition

### Phase 3: Production Features 📋
- [ ] Web interface
- [ ] API endpoints
- [ ] Performance monitoring
- [ ] Automated retraining
