# 🚀 Quick Start Guide

## Overview

This quick start guide will help you get up and running with the FNO RAG system in minutes.

## Prerequisites

- Python 3.10-3.13
- Poetry (dependency management)
- 8GB+ RAM (for vector operations)
- Internet connection

## Installation

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd ai-hedge-fund

# Install dependencies
poetry install

# Set environment
export PYTHONPATH=./src
```

### 2. Initialize the System

```bash
# Train ML models (takes 2-5 minutes)
poetry run python train_fno_models_optimized.py

# Build vector store (takes 1-2 minutes)
poetry run python build_vector_store_simple.py
```

### 3. Verify Installation

```bash
# Run a quick test
poetry run python test_prediction_simple.py
```

## Basic Usage

### 1. Simple Prediction

```python
from src.fno_rag import FNOEngine, HorizonType

# Initialize the system
fno_engine = FNOEngine()

# Get prediction for NIFTY
result = fno_engine.predict_probability("NIFTY", HorizonType.DAILY)

if result:
    print(f"📊 NIFTY Daily Prediction:")
    print(f"   🟢 Up: {result.up_probability:.1%}")
    print(f"   🔴 Down: {result.down_probability:.1%}")
    print(f"   ⚪ Neutral: {result.neutral_probability:.1%}")
    print(f"   🎯 Confidence: {result.confidence_score:.1%}")
```

### 2. Natural Language Queries

```python
# Ask questions in plain English
queries = [
    "What's the probability of NIFTY moving up tomorrow?",
    "Tell me about RELIANCE's weekly outlook",
    "Find stocks with high daily up probability"
]

for query in queries:
    response = fno_engine.chat(query)
    print(f"Q: {query}")
    print(f"A: {response}\n")
```

### 3. Multi-Timeframe Analysis

```python
# Get predictions for all timeframes
symbol = "NIFTY"

for horizon in [HorizonType.DAILY, HorizonType.WEEKLY, HorizonType.MONTHLY]:
    result = fno_engine.predict_probability(symbol, horizon)
    if result:
        print(f"📈 {symbol} {horizon.value.title()}:")
        print(f"   Up: {result.up_probability:.1%} | Down: {result.down_probability:.1%}")
        print(f"   Confidence: {result.confidence_score:.1%}\n")
```

## Trading Examples

### 1. High Confidence Trading

```python
def find_high_confidence_signals():
    """Find high confidence trading signals."""
    
    symbols = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY"]
    signals = []
    
    for symbol in symbols:
        result = fno_engine.predict_probability(symbol, HorizonType.DAILY)
        
        if result and result.confidence_score > 0.8:
            if result.up_probability > 0.7:
                signals.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'probability': result.up_probability,
                    'confidence': result.confidence_score
                })
            elif result.down_probability > 0.7:
                signals.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'probability': result.down_probability,
                    'confidence': result.confidence_score
                })
    
    return signals

# Find signals
signals = find_high_confidence_signals()
for signal in signals:
    print(f"🎯 {signal['action']} {signal['symbol']}")
    print(f"   Probability: {signal['probability']:.1%}")
    print(f"   Confidence: {signal['confidence']:.1%}")
```

### 2. Risk Management

```python
def calculate_position_size(capital: float, confidence: float, risk_per_trade: float = 0.02):
    """Calculate position size based on confidence."""
    
    base_size = capital * risk_per_trade
    
    if confidence >= 0.8:
        size_multiplier = 1.5  # High confidence
    elif confidence >= 0.6:
        size_multiplier = 1.0  # Medium confidence
    else:
        size_multiplier = 0.5  # Low confidence
    
    return base_size * size_multiplier

# Example usage
capital = 100000  # ₹1 lakh
result = fno_engine.predict_probability("NIFTY", HorizonType.DAILY)

if result:
    position_size = calculate_position_size(capital, result.confidence_score)
    print(f"💰 Recommended Position Size: ₹{position_size:,.2f}")
```

### 3. Portfolio Analysis

```python
def analyze_portfolio(symbols: list):
    """Analyze portfolio of symbols."""
    
    portfolio_analysis = {}
    
    for symbol in symbols:
        # Get multi-timeframe analysis
        daily = fno_engine.predict_probability(symbol, HorizonType.DAILY)
        weekly = fno_engine.predict_probability(symbol, HorizonType.WEEKLY)
        monthly = fno_engine.predict_probability(symbol, HorizonType.MONTHLY)
        
        if daily and weekly and monthly:
            # Calculate weighted probability
            weighted_up = (
                daily.up_probability * 0.5 +
                weekly.up_probability * 0.3 +
                monthly.up_probability * 0.2
            )
            
            portfolio_analysis[symbol] = {
                'daily': daily,
                'weekly': weekly,
                'monthly': monthly,
                'weighted_up': weighted_up,
                'recommendation': 'BUY' if weighted_up > 0.6 else 'SELL' if weighted_up < 0.4 else 'HOLD'
            }
    
    return portfolio_analysis

# Analyze portfolio
portfolio = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY"]
analysis = analyze_portfolio(portfolio)

for symbol, data in analysis.items():
    print(f"📊 {symbol}: {data['recommendation']} ({data['weighted_up']:.1%} up probability)")
```

## Demo Scripts

### 1. Complete System Demo

```bash
# Run the complete demo
poetry run python fno_rag_demo.py
```

This will show:
- ML predictions
- RAG-enhanced predictions
- Natural language queries
- System capabilities

### 2. Trading Decision Guide

```bash
# Run trading guide
poetry run python trading_decision_guide.py
```

This will show:
- Trading strategies
- Risk management rules
- Position sizing examples
- Portfolio analysis

### 3. Quick Backtesting

```bash
# Run quick backtest
poetry run python quick_backtest.py
```

This will:
- Test predictions on historical data
- Show accuracy metrics
- Generate performance report

## Common Use Cases

### 1. Morning Market Analysis

```python
def morning_analysis():
    """Perform morning market analysis."""
    
    print("🌅 Morning Market Analysis")
    print("=" * 50)
    
    key_symbols = ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS', 'INFY']
    
    for symbol in key_symbols:
        result = fno_engine.predict_probability(symbol, HorizonType.DAILY)
        
        if result:
            print(f"\n📊 {symbol}:")
            print(f"   Up: {result.up_probability:.1%} | Down: {result.down_probability:.1%}")
            print(f"   Confidence: {result.confidence_score:.1%}")
            
            if result.confidence_score > 0.7:
                if result.up_probability > 0.6:
                    print(f"   💡 BUY {symbol}")
                elif result.down_probability > 0.6:
                    print(f"   💡 SELL {symbol}")
                else:
                    print(f"   💡 HOLD {symbol}")

# Run morning analysis
morning_analysis()
```

### 2. Intraday Trading Signals

```python
def get_intraday_signals():
    """Get intraday trading signals."""
    
    intraday_symbols = ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS']
    signals = []
    
    for symbol in intraday_symbols:
        result = fno_engine.predict_probability(symbol, HorizonType.DAILY)
        
        if result and result.confidence_score > 0.75:
            direction = 'BUY' if result.up_probability > result.down_probability else 'SELL'
            probability = max(result.up_probability, result.down_probability)
            
            signals.append({
                'symbol': symbol,
                'direction': direction,
                'probability': probability,
                'confidence': result.confidence_score
            })
    
    return signals

# Get intraday signals
signals = get_intraday_signals()
for signal in signals:
    print(f"📈 {signal['direction']} {signal['symbol']} ({signal['probability']:.1%} probability)")
```

### 3. Options Trading Strategy

```python
def options_strategy(symbol: str):
    """Generate options trading strategy."""
    
    result = fno_engine.predict_probability(symbol, HorizonType.DAILY)
    
    if not result or result.confidence_score < 0.7:
        return "Low confidence - avoid options trading"
    
    if result.up_probability > 0.7:
        return {
            'strategy': 'Buy Call Options',
            'reason': f'Strong bullish signal ({result.up_probability:.1%})',
            'confidence': result.confidence_score
        }
    elif result.down_probability > 0.7:
        return {
            'strategy': 'Buy Put Options',
            'reason': f'Strong bearish signal ({result.down_probability:.1%})',
            'confidence': result.confidence_score
        }
    else:
        return {
            'strategy': 'Iron Condor or Straddle',
            'reason': 'Neutral signals - consider volatility strategies',
            'confidence': result.confidence_score
        }

# Generate options strategy
strategy = options_strategy("NIFTY")
print(f"🎯 Options Strategy: {strategy['strategy']}")
print(f"📝 Reason: {strategy['reason']}")
print(f"🎯 Confidence: {strategy['confidence']:.1%}")
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure PYTHONPATH is set
   export PYTHONPATH=./src
   ```

2. **Model Not Found**
   ```bash
   # Retrain models
   poetry run python train_fno_models_optimized.py
   ```

3. **Memory Issues**
   ```bash
   # Use simple vector store builder
   poetry run python build_vector_store_simple.py
   ```

4. **Database Connection**
   ```bash
   # Check database file exists
   ls -la data/comprehensive_equity.duckdb
   ```

### Performance Tips

1. **Use chunked processing** for large datasets
2. **Cache frequently used results**
3. **Monitor memory usage**
4. **Use appropriate batch sizes**

## Next Steps

### 1. Explore Advanced Features

- **RAG Enhancement**: Use `use_rag=True` in predictions
- **Multi-Symbol Analysis**: Analyze multiple symbols simultaneously
- **Custom Strategies**: Build your own trading strategies

### 2. Integration

- **Web Dashboard**: Build a web interface
- **Real-time Data**: Connect to live data feeds
- **Automated Trading**: Integrate with trading platforms

### 3. Customization

- **Model Parameters**: Adjust ML model configurations
- **Feature Engineering**: Add custom technical indicators
- **Risk Management**: Implement custom risk rules

## Support

### Documentation

- **Main README**: `docs/README.md`
- **API Reference**: `docs/API_REFERENCE.md`
- **Trading Guide**: `docs/TRADING_GUIDE.md`
- **ML Models**: `docs/ML_MODELS.md`
- **RAG System**: `docs/RAG_SYSTEM.md`

### Examples

- **Basic Usage**: `test_prediction_simple.py`
- **Complete Demo**: `fno_rag_demo.py`
- **Trading Guide**: `trading_decision_guide.py`
- **Backtesting**: `quick_backtest.py`

### Getting Help

1. **Check Documentation**: Review the docs folder
2. **Run Examples**: Test with provided scripts
3. **Check Logs**: Look for error messages
4. **Community**: Ask questions in discussions

## Quick Reference

### Key Commands

```bash
# Initialize system
poetry run python train_fno_models_optimized.py
poetry run python build_vector_store_simple.py

# Test system
poetry run python test_prediction_simple.py

# Run demos
poetry run python fno_rag_demo.py
poetry run python trading_decision_guide.py

# Backtesting
poetry run python quick_backtest.py
```

### Key Classes

```python
from src.fno_rag import FNOEngine, HorizonType

# Main engine
fno_engine = FNOEngine()

# Prediction
result = fno_engine.predict_probability("NIFTY", HorizonType.DAILY)

# Chat interface
response = fno_engine.chat("What's the probability of NIFTY moving up?")
```

### Key Data Structures

```python
# Prediction result
result.up_probability      # Up probability (0.0-1.0)
result.down_probability    # Down probability (0.0-1.0)
result.confidence_score    # Model confidence (0.0-1.0)

# Timeframes
HorizonType.DAILY         # Daily predictions
HorizonType.WEEKLY        # Weekly predictions
HorizonType.MONTHLY       # Monthly predictions
```

---

**🎉 Congratulations!** You're now ready to use the FNO RAG system for intelligent trading decisions. Start with the basic examples and gradually explore more advanced features as you become comfortable with the system.
