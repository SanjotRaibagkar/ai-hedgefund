# AI Hedge Fund with Indian Market Integration

A comprehensive AI-powered hedge fund system that integrates Indian stock market data with advanced trading strategies, real-time analysis, and modular architecture.

## 🚀 Features

### Core Features
- **Multi-Agent AI Analysis**: 15+ AI analysts (Warren Buffett, Peter Lynch, Phil Fisher, etc.)
- **Indian Market Integration**: Complete support for NSE/BSE stocks with real-time data
- **Modular Strategy Framework**: 10+ trading strategies (intraday + options)
- **Real-time Data**: NSEUtility integration for live Indian market data
- **Technical & Fundamental Analysis**: Comprehensive stock analysis
- **Risk Management**: Built-in risk assessment and portfolio management

### Indian Market Specific Features
- **NSEUtility Integration**: Real-time NSE data, options chains, market depth
- **Multi-Data Provider Support**: Yahoo Finance, NSEUtility, custom providers
- **Indian Market Calendar**: Trading hours, holidays, corporate actions
- **Currency Support**: INR/USD conversion, Indian numbering system
- **News Integration**: Indian financial news aggregation and sentiment analysis

### Trading Strategies
#### Intraday Strategies (5)
- **Momentum Breakout Strategy**: Identifies breakout opportunities
- **Market Depth Strategy**: Analyzes order book depth
- **VWAP Strategy**: Volume-weighted average price analysis
- **Gap Trading Strategy**: Exploits price gaps
- **Intraday Mean Reversion**: Mean reversion opportunities

#### Options Strategies (5)
- **IV Skew Strategy**: Implied volatility skew analysis
- **Gamma Exposure Strategy**: Gamma risk management
- **Options Flow Strategy**: Unusual options activity tracking
- **Iron Condor Strategy**: Range-bound market strategy
- **Straddle Strategy**: Volatility-based strategy

## 📋 Prerequisites

- Python 3.8+
- Poetry (for dependency management)
- Git
- Internet connection for data fetching

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/SanjotRaibagkar/ai-hedgefund.git
cd ai-hedgefund
```

### 2. Install Poetry (if not installed)
```bash
# Windows
powershell -Command "Invoke-Expression (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content"

# macOS/Linux
curl -sSL https://install.python-poetry.org | python3 -
```

### 3. Install Dependencies
```bash
poetry install
```

### 4. Activate Virtual Environment
```bash
poetry shell
```

## ⚙️ Configuration

### 1. Environment Variables
Create a `.env` file in the root directory:
```bash
# API Keys (optional for basic functionality)
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here

# Database Configuration
DATABASE_URL=sqlite:///ai_hedge_fund.db

# Logging Level
LOG_LEVEL=INFO
```

### 2. Data Provider Configuration
The system automatically detects and uses the best data provider:
- **Indian Stocks**: NSEUtility (default) → Yahoo Finance (fallback)
- **US Stocks**: Yahoo Finance
- **Options Data**: NSEUtility for Indian options

## 🚀 Quick Start

### 1. Test Indian Stock Analysis
```bash
# Analyze an Indian stock
poetry run python src/main.py --ticker RELIANCE.NS

# Analyze multiple stocks
poetry run python src/main.py --ticker "RELIANCE.NS,TCS.NS,HDFCBANK.NS"
```

### 2. Run Strategy Framework
```python
from src.strategies.strategy_manager import get_strategy_manager

# Get strategy manager
manager = get_strategy_manager()

# Execute all strategies
results = manager.execute_all_strategies(market_data)

# Execute specific category
intraday_results = manager.execute_intraday_strategies(market_data)
options_results = manager.execute_options_strategies(market_data)
```

### 3. Use Enhanced API
```python
from src.tools.enhanced_api import get_prices, get_intraday_prices

# Get historical prices
prices = get_prices("RELIANCE.NS", "2024-01-01", "2024-12-31")

# Get intraday data
intraday = get_intraday_prices("RELIANCE.NS")

# Get options data
options = get_live_option_chain("RELIANCE.NS")
```

## 📊 System Architecture

### Data Flow
```
Market Data Sources → Data Providers → Enhanced API → AI Agents → Strategy Framework → Trading Signals
```

### Key Components
1. **Data Providers** (`src/data/providers/`)
   - `NSEUtilityProvider`: Real-time NSE data
   - `YahooFinanceProvider`: Yahoo Finance data
   - `IndianNewsProvider`: Indian financial news
   - `CurrencyProvider`: INR/USD conversion

2. **AI Agents** (`src/agents/`)
   - 15+ specialized AI analysts
   - Each agent has unique investment philosophy
   - Generate buy/sell/hold signals with confidence scores

3. **Strategy Framework** (`src/strategies/`)
   - Modular strategy system
   - Intraday and options strategies
   - Strategy manager for execution

4. **Enhanced API** (`src/tools/enhanced_api.py`)
   - Unified API layer
   - Intelligent provider selection
   - Real-time data access

## 🎯 Usage Examples

### Example 1: Complete Stock Analysis
```bash
# Run full analysis on RELIANCE
poetry run python src/main.py --ticker RELIANCE.NS --analysis full
```

### Example 2: Strategy Testing
```python
from src.strategies.strategy_manager import execute_strategies

# Test intraday strategies
data = {
    "ticker": "RELIANCE.NS",
    "prices": [...],
    "volume": [...],
    "market_depth": {...}
}

results = execute_strategies(data, category="intraday")
print(results)
```

### Example 3: Custom Data Provider
```python
from src.data.providers.provider_factory import get_provider_factory

# Get specific provider
factory = get_provider_factory()
nse_provider = factory.get_nse_utility_provider()

# Use provider directly
prices = nse_provider.get_prices("RELIANCE.NS", "2024-01-01", "2024-12-31")
```

## 📈 Data Sources

### Indian Market Data
- **NSEUtility**: Real-time NSE data, options, market depth
- **Yahoo Finance**: Historical data, financial metrics
- **Indian News**: Financial news aggregation
- **Corporate Actions**: Dividends, splits, bonus issues

### US Market Data
- **Yahoo Finance**: Comprehensive US market data
- **Financial APIs**: Additional data sources

## 🔧 Advanced Configuration

### Custom Strategy Development
```python
from src.strategies.base_strategy import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("My Custom Strategy", "Custom strategy description")
    
    def generate_signals(self, data):
        # Your strategy logic here
        return {"action": "BUY", "confidence": 0.8}
    
    def validate_data(self, data):
        # Data validation logic
        return True
```

### Provider Selection
```python
from src.tools.enhanced_api import get_prices_with_provider

# Force specific provider
prices = get_prices_with_provider("RELIANCE.NS", "2024-01-01", "2024-12-31", provider="nse")
prices = get_prices_with_provider("AAPL", "2024-01-01", "2024-12-31", provider="yahoo")
```

## 🧪 Testing

### Run All Tests
```bash
# Test Indian market integration
poetry run python -c "
from src.tools.enhanced_api import get_prices
prices = get_prices('RELIANCE.NS', '2024-01-01', '2024-12-31')
print(f'Fetched {len(prices)} price records for RELIANCE.NS')
"

# Test strategy framework
poetry run python -c "
from src.strategies.strategy_manager import get_strategy_summary
summary = get_strategy_summary()
print(f'Total strategies: {summary[\"total_strategies\"]}')
"
```

### Test Individual Components
```bash
# Test NSEUtility
poetry run python -c "
from src.nsedata.NseUtility import NseUtils
nse = NseUtils()
info = nse.get_quote('RELIANCE')
print(f'RELIANCE Price: ₹{info[\"lastPrice\"]}')
"
```

## 📚 Documentation

### Phase Documentation
- [Phase 1: Indian Stocks Integration](INDIAN_STOCKS_INTEGRATION.md)
- [Phase 2: Indian Market Specifics](PHASE2_INDIAN_MARKET_SPECIFICS.md)
- [Phase 3: Advanced Features](PHASE3_ADVANCED_FEATURES.md)
- [Phase 4: NSEUtility Integration](PHASE4_NSEUTILITY_INTEGRATION.md)
- [Recovery Checkpoint](RECOVERY_CHECKPOINT.md)

### API Documentation
- [Enhanced API Reference](src/tools/enhanced_api.py)
- [Strategy Framework](src/strategies/)
- [Data Providers](src/data/providers/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Common Issues

1. **Poetry not found**: Install Poetry first
2. **API key errors**: Check your `.env` file
3. **Data fetch failures**: Check internet connection and API limits
4. **Strategy execution errors**: Verify data format and validation

### Getting Help
- Check the [Issues](https://github.com/SanjotRaibagkar/ai-hedgefund/issues) page
- Review the documentation files
- Test individual components

## 🎯 Roadmap

- [ ] Web interface for strategy management
- [ ] Real-time portfolio tracking
- [ ] Advanced risk management
- [ ] Machine learning model integration
- [ ] Multi-exchange support
- [ ] Mobile app

## 🙏 Acknowledgments

- NSEUtility for Indian market data
- Yahoo Finance for comprehensive market data
- All contributors and testers

---

**Note**: This system is for educational and research purposes. Always do your own research before making investment decisions.
