# AI Hedge Fund with Indian Market Integration

A comprehensive AI-powered hedge fund system that integrates Indian stock market data with advanced trading strategies, real-time analysis, and modular architecture.

## 🚀 Features

### Core Features
- **Multi-Agent AI Analysis**: 15+ AI analysts (Warren Buffett, Peter Lynch, Phil Fisher, etc.)
- **Indian Market Integration**: Complete support for NSE/BSE stocks with real-time data
- **EOD Momentum Strategies**: Production-ready swing trading strategies with advanced risk management
- **Machine Learning Integration**: ML-enhanced strategies with comprehensive backtesting
- **Modular Strategy Framework**: 12+ trading strategies (EOD + intraday + options)
- **Real-time Data**: NSEUtility integration for live Indian market data
- **Technical & Fundamental Analysis**: Comprehensive stock analysis with 15+ indicators
- **Advanced Risk Management**: Multi-method stop loss, position sizing, and portfolio controls
- **Data Infrastructure**: SQLite database with historical data collection and daily updates

### Indian Market Specific Features
- **NSEUtility Integration**: Real-time NSE data, options chains, market depth
- **Multi-Data Provider Support**: Yahoo Finance, NSEUtility, custom providers
- **Indian Market Calendar**: Trading hours, holidays, corporate actions
- **Currency Support**: INR/USD conversion, Indian numbering system
- **News Integration**: Indian financial news aggregation and sentiment analysis

### Trading Strategies
#### EOD Momentum Strategies (2) - **Phase 3**
- **Long Momentum Strategy**: Bullish momentum-based swing trading
- **Short Momentum Strategy**: Bearish momentum-based swing trading
- **Advanced Technical Analysis**: 15+ indicators (RSI, MACD, Bollinger Bands, etc.)
- **Risk Management**: Multiple stop loss/take profit methods
- **Position Sizing**: 6 different sizing methodologies
- **Portfolio Coordination**: Multi-strategy framework

#### Machine Learning Enhanced Strategies - **Phase 4**
- **ML-Enhanced EOD Strategy**: Combines traditional momentum with ML predictions
- **Multi-Model Framework**: XGBoost, LightGBM, Random Forest, Linear Regression
- **Advanced Feature Engineering**: 50+ technical and fundamental features
- **MLflow Integration**: Complete experiment tracking and model versioning
- **Comprehensive Backtesting**: Realistic portfolio simulation with costs
- **Ensemble Methods**: Voting regressor for optimal model combination

#### Stock Screening & Analysis - **NEW in Phase 5!**
- **EOD Stock Screener**: Bullish/bearish signals with entry, stop loss, and targets
- **Intraday Stock Screener**: Breakout, reversal, and momentum detection
- **Options Analyzer**: Nifty & BankNifty analysis with OI and volatility metrics
- **Market Predictor**: Multi-timeframe predictions (15min to multi-day)
- **Modern Web UI**: Professional dashboard with MokshTechandInvestment branding
- **Comprehensive Screening Manager**: Unified interface for all screening modules

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
# Install all dependencies
poetry install

# Install ML dependencies (for Phase 4 features)
poetry install --with ml

# Install screening dependencies (for Phase 5 features)
poetry install --with screening
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

## 🚀 Quick Start

### 1. Basic Usage
```bash
# Run the main application
poetry run python src/main.py --ticker RELIANCE.NS

# Run with specific date range
poetry run python src/main.py --ticker RELIANCE.NS --start-date 2024-01-01 --end-date 2024-12-31
```

### 2. Strategy Analysis
```bash
# Analyze with specific strategy
poetry run python src/main.py --ticker RELIANCE.NS --strategy long_momentum

# Run EOD momentum analysis
poetry run python src/main.py --ticker RELIANCE.NS --strategy eod_momentum
```

### 3. ML-Enhanced Analysis (Phase 4)
```bash
# Run ML-enhanced strategy
poetry run python -c "
from src.ml.ml_strategies import MLEnhancedEODStrategy
strategy = MLEnhancedEODStrategy()
result = strategy.train_model('AAPL', '2023-01-01', '2023-12-31')
print('ML model training completed')
"

# Run comprehensive backtesting
poetry run python -c "
from src.ml.backtesting import MLBacktestingFramework
framework = MLBacktestingFramework()
results = framework.run_backtest('AAPL', '2023-01-01', '2023-12-31')
print('Backtesting completed')
"
```

## 📊 Data Infrastructure

### Historical Data Collection
```bash
# Collect historical data for multiple tickers
poetry run python -c "
from src.data.collectors.async_data_collector import AsyncDataCollector
from src.data.database.duckdb_manager import DatabaseManager

db_manager = DatabaseManager()
collector = AsyncDataCollector(db_manager)

# Collect data for Indian stocks
tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
result = collector.collect_multiple_tickers(tickers, '2023-01-01', '2023-12-31')
print(f'Collected data for {len(result)} tickers')
"
```

### Daily Data Updates
```bash
# Run daily data updates
poetry run python -c "
from src.data.update.update_manager import UpdateManager
manager = UpdateManager()
manager.run_daily_updates()
print('Daily updates completed')
"
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
- [**Phase 3 Completion Summary**](PHASE3_COMPLETION_SUMMARY.md)
- [**Phase 4 Completion Summary**](PHASE4_COMPLETION_SUMMARY.md)
- [**Phase 5 Completion Summary**](PHASE5_COMPLETION_SUMMARY.md) - **NEW!**
- [Recovery Checkpoint](RECOVERY_CHECKPOINT.md)

### API Documentation
- [Enhanced API Reference](src/tools/enhanced_api.py)
- [Strategy Framework](src/strategies/)
- [Data Providers](src/data/providers/)
- [ML Components](src/ml/) - **NEW!**

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
5. **ML dependencies**: Run `poetry install --with ml` for Phase 4 features

### Getting Help
- Check the [Issues](https://github.com/SanjotRaibagkar/ai-hedgefund/issues) page
- Review the documentation files
- Test individual components

## 🎯 Roadmap

### Phase 5: Advanced ML Features (Next)
- [ ] Deep Learning Models (LSTM, Transformers)
- [ ] Reinforcement Learning strategies
- [ ] Advanced ensemble methods
- [ ] Online learning capabilities

### Future Phases
- [ ] Zipline backtesting integration
- [ ] Web interface for strategy management  
- [ ] Real-time portfolio tracking
- [ ] Multi-exchange support
- [ ] Mobile app

### ✅ Completed Phases
- ✅ **Phase 1**: Indian Stock Market Integration
- ✅ **Phase 2**: Data Infrastructure & Daily Updates
- ✅ **Phase 3**: EOD Momentum Strategies
- ✅ **Phase 4**: Machine Learning Integration
- ✅ **Phase 5**: Stock Screening & Advanced UI System

## 🙏 Acknowledgments

- NSEUtility for Indian market data
- Yahoo Finance for comprehensive market data
- All contributors and testers

---

**Note**: This system is for educational and research purposes. Always do your own research before making investment decisions.
