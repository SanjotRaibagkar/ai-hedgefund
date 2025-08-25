# AI Hedge Fund

A comprehensive AI-powered hedge fund system with real-time options data collection, analysis, and automated trading strategies.

## 🚀 Features

- **Real-time Options Data Collection**: 1-minute interval data for NIFTY and BANKNIFTY
- **Comprehensive Database**: DuckDB-based data storage with 20+ tables
- **AI Agents**: 15+ specialized trading agents (Warren Buffett, Ben Graham, etc.)
- **Backtesting Engine**: ML-based backtesting with MLflow integration
- **Web Interface**: Modern React-based frontend with real-time data visualization
- **Automated Trading**: End-to-end trading pipeline with risk management

## 📊 Current Status

- ✅ **Options Data Collection**: Live and collecting 1,181+ records per minute
- ✅ **Database**: 915,470+ price records, 2,362+ options records
- ✅ **Trading Hours**: 9:30 AM - 3:30 PM IST (automated)
- ✅ **Market Coverage**: NIFTY, BANKNIFTY, 2,129+ equity symbols

## 🛠️ Environment Setup

### Prerequisites

- Python 3.13+
- Poetry (for dependency management)
- Git

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ai-hedge-fund
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   ```

3. **Activate environment**:
   ```bash
   # Windows
   scripts\activate_env.bat
   
   # Unix/Linux/Mac
   source scripts/activate_env.sh
   ```

4. **Verify installation**:
   ```bash
   poetry run python -c "import duckdb; print('✅ Environment ready!')"
   ```

## 🏃‍♂️ Running the System

### Start Options Data Collection

```bash
# Start the options data collector service
poetry run python start_options_collector.py
```

### Monitor Data Collection

```bash
# Check data collection status
poetry run python monitor_options_data.py
```

### Run Tests

```bash
# Run all tests
poetry run pytest

# Run specific test
poetry run python -m src.data.downloaders.test_options_collector
```

### Start Web Interface

```bash
# Start the web application
cd app
poetry run python -m uvicorn main:app --reload
```

## 📁 Project Structure

```
ai-hedge-fund/
├── src/                    # Main source code
│   ├── agents/            # AI trading agents
│   ├── data/              # Data collection & processing
│   ├── ml/                # Machine learning models
│   ├── strategies/        # Trading strategies
│   └── utils/             # Utilities
├── app/                   # Web application
│   ├── frontend/          # React frontend
│   └── backend/           # FastAPI backend
├── data/                  # Database files
├── docs/                  # Documentation
├── tests/                 # Test suite
├── scripts/               # Environment scripts
└── pyproject.toml         # Poetry configuration
```

## 🔧 Configuration

### Environment Variables

The project uses Poetry for dependency management. Key configurations:

- **Python Interpreter**: Poetry-managed virtual environment
- **Database**: `data/comprehensive_equity.duckdb`
- **Logging**: Configured via loguru
- **API Keys**: Managed through the web interface

### IDE Setup

For Cursor/VS Code, the project includes:

- `.vscode/settings.json`: IDE configuration
- `pyrightconfig.json`: Python analysis settings
- `.cursorrules`: Cursor-specific rules

## 📈 Data Collection

### Options Data

- **NIFTY**: 816 records per collection
- **BANKNIFTY**: 365 records per collection
- **Interval**: 1 minute during market hours
- **Coverage**: All strike prices, expiry dates, CE/PE options

### Equity Data

- **Symbols**: 2,129+ equity symbols
- **Records**: 915,470+ price records
- **Date Range**: 2024-01-01 to present
- **Frequency**: Daily OHLCV data

## 🤖 AI Agents

The system includes 15+ specialized AI agents:

- Warren Buffett (Value Investing)
- Ben Graham (Fundamental Analysis)
- Aswath Damodaran (Valuation)
- And 12+ more specialized agents

## 📊 Database Schema

### Key Tables

- `options_chain_data`: Real-time options data
- `price_data`: Historical equity prices
- `fundamental_data`: Company fundamentals
- `technical_data`: Technical indicators
- `market_data`: Market metrics

## 🔍 Monitoring

### Data Quality

- Real-time data validation
- Missing data detection
- Quality metrics tracking
- Automated alerts

### Performance

- Collection success rates
- Database performance
- API response times
- Error tracking

## 🚨 Troubleshooting

### Common Issues

1. **Environment not found**: Run `poetry install`
2. **Database locked**: Check if options collector is running
3. **Import errors**: Ensure `PYTHONPATH=./src`
4. **Package not found**: Use `poetry add <package>`

### Logs

- Application logs: `logs/ai_hedge_fund.log`
- Options collector: Real-time console output
- Database logs: DuckDB internal logging

## 📚 Documentation

- [Options Data Collection](docs/OPTIONS_DATA_COLLECTION.md)
- [Database Status](docs/DATABASE_STATUS_REPORT.md)
- [Setup Guide](docs/SETUP_GUIDE.md)
- [Usage Guide](docs/USAGE_GUIDE.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Run tests: `poetry run pytest`
5. Submit a pull request

## 📄 License

This project is proprietary software. All rights reserved.

---

**🎉 Your AI Hedge Fund is ready to trade!**
