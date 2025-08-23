# AI Hedge Fund - Comprehensive Usage Guide

## 📚 Table of Contents

1. [Quick Start](#quick-start)
2. [Phase 3: EOD Momentum Strategies](#phase-3-eod-momentum-strategies)
3. [Data Infrastructure](#data-infrastructure)
4. [AI Agents System](#ai-agents-system)
5. [Strategy Framework](#strategy-framework)
6. [Data Providers](#data-providers)
7. [Configuration Management](#configuration-management)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Prerequisites Installation
```bash
# Clone the repository
git clone https://github.com/SanjotRaibagkar/ai-hedgefund.git
cd ai-hedgefund

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate environment
poetry shell
```

### Basic Stock Analysis
```bash
# Analyze a single Indian stock
poetry run python src/main.py --ticker RELIANCE.NS

# Analyze multiple stocks
poetry run python src/main.py --ticker "RELIANCE.NS,TCS.NS,HDFCBANK.NS"
```

## 🎯 Phase 3: EOD Momentum Strategies

### Overview
Phase 3 provides production-ready End-of-Day momentum strategies for swing trading with:
- **15+ Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic, etc.
- **Advanced Risk Management**: Multiple stop loss and take profit methods
- **Position Sizing**: 6 different methodologies including Kelly Criterion
- **Portfolio Coordination**: Multi-strategy framework with risk controls

### Quick Setup
```python
from src.strategies.eod.strategy_manager import EODStrategyManager

# Initialize with default configuration
manager = EODStrategyManager(portfolio_value=100000)
```

### 1. Running Daily Analysis

#### Basic Universe Analysis
```python
from src.strategies.eod.strategy_manager import EODStrategyManager
from src.tools.enhanced_api import get_prices
from datetime import datetime, timedelta

# Initialize manager
manager = EODStrategyManager(portfolio_value=100000)

# Prepare universe data (last 100 days)
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d')

universe_data = {}
tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

for ticker in tickers:
    try:
        prices_df = get_prices(ticker, start_date, end_date)
        if prices_df is not None and not prices_df.empty:
            universe_data[ticker] = prices_df
            print(f"✓ Loaded {len(prices_df)} records for {ticker}")
    except Exception as e:
        print(f"✗ Failed to load {ticker}: {e}")

# Run daily analysis
print(f"\n📊 Running analysis for {len(universe_data)} stocks...")
results = manager.run_daily_analysis(universe_data)

# Display results
framework_analysis = results['framework_analysis']
print(f"\n📈 Long Signals: {len(framework_analysis['long_signals'])}")
print(f"📉 Short Signals: {len(framework_analysis['short_signals'])}")
print(f"⏸️  Hold Signals: {len(framework_analysis['hold_signals'])}")

# Show top signals
if framework_analysis['long_signals']:
    print("\n🟢 Top Long Signals:")
    for signal in framework_analysis['long_signals'][:3]:
        print(f"  • {signal['ticker']}: {signal['confidence']:.2f} confidence - {signal['reason']}")

if framework_analysis['short_signals']:
    print("\n🔴 Top Short Signals:")
    for signal in framework_analysis['short_signals'][:3]:
        print(f"  • {signal['ticker']}: {signal['confidence']:.2f} confidence - {signal['reason']}")
```

### 2. Strategy Configuration

#### Custom Configuration
```python
# Create custom configuration
custom_config = {
    "portfolio_settings": {
        "max_positions": 8,
        "max_long_positions": 4,
        "max_short_positions": 4,
        "portfolio_risk_limit": 0.015  # 1.5% risk per trade
    },
    "strategies": {
        "long_momentum": {
            "min_momentum_score": 25.0,  # Higher threshold
            "min_volume_ratio": 2.0,     # Require 2x volume
            "max_holding_period": 15     # Shorter holding period
        }
    }
}

# Update configuration
manager.update_configuration(custom_config)
```

#### Configuration File Management
```python
import json

# Load configuration from file
with open('config/eod_strategies.json', 'r') as f:
    config = json.load(f)

# Modify parameters
config['strategies']['long_momentum']['min_signal_strength'] = 0.4
config['risk_management']['risk_reward_ratio'] = 2.5

# Save configuration
with open('config/custom_eod_config.json', 'w') as f:
    json.dump(config, f, indent=2)

# Use custom configuration
manager = EODStrategyManager(config_path='config/custom_eod_config.json')
```

### 3. Individual Strategy Usage

#### Long Momentum Strategy
```python
from src.strategies.eod.long_momentum import LongMomentumStrategy

# Initialize strategy
long_strategy = LongMomentumStrategy(
    min_momentum_score=20.0,
    min_volume_ratio=1.5,
    max_holding_period=20
)

# Analyze a stock
df = get_prices("RELIANCE.NS", "2024-01-01", "2024-12-31")
analysis = long_strategy.analyze_stock(df, "RELIANCE.NS")

print(f"Signal: {analysis['signal']}")
print(f"Confidence: {analysis['confidence']:.2f}")
print(f"Entry Price: ₹{analysis.get('entry_price', 0):.2f}")
print(f"Stop Loss: ₹{analysis.get('stop_loss', 0):.2f}")
print(f"Take Profit: ₹{analysis.get('take_profit', 0):.2f}")
print(f"Reason: {analysis['reason']}")
```

#### Short Momentum Strategy
```python
from src.strategies.eod.short_momentum import ShortMomentumStrategy

# Initialize strategy
short_strategy = ShortMomentumStrategy(
    min_momentum_score=20.0,
    min_volume_ratio=1.5
)

# Analyze for short opportunities
analysis = short_strategy.analyze_stock(df, "RELIANCE.NS")
print(f"Short Signal: {analysis['signal']}")
print(f"Confidence: {analysis['confidence']:.2f}")
```

### 4. Technical Indicators Usage

#### Calculate All Indicators
```python
from src.strategies.eod.momentum_indicators import MomentumIndicators

# Initialize indicators calculator
indicators = MomentumIndicators()

# Calculate all indicators for a stock
df_with_indicators = indicators.calculate_all_indicators(df)

# Available indicators
print("Available indicators:")
indicator_cols = [col for col in df_with_indicators.columns if col not in ['open_price', 'high_price', 'low_price', 'close_price', 'volume']]
for col in indicator_cols:
    print(f"  • {col}")

# Get momentum signals
momentum_signals = indicators.get_momentum_signals(df_with_indicators)
overall_momentum = indicators.get_overall_momentum_score(momentum_signals)

print(f"\nOverall Momentum Score: {overall_momentum['score']:.1f}")
print(f"Direction: {overall_momentum['direction']}")
print(f"Strength: {overall_momentum['strength']:.3f}")
```

#### Individual Indicator Calculation
```python
# Calculate specific indicators
rsi = indicators._calculate_rsi(df, period=14)
macd_data = indicators._calculate_macd(df)
bollinger = indicators._calculate_bollinger_bands(df)

print(f"Current RSI: {rsi.iloc[-1]:.2f}")
print(f"Current MACD: {macd_data['macd'].iloc[-1]:.4f}")
print(f"Bollinger Upper: {bollinger['bollinger_upper'].iloc[-1]:.2f}")
```

### 5. Risk Management

#### Position Sizing
```python
from src.strategies.eod.position_sizing import PositionSizing

# Initialize position sizing
position_sizer = PositionSizing(
    account_size=100000,
    max_position_size=0.1,  # 10% max per position
    min_position_size=0.01  # 1% minimum
)

# Calculate position size using different methods
methods = ['fixed', 'kelly', 'volatility', 'adaptive']

for method in methods:
    result = position_sizer.calculate_position_size(
        ticker="RELIANCE.NS",
        current_price=2500.0,
        signal_strength=0.8,
        volatility=0.02,
        method=method
    )
    
    print(f"\n{method.title()} Sizing:")
    print(f"  Position Value: ₹{result['position_value']:,.0f}")
    print(f"  Shares: {result['shares']}")
    print(f"  Account %: {result['account_percentage']:.1%}")
```

#### Risk Management
```python
from src.strategies.eod.risk_management import RiskManager

# Initialize risk manager
risk_manager = RiskManager(
    max_portfolio_risk=0.02,
    max_position_risk=0.01,
    max_drawdown=0.15
)

# Calculate stop loss
stop_loss_info = risk_manager.calculate_stop_loss(
    ticker="RELIANCE.NS",
    entry_price=2500.0,
    position_type='long',
    method='atr',
    volatility=0.02
)

print(f"Stop Loss: ₹{stop_loss_info['stop_loss']:.2f}")
print(f"Risk Amount: ₹{stop_loss_info['risk_amount']:.2f}")

# Calculate take profit
take_profit_info = risk_manager.calculate_take_profit(
    ticker="RELIANCE.NS",
    entry_price=2500.0,
    stop_loss=stop_loss_info['stop_loss'],
    position_type='long',
    method='fixed_ratio',
    risk_reward_ratio=2.0
)

print(f"Take Profit: ₹{take_profit_info['take_profit']:.2f}")
print(f"Risk/Reward: 1:{take_profit_info['risk_reward_ratio']:.1f}")
```

### 6. Performance Monitoring

#### Strategy Performance
```python
# Get strategy performance over time
performance = manager.get_strategy_performance(
    strategy_name="long_momentum",
    start_date=datetime.now() - timedelta(days=30)
)

print("Strategy Performance:")
print(f"  Total Signals: {performance['strategies']['long_momentum']['total_signals']}")
print(f"  Buy Signals: {performance['strategies']['long_momentum']['buy_signals']}")
print(f"  Average Confidence: {performance['strategies']['long_momentum']['avg_confidence']:.2f}")
```

#### Manager Summary
```python
# Get comprehensive manager summary
summary = manager.get_manager_summary()

print("Portfolio Summary:")
print(f"  Portfolio Value: ₹{summary['manager_info']['portfolio_value']:,.0f}")
print(f"  Total Positions: {summary['manager_info']['total_positions']}")
print(f"  Active Strategies: {summary['manager_info']['active_strategies']}")
print(f"  Performance History: {summary['performance_history_length']} records")
```

## 🗄️ Data Infrastructure

### Database Operations
```python
from src.data.database.duckdb_manager import DatabaseManager

# Initialize database
db_manager = DatabaseManager("data/ai_hedge_fund.db")

# Get latest data
latest_date = db_manager.get_latest_data_date("RELIANCE.NS")
print(f"Latest data for RELIANCE.NS: {latest_date}")

# Get technical data
technical_data = db_manager.get_technical_data(
    ticker="RELIANCE.NS",
    start_date="2024-01-01",
    end_date="2024-12-31"
)
print(f"Retrieved {len(technical_data)} technical records")
```

### Historical Data Collection
```python
from src.data.collectors.async_data_collector import AsyncDataCollector

# Initialize collector
collector = AsyncDataCollector(db_manager)

# Collect historical data for a ticker
result = await collector.collect_historical_data(
    ticker="RELIANCE.NS",
    start_date="2019-01-01",
    end_date="2024-12-31",
    data_types=['technical', 'fundamental']
)

print(f"Collection Summary:")
print(f"  Technical Records: {result['technical_data']['records_collected']}")
print(f"  Fundamental Records: {result['fundamental_data']['records_collected']}")
```

### Daily Updates
```python
from src.data.update.daily_updater import DailyDataUpdater

# Initialize updater
updater = DailyDataUpdater(db_manager)

# Update all configured tickers
update_results = updater.update_all_tickers()

print("Update Results:")
for ticker, result in update_results.items():
    print(f"  {ticker}: {result['status']} - {result.get('records_added', 0)} new records")
```

## 🤖 AI Agents System

### Using Individual Agents
```python
from src.agents.warren_buffett import WarrenBuffettAgent

# Initialize agent
buffett = WarrenBuffettAgent()

# Get analysis
ticker = "RELIANCE.NS"
prices = get_prices(ticker, "2024-01-01", "2024-12-31")
financials = get_financial_metrics(ticker, "2024-01-01", "2024-12-31")

analysis = buffett.analyze_investment(ticker, prices, financials)
print(f"Warren Buffett's Analysis:")
print(f"  Action: {analysis['action']}")
print(f"  Confidence: {analysis['confidence']:.2f}")
print(f"  Reasoning: {analysis['reasoning']}")
```

### Running All Agents
```python
from src.main import run_analysis

# Run comprehensive analysis with all agents
results = run_analysis("RELIANCE.NS")

print("AI Agents Consensus:")
for agent_name, result in results.items():
    print(f"  {agent_name}: {result['action']} ({result['confidence']:.2f})")
```

## 🔧 Strategy Framework

### Legacy Intraday Strategies
```python
from src.strategies.strategy_manager import get_strategy_manager

# Get strategy manager
manager = get_strategy_manager()

# Execute intraday strategies
market_data = {
    "ticker": "RELIANCE.NS",
    "prices": prices_df,
    "volume": volume_data,
    "market_depth": depth_data
}

intraday_results = manager.execute_intraday_strategies(market_data)
print(f"Intraday strategies executed: {len(intraday_results)}")
```

### Options Strategies
```python
# Execute options strategies
options_data = {
    "ticker": "RELIANCE.NS",
    "option_chain": option_chain_data,
    "iv_data": implied_vol_data
}

options_results = manager.execute_options_strategies(options_data)
print(f"Options strategies executed: {len(options_results)}")
```

## 📡 Data Providers

### Using Specific Providers
```python
from src.data.providers.provider_factory import get_provider_factory

# Get provider factory
factory = get_provider_factory()

# Use NSE provider
nse_provider = factory.get_nse_utility_provider()
prices = nse_provider.get_prices("RELIANCE.NS", "2024-01-01", "2024-12-31")

# Use Yahoo Finance provider
yahoo_provider = factory.get_yahoo_provider()
financial_metrics = yahoo_provider.get_financial_metrics("RELIANCE.NS", "2024-01-01", "2024-12-31")
```

### Enhanced API Usage
```python
from src.tools.enhanced_api import (
    get_prices, get_intraday_prices, get_financial_metrics,
    get_live_option_chain, get_market_depth
)

# Get different types of data
historical_prices = get_prices("RELIANCE.NS", "2024-01-01", "2024-12-31")
intraday_data = get_intraday_prices("RELIANCE.NS")
option_chain = get_live_option_chain("RELIANCE.NS")
market_depth = get_market_depth("RELIANCE.NS")

print(f"Historical records: {len(historical_prices)}")
print(f"Intraday records: {len(intraday_data)}")
print(f"Options available: {len(option_chain)}")
```

## ⚙️ Configuration Management

### EOD Strategy Configuration
```json
{
  "portfolio_settings": {
    "initial_value": 100000,
    "max_positions": 10,
    "max_long_positions": 5,
    "max_short_positions": 5,
    "portfolio_risk_limit": 0.02,
    "correlation_threshold": 0.7
  },
  "strategies": {
    "long_momentum": {
      "enabled": true,
      "min_signal_strength": 0.3,
      "min_momentum_score": 20.0,
      "min_volume_ratio": 1.5,
      "max_holding_period": 20
    }
  },
  "position_sizing": {
    "method": "adaptive",
    "max_position_size": 0.1,
    "min_position_size": 0.01
  },
  "risk_management": {
    "max_portfolio_risk": 0.02,
    "stop_loss_method": "adaptive",
    "risk_reward_ratio": 2.0
  }
}
```

### Environment Configuration
```bash
# .env file
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
DATABASE_URL=sqlite:///ai_hedge_fund.db
LOG_LEVEL=INFO
```

## 🔬 Advanced Usage

### Custom Strategy Development
```python
from src.strategies.eod.momentum_framework import MomentumStrategyFramework

class CustomMomentumStrategy(MomentumStrategyFramework):
    def __init__(self):
        super().__init__(framework_name="Custom Momentum")
        
    def analyze_universe(self, universe_data):
        # Custom analysis logic
        results = super().analyze_universe(universe_data)
        
        # Add custom filtering
        filtered_long_signals = []
        for signal in results['long_signals']:
            if signal['confidence'] > 0.7:  # Higher confidence threshold
                filtered_long_signals.append(signal)
        
        results['long_signals'] = filtered_long_signals
        return results
```

### Backtesting Framework
```python
from datetime import datetime, timedelta

# Simple backtesting example
def backtest_strategy(strategy, universe_data, start_date, end_date):
    results = []
    current_date = start_date
    
    while current_date <= end_date:
        # Get data for current date
        daily_data = get_data_for_date(universe_data, current_date)
        
        # Run strategy analysis
        analysis = strategy.analyze_universe(daily_data)
        
        # Record results
        results.append({
            'date': current_date,
            'signals': len(analysis['long_signals']) + len(analysis['short_signals']),
            'long_signals': len(analysis['long_signals']),
            'short_signals': len(analysis['short_signals'])
        })
        
        current_date += timedelta(days=1)
    
    return results

# Run backtest
backtest_results = backtest_strategy(
    manager.frameworks['momentum_framework'],
    universe_data,
    datetime(2024, 1, 1),
    datetime(2024, 12, 31)
)
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Data Fetching Errors
```python
# Test data connectivity
from src.tools.enhanced_api import test_data_providers

test_results = test_data_providers()
for provider, status in test_results.items():
    print(f"{provider}: {'✓' if status else '✗'}")
```

#### 2. Strategy Execution Errors
```python
# Debug strategy execution
try:
    results = manager.run_daily_analysis(universe_data)
except Exception as e:
    print(f"Strategy execution failed: {e}")
    
    # Check data quality
    for ticker, df in universe_data.items():
        if df.empty:
            print(f"Empty data for {ticker}")
        elif len(df) < 50:
            print(f"Insufficient data for {ticker}: {len(df)} records")
```

#### 3. Database Issues
```python
# Test database connectivity
try:
    db_manager = DatabaseManager()
    tables = db_manager._get_table_list()
    print(f"Database tables: {tables}")
except Exception as e:
    print(f"Database connection failed: {e}")
```

#### 4. Configuration Errors
```python
# Validate configuration
def validate_config(config):
    required_keys = ['portfolio_settings', 'strategies', 'risk_management']
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        print(f"Missing configuration keys: {missing_keys}")
        return False
    
    print("Configuration validation passed")
    return True

# Test configuration
with open('config/eod_strategies.json', 'r') as f:
    config = json.load(f)
    
validate_config(config)
```

### Performance Optimization

#### 1. Data Loading Optimization
```python
# Use efficient data loading
import pandas as pd

def load_universe_efficiently(tickers, days=100):
    universe_data = {}
    
    # Load in parallel (if implemented)
    for ticker in tickers:
        df = get_prices(ticker, days=days)
        if df is not None and len(df) >= 50:  # Minimum data requirement
            universe_data[ticker] = df
    
    return universe_data
```

#### 2. Strategy Execution Optimization
```python
# Optimize strategy parameters for faster execution
fast_config = {
    "strategies": {
        "long_momentum": {
            "min_signal_strength": 0.5,  # Higher threshold = fewer calculations
            "min_momentum_score": 30.0   # Higher threshold = faster filtering
        }
    }
}

manager.update_configuration(fast_config)
```

### Logging and Debugging

#### Enable Debug Logging
```python
import logging
from loguru import logger

# Enable debug logging
logger.add("debug.log", level="DEBUG")

# Or use Python logging
logging.basicConfig(level=logging.DEBUG)
```

#### Strategy Debug Mode
```python
# Enable debug mode in strategies
debug_manager = EODStrategyManager(portfolio_value=100000)

# Add debug logging to see detailed analysis
results = debug_manager.run_daily_analysis(universe_data)

# Check individual strategy results
for strategy_name, strategy_results in results['individual_analysis'].items():
    print(f"\n{strategy_name} Results:")
    for ticker, result in strategy_results.items():
        if result['signal'] != 'hold':
            print(f"  {ticker}: {result['signal']} (confidence: {result['confidence']:.2f})")
```

---

This comprehensive usage guide covers all major features and components of the AI Hedge Fund system. For specific questions or advanced customization, refer to the source code documentation or create an issue on GitHub.