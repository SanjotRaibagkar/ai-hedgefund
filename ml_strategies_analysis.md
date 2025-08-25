# ML Strategies Analysis - AI Hedge Fund

## 📊 **Overview of ML Strategies in `src/ml/` Folder**

This document provides a comprehensive analysis of all ML strategies, their data sources, usage patterns, and data types used in the AI Hedge Fund project.

---

## 🏗️ **ML Strategy Architecture**

### **Core Components:**

1. **ML Strategies** (`ml_strategies.py`)
2. **Feature Engineering** (`feature_engineering.py`)
3. **Backtesting Framework** (`backtesting.py`)
4. **Model Management** (`model_manager.py`)
5. **MLflow Tracking** (`mlflow_tracker.py`)
6. **Options ML Integration** (`options_ml_integration.py`, `enhanced_options_ml_integration.py`)

---

## 📈 **1. ML-Enhanced EOD Strategy (`ml_strategies.py`)**

### **Purpose:**
- Combines traditional momentum strategies with machine learning predictions
- Uses ensemble of ML models for enhanced signal generation
- Provides confidence-weighted trading signals

### **Data Sources:**
- **Primary**: Historical price data from DuckDB database (`data/comprehensive_equity.duckdb`)
- **Secondary**: Technical indicators calculated from price data
- **Fallback**: Yahoo Finance API (if DuckDB data unavailable)

### **Data Types:**
- **Historical Data**: OHLCV (Open, High, Low, Close, Volume) data
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Derived Features**: Momentum, volatility, trend indicators
- **Target Variable**: Future returns (5-day horizon by default)

### **Usage:**
```python
# Training the model
strategy = MLEnhancedEODStrategy()
results = strategy.train_model('RELIANCE.NS', '2023-01-01', '2024-01-01')

# Generating predictions
prediction = strategy.predict('RELIANCE.NS', current_data)
```

### **ML Models Used:**
- XGBoost (default)
- LightGBM
- Random Forest
- Linear Regression (fallback)

---

## 🔧 **2. Feature Engineering (`feature_engineering.py`)**

### **Purpose:**
- Creates comprehensive feature sets for ML models
- Handles data preprocessing and scaling
- Manages feature selection and importance

### **Data Sources:**
- **Price Data**: `get_prices()` from enhanced API
- **Financial Data**: `get_financial_metrics()` (currently disabled)
- **Technical Indicators**: Calculated from price data

### **Feature Categories:**

#### **Technical Features:**
- Price-based: OHLCV, returns, price levels
- Volume-based: Volume trends, volume ratios
- Momentum: RSI, MACD, price momentum
- Volatility: Rolling standard deviation, ATR
- Trend: Moving averages, trend indicators

#### **Fundamental Features:**
- **Status**: Currently disabled (fundamental data not available)
- **Planned**: P/E ratios, P/B ratios, financial metrics

#### **Derived Features:**
- Interaction features (price × volume)
- Lag features (previous day values)
- Rolling features (moving averages)

### **Data Types:**
- **Live Data**: Real-time price data when available
- **Historical Data**: Stored in DuckDB database
- **Calculated Features**: Technical indicators derived from price data

---

## 🧪 **3. Backtesting Framework (`backtesting.py`)**

### **Purpose:**
- Comprehensive backtesting for ML-enhanced strategies
- Portfolio simulation with realistic constraints
- Performance evaluation and metrics calculation

### **Data Sources:**
- **Historical Data**: DuckDB database for backtesting
- **Live Data**: Real-time data for forward testing
- **Market Data**: Current prices for portfolio valuation

### **Backtesting Features:**
- Portfolio simulation with capital constraints
- Commission and slippage modeling
- Risk management and position sizing
- Performance metrics calculation

### **Usage:**
```python
backtester = MLBacktestingFramework()
results = backtester.run_backtest('RELIANCE.NS', '2023-01-01', '2024-01-01')
```

---

## 🎯 **4. Model Management (`model_manager.py`)**

### **Purpose:**
- Manages multiple ML models and ensemble methods
- Handles model training, validation, and selection
- Provides model persistence and loading

### **Data Sources:**
- **Training Data**: Historical data from DuckDB
- **Validation Data**: Split from training data
- **Test Data**: Out-of-sample data for final evaluation

### **Model Types:**
- **Individual Models**: XGBoost, LightGBM, Random Forest, Linear Regression
- **Ensemble Models**: Voting regressor combining multiple models
- **Model Selection**: Based on cross-validation performance

### **Data Types:**
- **Training Data**: Historical OHLCV data with technical features
- **Target Data**: Future returns (configurable horizon)
- **Validation Data**: Time-series split for validation

---

## 📊 **5. MLflow Tracking (`mlflow_tracker.py`)**

### **Purpose:**
- Experiment tracking and model versioning
- Performance metrics logging
- Model artifact management

### **Data Sources:**
- **Experiment Data**: Training metrics and parameters
- **Model Artifacts**: Trained models and feature importance
- **Performance Data**: Backtesting results and metrics

### **Features:**
- Experiment tracking with MLflow
- Model versioning and registry
- Performance comparison across experiments
- Artifact storage and retrieval

---

## ⚡ **6. Options ML Integration**

### **A. Basic Options ML Integration (`options_ml_integration.py`)**

#### **Purpose:**
- Integrates options analysis with ML models
- Provides options-based market sentiment
- Combines options signals with ML predictions

#### **Data Sources:**
- **Options Data**: Live NSE options chain data via `NseUtils`
- **Price Data**: Current spot prices from options chain
- **ML Predictions**: Simulated ML predictions (hardcoded)

#### **Data Types:**
- **Live Data**: Real-time options chain data
- **Simulated Data**: Hardcoded ML predictions (not real ML)
- **Calculated Data**: PCR, OI analysis, signal strength

### **B. Enhanced Options ML Integration (`enhanced_options_ml_integration.py`)**

#### **Purpose:**
- Real ML models instead of simulated data
- Enhanced sentiment logic with conflict detection
- Better signal generation and recommendations

#### **Data Sources:**
- **Options Data**: Live NSE options chain data
- **Price Data**: Historical data for ML model training
- **ML Models**: Real RandomForest models trained on historical data

#### **Data Types:**
- **Live Data**: Real-time options and price data
- **Historical Data**: For ML model training
- **Real ML Predictions**: Trained models on actual data

---

## 🗄️ **Data Source Analysis**

### **Primary Data Sources:**

#### **1. DuckDB Database (`data/comprehensive_equity.duckdb`)**
- **Type**: Historical data storage
- **Content**: OHLCV data for Indian stocks
- **Usage**: Primary source for ML training and backtesting
- **Status**: ✅ Active and used

#### **2. NSE Utility (`src/nsedata/NseUtility.py`)**
- **Type**: Live market data
- **Content**: Real-time options chain, spot prices
- **Usage**: Options analysis and live predictions
- **Status**: ✅ Active and used

#### **3. Enhanced API (`src/tools/enhanced_api.py`)**
- **Type**: Data provider abstraction
- **Content**: Price data, financial metrics
- **Usage**: Feature engineering and data access
- **Status**: ✅ Active and used

#### **4. Yahoo Finance API**
- **Type**: External data source
- **Content**: US stock data, some Indian data
- **Usage**: Fallback for missing data
- **Status**: ⚠️ Limited availability

### **Data Availability Status:**

| Data Type | Source | Status | Usage |
|-----------|--------|--------|-------|
| Historical Prices | DuckDB | ✅ Available | ML Training, Backtesting |
| Live Options | NSE Utility | ✅ Available | Options Analysis |
| Live Prices | NSE Utility | ✅ Available | Real-time Predictions |
| Financial Metrics | Enhanced API | ❌ Not Available | Disabled in Features |
| US Stock Data | Yahoo Finance | ⚠️ Limited | Fallback Only |

---

## 🔄 **Data Flow Architecture**

### **Training Phase:**
```
DuckDB Database → Feature Engineering → ML Models → Model Storage
```

### **Prediction Phase:**
```
Live Data (NSE) → Feature Engineering → Trained Models → Predictions
```

### **Backtesting Phase:**
```
Historical Data (DuckDB) → Feature Engineering → ML Models → Portfolio Simulation
```

---

## 📊 **Current Limitations and Issues**

### **1. Data Availability:**
- **Fundamental Data**: Not available, features disabled
- **Historical Data**: Limited to DuckDB database content
- **Live Data**: Dependent on NSE API availability

### **2. Model Performance:**
- **Options ML**: Uses fallback models when historical data unavailable
- **Feature Engineering**: Limited to technical indicators only
- **Backtesting**: Requires sufficient historical data

### **3. Real-time Capabilities:**
- **Live Predictions**: Available for options analysis
- **ML Model Updates**: Requires retraining with new data
- **Feature Updates**: Real-time technical indicators available

---

## 🚀 **Recommendations for Improvement**

### **1. Data Enhancement:**
- Integrate fundamental data sources
- Expand historical data coverage
- Add alternative data sources

### **2. Model Improvements:**
- Implement online learning for real-time updates
- Add more sophisticated ensemble methods
- Improve feature selection algorithms

### **3. Performance Optimization:**
- Implement caching for frequently used data
- Optimize feature calculation for real-time use
- Add parallel processing for model training

---

## 📈 **Usage Examples**

### **Training ML Models:**
```python
from src.ml.ml_strategies import MLEnhancedEODStrategy

# Initialize strategy
strategy = MLEnhancedEODStrategy()

# Train model on historical data
results = strategy.train_model(
    ticker='RELIANCE.NS',
    start_date='2023-01-01',
    end_date='2024-01-01'
)
```

### **Running Backtests:**
```python
from src.ml.backtesting import MLBacktestingFramework

# Initialize backtester
backtester = MLBacktestingFramework()

# Run backtest
results = backtester.run_backtest(
    ticker='RELIANCE.NS',
    start_date='2023-01-01',
    end_date='2024-01-01'
)
```

### **Options Analysis:**
```python
from src.ml.enhanced_options_ml_integration import EnhancedOptionsMLIntegration

# Initialize enhanced options ML
options_ml = EnhancedOptionsMLIntegration()

# Get options signals
signals = options_ml.get_options_signals(['NIFTY', 'BANKNIFTY'])
```

---

## 🎯 **Summary**

The ML strategies in the AI Hedge Fund project provide a comprehensive framework for:

1. **ML-Enhanced Trading**: Combining traditional strategies with ML predictions
2. **Feature Engineering**: Advanced technical indicator creation
3. **Backtesting**: Realistic portfolio simulation
4. **Model Management**: Multi-model ensemble approach
5. **Options Integration**: Real-time options analysis with ML
6. **Experiment Tracking**: MLflow integration for model management

**Data Sources**: Primarily DuckDB for historical data, NSE Utility for live data
**Data Types**: Historical OHLCV data, live options data, technical indicators
**Status**: Fully functional with some limitations on fundamental data availability
