# ML Strategies Detailed Analysis - `ml_strategies.py`

## 📊 **Overview**

`ml_strategies.py` is the core ML-enhanced trading strategy module that combines traditional momentum strategies with machine learning predictions. It provides a comprehensive framework for stock analysis, prediction, and trading recommendations.

---

## 🗄️ **Data Sources and Tables**

### **Primary Data Source: DuckDB Database**

#### **Main Table: `price_data`**
- **Location**: `data/comprehensive_equity.duckdb`
- **Schema**:
  ```sql
  CREATE TABLE price_data (
      symbol VARCHAR,
      date DATE,
      open_price DOUBLE,
      high_price DOUBLE,
      low_price DOUBLE,
      close_price DOUBLE,
      volume BIGINT,
      turnover DOUBLE
  );
  ```

#### **Data Flow:**
```
DuckDB Database (price_data table) 
    ↓
DuckDBProvider.get_prices_as_dataframe()
    ↓
Enhanced API (get_prices function)
    ↓
Feature Engineering (create_features)
    ↓
ML Strategies (train_model/predict_returns)
```

### **Secondary Data Sources:**

#### **1. NSE Utility (Live Data)**
- **Purpose**: Real-time options chain and spot prices
- **Usage**: For live predictions and current market data
- **Method**: `NseUtils.get_options_chain()`, `NseUtils.get_spot_price()`

#### **2. Enhanced API (Abstraction Layer)**
- **Purpose**: Data provider abstraction
- **Primary Function**: `get_prices(ticker, start_date, end_date)`
- **Fallback**: Yahoo Finance API for missing data

#### **3. Financial Metrics (Currently Disabled)**
- **Purpose**: Fundamental analysis features
- **Status**: ❌ Not available, features disabled
- **Method**: `get_financial_metrics()` returns empty data

---

## 🔧 **Core Logic and Architecture**

### **1. Strategy Configuration**

#### **Default Strategy Config:**
```python
{
    'momentum_weight': 0.6,        # Weight for traditional momentum signals
    'ml_weight': 0.4,             # Weight for ML predictions
    'confidence_threshold': 0.7,   # Minimum confidence for trading signals
    'min_data_points': 100        # Minimum data points required for training
}
```

#### **Default ML Config:**
```python
{
    'model_type': 'xgboost',      # Default ML model
    'test_size': 0.2,            # Validation split ratio
    'random_state': 42,           # Reproducibility
    'target_horizon': 5,          # Prediction horizon (days)
    'feature_selection': True     # Enable feature selection
}
```

### **2. ML Model Types**

#### **Supported Models:**
1. **XGBoost** (Default)
   ```python
   xgb.XGBRegressor(
       n_estimators=100,
       max_depth=6,
       learning_rate=0.1,
       random_state=42,
       n_jobs=-1
   )
   ```

2. **LightGBM**
   ```python
   lgb.LGBMRegressor(
       n_estimators=100,
       max_depth=6,
       learning_rate=0.1,
       random_state=42,
       n_jobs=-1
   )
   ```

3. **Random Forest**
   ```python
   RandomForestRegressor(
       n_estimators=100,
       max_depth=10,
       random_state=42,
       n_jobs=-1
   )
   ```

4. **Linear Regression** (Fallback)
   ```python
   LinearRegression()
   ```

### **3. Feature Engineering Pipeline**

#### **Technical Features (Active):**
- **Price Features**: OHLCV ratios, price changes, price levels
- **Volume Features**: Volume trends, volume ratios, volume momentum
- **Momentum Features**: RSI, MACD, price momentum indicators
- **Volatility Features**: Rolling standard deviation, ATR, volatility ratios
- **Trend Features**: Moving averages, trend indicators, trend strength

#### **Fundamental Features (Disabled):**
- **Valuation**: P/E, P/B, P/S ratios
- **Financial**: ROE, ROA, profitability metrics
- **Status**: Currently disabled due to data unavailability

#### **Derived Features (Active):**
- **Interaction Features**: RSI × Volume, MACD × Volatility
- **Lag Features**: Previous day values, lagged indicators
- **Rolling Features**: Moving averages, rolling statistics

### **4. Signal Combination Logic**

#### **Momentum Signal Extraction:**
```python
# Extract from traditional momentum analysis
if recommendation['action'] == 'BUY':
    momentum_signal = recommendation.get('confidence', 0.5)
elif recommendation['action'] == 'SELL':
    momentum_signal = -recommendation.get('confidence', 0.5)
```

#### **ML Signal Processing:**
```python
# Normalize ML prediction to [-1, 1] range
ml_signal = np.clip(ml_predictions['latest_prediction'] * 10, -1, 1)
```

#### **Combined Signal Calculation:**
```python
# Weighted combination
combined_signal = (momentum_weight * momentum_signal + ml_weight * ml_signal)
combined_confidence = (momentum_weight * momentum_confidence + ml_weight * ml_confidence)
```

### **5. Recommendation Generation**

#### **Action Determination:**
```python
if combined_signal > confidence_threshold:
    action = 'BUY'
    strength = 'STRONG' if combined_confidence > 0.8 else 'MODERATE'
elif combined_signal < -confidence_threshold:
    action = 'SELL'
    strength = 'STRONG' if combined_confidence > 0.8 else 'MODERATE'
else:
    action = 'HOLD'
    strength = 'NEUTRAL'
```

#### **Position Sizing:**
```python
position_size = min(combined_confidence, 1.0) if action in ['BUY', 'SELL'] else 0.0
```

---

## 📈 **Usage Patterns and Examples**

### **1. Training ML Models**

#### **Basic Training:**
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

# Check training results
print(f"Model Performance: {results['model_performance']}")
print(f"Feature Importance: {results['feature_importance']}")
```

#### **Custom Configuration:**
```python
# Custom strategy configuration
strategy_config = {
    'momentum_weight': 0.7,
    'ml_weight': 0.3,
    'confidence_threshold': 0.8,
    'min_data_points': 200
}

# Custom ML configuration
ml_config = {
    'model_type': 'lightgbm',
    'test_size': 0.3,
    'target_horizon': 10
}

strategy = MLEnhancedEODStrategy(
    strategy_config=strategy_config,
    ml_config=ml_config
)
```

### **2. Making Predictions**

#### **Return Predictions:**
```python
# Predict future returns
predictions = strategy.predict_returns(
    ticker='RELIANCE.NS',
    start_date='2024-01-01',
    end_date='2024-01-31'
)

# Access prediction results
latest_prediction = predictions['latest_prediction']
confidence = predictions['prediction_confidence']
prediction_df = predictions['predictions']
```

### **3. Complete Stock Analysis**

#### **Combined Analysis:**
```python
# Perform complete analysis
analysis = strategy.analyze_stock(
    ticker='RELIANCE.NS',
    start_date='2023-01-01',
    end_date='2024-01-01'
)

# Access analysis results
momentum_analysis = analysis['momentum_analysis']
ml_predictions = analysis['ml_predictions']
combined_signals = analysis['combined_signals']
final_recommendation = analysis['final_recommendation']

# Print recommendation
print(f"Action: {final_recommendation['action']}")
print(f"Strength: {final_recommendation['strength']}")
print(f"Confidence: {final_recommendation['confidence']:.2f}")
print(f"Position Size: {final_recommendation['position_size']:.2f}")
print(f"Reasoning: {final_recommendation['reasoning']}")
```

### **4. Strategy Summary**

#### **Get Strategy Information:**
```python
# Get comprehensive strategy summary
summary = strategy.get_strategy_summary()

print(f"Strategy Type: {summary['strategy_type']}")
print(f"Model Trained: {summary['model_trained']}")
print(f"Model Type: {summary['model_type']}")
print(f"Total Predictions: {summary['total_predictions']}")
print(f"Model Performance: {summary['model_performance']}")
```

---

## 🔄 **Data Processing Pipeline**

### **1. Data Retrieval**
```python
# From DuckDB database
prices_df = get_prices(ticker, start_date, end_date)
# Returns DataFrame with columns: [symbol, date, open_price, high_price, low_price, close_price, volume, turnover]
```

### **2. Feature Creation**
```python
# Technical features
technical_features = self._create_technical_features(prices_df)
# Creates 50+ technical indicators

# Fundamental features (currently empty)
fundamental_features = self._create_fundamental_features(ticker, start_date, end_date)
# Returns empty DataFrame

# Derived features
derived_features = self._create_derived_features(prices_df, technical_features)
# Creates interaction and lag features
```

### **3. Model Training**
```python
# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

# Scale features
X_train_scaled = self.feature_engineer.fit_transform(X_train)
X_test_scaled = self.feature_engineer.transform(X_test)

# Train model
self.ml_model.fit(X_train_scaled, y_train)
```

### **4. Prediction Generation**
```python
# Create features for prediction
features, _ = self.feature_engineer.create_features(ticker, start_date, end_date)

# Scale features
features_scaled = self.feature_engineer.transform(features)

# Make predictions
predictions = self.ml_model.predict(features_scaled)
```

---

## 📊 **Performance Metrics**

### **Training Metrics:**
- **RMSE**: Root Mean Square Error
- **R² Score**: Coefficient of determination
- **MAE**: Mean Absolute Error
- **Training/Test Samples**: Data split information

### **Prediction Metrics:**
- **Prediction Confidence**: Based on feature variance
- **Signal Strength**: Normalized prediction values
- **Combined Confidence**: Weighted average of momentum and ML confidence

### **Strategy Metrics:**
- **Action Strength**: STRONG/MODERATE/NEUTRAL
- **Position Size**: 0.0 to 1.0 based on confidence
- **Signal Direction**: BUY/SELL/HOLD

---

## ⚠️ **Current Limitations**

### **1. Data Availability:**
- **Fundamental Data**: Not available, fundamental features disabled
- **Historical Data**: Limited to DuckDB database content
- **Live Data**: Dependent on NSE API availability

### **2. Model Performance:**
- **Feature Engineering**: Limited to technical indicators only
- **Training Data**: Requires sufficient historical data (minimum 100 points)
- **Model Updates**: Requires retraining with new data

### **3. Integration Issues:**
- **Momentum Framework**: Traditional momentum analysis not fully integrated
- **Real-time Updates**: No online learning capability
- **Multi-timeframe**: Limited to single prediction horizon

---

## 🚀 **Integration with Other Components**

### **1. Feature Engineering:**
- Uses `FeatureEngineer` for comprehensive feature creation
- Integrates with `MomentumIndicators` for technical analysis
- Handles data preprocessing and scaling

### **2. Model Management:**
- Can be integrated with `MLModelManager` for ensemble methods
- Supports MLflow tracking for experiment management
- Provides model persistence capabilities

### **3. Backtesting:**
- Compatible with `MLBacktestingFramework`
- Supports portfolio simulation and performance evaluation
- Integrates with risk management systems

### **4. Options Integration:**
- Can be combined with options analysis for enhanced signals
- Supports multi-asset analysis (stocks + options)
- Provides comprehensive market sentiment analysis

---

## 🎯 **Best Practices and Recommendations**

### **1. Data Quality:**
- Ensure sufficient historical data (minimum 100 data points)
- Validate data completeness and quality
- Handle missing values appropriately

### **2. Model Selection:**
- Start with XGBoost for best performance
- Use cross-validation for model selection
- Monitor model performance over time

### **3. Feature Engineering:**
- Focus on technical indicators for current implementation
- Plan for fundamental data integration when available
- Regular feature importance analysis

### **4. Risk Management:**
- Use confidence thresholds for signal filtering
- Implement position sizing based on confidence
- Monitor model performance and retrain as needed

---

## 📈 **Example Output Structure**

### **Training Results:**
```python
{
    'model_performance': {
        'train_rmse': 0.0234,
        'test_rmse': 0.0256,
        'train_r2': 0.7845,
        'test_r2': 0.7654,
        'train_mae': 0.0189,
        'test_mae': 0.0201,
        'training_samples': 800,
        'test_samples': 200
    },
    'feature_importance': {
        'rsi_14': 0.156,
        'macd': 0.134,
        'volume_sma_ratio': 0.098,
        ...
    },
    'model_type': 'xgboost'
}
```

### **Prediction Results:**
```python
{
    'predictions': DataFrame({
        'date': ['2024-01-01', '2024-01-02', ...],
        'predicted_return': [0.0234, -0.0156, ...],
        'confidence': [0.85, 0.72, ...]
    }),
    'latest_prediction': 0.0234,
    'prediction_confidence': 0.85
}
```

### **Analysis Results:**
```python
{
    'ticker': 'RELIANCE.NS',
    'momentum_analysis': {...},
    'ml_predictions': {...},
    'combined_signals': {
        'momentum_signal': 0.65,
        'ml_signal': 0.78,
        'combined_signal': 0.70,
        'combined_confidence': 0.82
    },
    'final_recommendation': {
        'action': 'BUY',
        'strength': 'STRONG',
        'confidence': 0.82,
        'position_size': 0.82,
        'reasoning': 'Strong positive momentum indicators; ML model predicts positive returns; Strong combined buy signal'
    }
}
```

This comprehensive analysis shows that `ml_strategies.py` provides a robust framework for ML-enhanced trading with real data from DuckDB, sophisticated feature engineering, and comprehensive signal combination logic.
