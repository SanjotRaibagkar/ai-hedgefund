# Intraday ML Prediction System

A comprehensive machine learning system for intraday trading predictions using options chain data, technical indicators, and market sentiment features.

## 📋 Overview

This system implements a 15-minute ML prediction model that combines:
- **Options Chain Data**: Greeks (Delta, Theta, Vega, Gamma), OI changes, PCR ratios
- **Index Technical Features**: OHLCV data, RSI, MACD, Bollinger Bands, VWAP
- **Market Sentiment**: FII/DII flows, India VIX
- **Target Variable**: 15-minute future price direction (UP/DOWN)

## 🏗️ Architecture

```
src/intradayML/
├── __init__.py              # Module initialization
├── data_collector.py        # Data collection from NSE
├── feature_engineer.py      # Feature engineering
├── model_trainer.py         # ML model training
├── predictor.py             # Prediction engine
├── utils.py                 # Utility functions
├── run_intraday_ml.py       # Main runner script
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
poetry install
```

### 2. Run the System

```bash
# Run all demos
poetry run python src/intradayML/run_intraday_ml.py

# Run specific demo
poetry run python src/intradayML/run_intraday_ml.py --demo data
poetry run python src/intradayML/run_intraday_ml.py --demo features
poetry run python src/intradayML/run_intraday_ml.py --demo training
poetry run python src/intradayML/run_intraday_ml.py --demo prediction
poetry run python src/intradayML/run_intraday_ml.py --demo health
```

## 📊 Data Requirements

### Options Chain Data (Nifty/BankNifty)
- **Greeks**: Delta (Δ), Theta (Θ), Vega, Gamma (optional)
- **Open Interest (OI)**: Changes per strike
- **Put-Call Ratio (PCR)**: Both OI & Volume
- **ATM Option Premium**: CE & PE separately

### Underlying Index Data
- **15-min OHLCV candles**: Nifty/BankNifty spot & futures
- **Technical Indicators**: RSI, MACD, Bollinger Bands, VWAP

### Optional Data
- **FII/DII cash flows**: Daily cash market buy/sell figures
- **India VIX**: Market volatility index

### Labels (Target Variable)
- **Direction**: UP = +1, DOWN = -1
- **Calculation**: `if Close(t+15m) > Close(t): Label = 1 else: Label = -1`

## 🔧 Feature Engineering

### Options Chain Features
- Δ (Delta), Θ (Theta), Vega of ATM options
- Change in OI (CE, PE separately, ATM + 1 OTM)
- PCR OI and PCR Volume
- IV change (ATM CE/PE)

### Index Technical Features
- 15-min returns = (Close - Open)/Open
- RSI (14), MACD (12,26,9), Bollinger Band deviation
- VWAP deviation

### Market Sentiment Features
- FII/DII net flows (normalized)
- India VIX level & 15-min change

## 🤖 ML Models

The system trains multiple models and selects the best performer:

1. **Random Forest**: Robust ensemble method
2. **XGBoost**: Gradient boosting with regularization
3. **LightGBM**: Fast gradient boosting
4. **Gradient Boosting**: Traditional gradient boosting
5. **Logistic Regression**: Linear model baseline
6. **SVM**: Support Vector Machine (for smaller datasets)

### Model Selection
- Uses F1-score for model comparison
- Automatically selects best performing model
- Supports ensemble voting

## 📈 Usage Examples

### Data Collection

```python
from intradayML import IntradayDataCollector

# Initialize collector
collector = IntradayDataCollector()

# Collect all data
data = collector.collect_all_data(['NIFTY', 'BANKNIFTY'])

# Close connection
collector.close()
```

### Feature Engineering

```python
from intradayML import IntradayFeatureEngineer
from datetime import datetime

# Initialize feature engineer
engineer = IntradayFeatureEngineer()

# Create features for current time
features = engineer.create_complete_features('NIFTY', datetime.now())

# Get training data
training_data = engineer.get_training_data('NIFTY', start_date, end_date)

engineer.close()
```

### Model Training

```python
from intradayML import IntradayMLTrainer
from datetime import date, timedelta

# Initialize trainer
trainer = IntradayMLTrainer()

# Train models
start_date = date.today() - timedelta(days=30)
end_date = date.today()
results = trainer.train_models('NIFTY', start_date, end_date)

trainer.close()
```

### Making Predictions

```python
from intradayML import IntradayPredictor
from datetime import datetime

# Initialize predictor
predictor = IntradayPredictor()

# Load models
predictor.load_models_for_index('NIFTY')

# Make prediction
prediction = predictor.predict_current('NIFTY')

# Get feature importance
importance = predictor.get_feature_importance_summary('NIFTY')

predictor.close()
```

### Complete Pipeline

```python
from intradayML import IntradayUtils
from datetime import date, timedelta

# Training pipeline
training_results = IntradayUtils.create_training_pipeline(
    index_symbol='NIFTY',
    start_date=date.today() - timedelta(days=30),
    end_date=date.today()
)

# Prediction pipeline
prediction_results = IntradayUtils.create_prediction_pipeline(
    index_symbol='NIFTY'
)
```

## 📊 Database Schema

### intraday_options_data
- `timestamp`: Timestamp of data collection
- `index_symbol`: NIFTY or BANKNIFTY
- `strike_price`: Option strike price
- `expiry_date`: Option expiry date
- `option_type`: CE or PE
- `last_price`, `bid_price`, `ask_price`: Option prices
- `volume`, `open_interest`, `change_in_oi`: Volume metrics
- `implied_volatility`, `delta`, `gamma`, `theta`, `vega`: Greeks
- `spot_price`, `atm_strike`: Underlying metrics
- `pcr_oi`, `pcr_volume`: PCR ratios

### intraday_index_data
- `timestamp`: Timestamp of data collection
- `index_symbol`: NIFTY or BANKNIFTY
- `open_price`, `high_price`, `low_price`, `close_price`: OHLC
- `volume`, `turnover`: Volume metrics

### intraday_fii_dii_data
- `date`: Trading date
- `fii_buy`, `fii_sell`, `fii_net`: FII flows
- `dii_buy`, `dii_sell`, `dii_net`: DII flows

### intraday_vix_data
- `timestamp`: Timestamp of data collection
- `vix_value`: India VIX value
- `vix_change`: VIX change

### intraday_labels
- `timestamp`: Timestamp of prediction
- `index_symbol`: NIFTY or BANKNIFTY
- `label`: 1 (UP) or -1 (DOWN)
- `future_close`, `current_close`: Price comparison
- `return_pct`: Percentage return

## 🔍 Monitoring & Validation

### System Health Check

```python
from intradayML import IntradayUtils

# Check system health
health = IntradayUtils.validate_system_health()
print(f"System status: {health['overall_status']}")
```

### Data Quality Validation

```python
from intradayML import IntradayUtils

# Validate data quality
quality = IntradayUtils.validate_data_quality(data_df, 'options')
print(f"Quality score: {quality['quality_score']}")
```

### Performance Metrics

- **Accuracy**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## 📈 Visualization

### Feature Importance Plot

```python
from intradayML import IntradayUtils

# Plot feature importance
IntradayUtils.plot_feature_importance(feature_importance, top_n=20)
```

### Prediction Results Plot

```python
from intradayML import IntradayUtils

# Plot prediction results
IntradayUtils.plot_prediction_results(predictions_df)
```

## ⚙️ Configuration

### Database Path
- Default: `data/intraday_ml_data.duckdb`
- Configurable in each component

### Model Directory
- Default: `models/intraday_ml/`
- Stores trained models, scalers, and metadata

### Logging
- Logs stored in: `logs/intraday_ml.log`
- Rotation: Daily
- Retention: 7 days

## 🚨 Important Notes

1. **Market Hours**: System designed for 9:30 AM - 3:30 PM IST trading hours
2. **Data Dependencies**: Requires NSE utility for data collection
3. **Model Training**: Requires sufficient historical data (recommended: 30+ days)
4. **Real-time Usage**: Models should be retrained periodically for best performance
5. **Risk Management**: This is a prediction system, not financial advice

## 🔧 Troubleshooting

### Common Issues

1. **No Data Available**
   - Check NSE utility connectivity
   - Verify market hours
   - Check database permissions

2. **Model Training Fails**
   - Ensure sufficient training data
   - Check feature availability
   - Verify data quality

3. **Prediction Errors**
   - Load models before prediction
   - Check feature vector preparation
   - Verify timestamp validity

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 License

This project is part of the AI Hedge Fund system. Please refer to the main project license.

## 🤝 Contributing

1. Follow the existing code structure
2. Add comprehensive logging
3. Include error handling
4. Update documentation
5. Test thoroughly

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review system health status
3. Check log files for errors
4. Contact the development team
