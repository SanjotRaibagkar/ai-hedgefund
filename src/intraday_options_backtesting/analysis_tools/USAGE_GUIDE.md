# 🔮 Options Analyzer V2 - Usage Guide

## 📋 Overview

**Options Analyzer V2** is a comprehensive intraday options analysis tool that provides real-time market predictions based on options chain data. It analyzes PCR, Max Pain, Gamma Exposure, Flow Dynamics, IV Skew, and other technical indicators to generate trading signals.

## 🚀 Quick Start

### 1. Basic Prediction
```python
from options_analyzer_v2 import OptionsAnalyzerV2

# Initialize analyzer
analyzer = OptionsAnalyzerV2("../../../data/options_parquet")

# Load data for a specific date
analyzer.load_data_for_date("20250829")

# Generate prediction for 1:30 PM
result = analyzer.generate_prediction_signal("13:30:00")

print(f"Direction: {result['direction']}")
print(f"Confidence: {result['confidence']}")
print(f"Signal Score: {result['signal_score']}")
```

### 2. Command Line Usage
```bash
# Basic prediction for August 29, 2025 at 1:30 PM
poetry run python predict_next_move_v2.py 20250829

# Custom date and time
poetry run python predict_next_move_v2.py 20250829 --time 11:00:00

# With custom spot price
poetry run python predict_next_move_v2.py 20250829 --time 14:00:00 --spot 24500

# With custom parquet path
poetry run python predict_next_move_v2.py 20250829 --path /path/to/parquet/files
```

## 📊 Available Tools

### 1. **OptionsAnalyzerV2** - Core Analysis Engine
- **File**: `options_analyzer_v2.py`
- **Purpose**: Main analysis class with all prediction methods
- **Key Methods**:
  - `generate_prediction_signal(target_time="13:30:00", spot_price=None)`
  - `calculate_pcr_indicators(snapshot)`
  - `find_max_pain(snapshot)`
  - `calculate_gamma_exposure(snapshot, spot_price)`
  - `analyze_flow_dynamics(snapshot)`
  - `estimate_spot_price(snapshot, target_time)` - **NEW: Advanced multi-factor estimation**

### 2. **Predict Next Move** - Single Prediction Tool
- **File**: `predict_next_move_v2.py`
- **Purpose**: Generate detailed predictions with trading recommendations
- **Features**:
  - Automatic spot price estimation (recommended)
  - Comprehensive technical analysis
  - Trading recommendations
  - Support/Resistance levels
  - **Usage**: `python predict_next_move_v2.py YYYYMMDD --time HH:MM:SS`

### 3. **Batch Predictor** - Multiple Date Analysis
- **File**: `batch_predictor_v2.py`
- **Purpose**: Analyze multiple dates and times for pattern recognition
- **Features**:
  - Batch analysis of multiple dates
  - CSV output for further analysis
  - Statistical summaries
  - **Methods**: `batch_predict_dates()`, `predict_week_range()`

### 4. **Test Suite** - Validation and Testing
- **File**: `test_options_analyzer_v2.py`
- **Purpose**: Test the analyzer at different times
- **Features**:
  - Multiple time slot testing
  - Detailed metrics comparison
  - Performance validation

### 5. **Validation Tool** - Data Quality Check
- **File**: `validate_options_analyzer_v2.py`
- **Purpose**: Validate options data quality and completeness
- **Features**:
  - Data completeness check
  - Time range validation
  - OI distribution analysis

### 6. **Data Analysis Tools** - Additional Utilities
- **File**: `options_data_table.py` - Display options data in table format
- **File**: `parquet_data_analyzer.py` - Analyze parquet file structure and content
- **File**: `check_sept1_data.py` - Specific data validation for September 1st

## 🎯 How to Use for Different Scenarios

### Scenario 1: Intraday Trading Decision
```python
from options_analyzer_v2 import OptionsAnalyzerV2

analyzer = OptionsAnalyzerV2("../../../data/options_parquet")
analyzer.load_data_for_date("20250829")

# Get current market analysis
current_analysis = analyzer.generate_prediction_signal("13:30:00")

if current_analysis['direction'] == "BULLISH":
    if current_analysis['confidence'] == "HIGH":
        print("🚀 Strong bullish signal - Consider buying calls")
    else:
        print("📈 Moderate bullish signal - Cautious call buying")
elif current_analysis['direction'] == "BEARISH":
    if current_analysis['confidence'] == "HIGH":
        print("📉 Strong bearish signal - Consider buying puts")
    else:
        print("🔻 Moderate bearish signal - Cautious put buying")
else:
    print("⚖️ Neutral signal - Range-bound trading")
```

### Scenario 2: Support/Resistance Analysis
```python
# Get support and resistance levels
snapshot = analyzer.get_snapshot_at_time("13:30:00")
levels = analyzer.identify_support_resistance(snapshot)

print("🔴 Resistance Levels:")
for _, row in levels['resistance'].iterrows():
    print(f"   {row['Strike_Price']:,} (Call OI: {row['CALLS_OI']:,.0f})")

print("🟢 Support Levels:")
for _, row in levels['support'].iterrows():
    print(f"   {row['Strike_Price']:,} (Put OI: {row['PUTS_OI']:,.0f})")
```

### Scenario 3: PCR Trend Analysis
```python
# Get intraday momentum
momentum = analyzer.get_intraday_momentum("13:30:00")
print(f"Trend: {momentum['momentum']}")
print(f"PCR Trend: {momentum['pcr_trend']:.3f}")

# Get PCR at different times
times = ["09:30:00", "11:00:00", "13:30:00", "15:00:00"]
for time in times:
    snapshot = analyzer.get_snapshot_at_time(time)
    pcr_data = analyzer.calculate_pcr_indicators(snapshot)
    print(f"{time}: PCR = {pcr_data['pcr_oi']:.3f} ({pcr_data['sentiment']})")
```

### Scenario 4: Unusual Activity Detection
```python
# Detect unusual options activity
snapshot = analyzer.get_snapshot_at_time("13:30:00")
unusual = analyzer.detect_unusual_activity(snapshot)

print("🚨 Unusual Call Activity:")
print(unusual['unusual_calls'])

print("🚨 Unusual Put Activity:")
print(unusual['unusual_puts'])
```

## 📈 Understanding the Output

### Signal Components
- **Direction**: BULLISH, BEARISH, or NEUTRAL
- **Confidence**: HIGH, MEDIUM, or LOW
- **Signal Score**: Range from -1.0 to +1.0
- **Key Factors**: List of contributing factors

### Technical Indicators
- **PCR (OI)**: Put-Call Ratio by Open Interest
- **PCR (Volume)**: Put-Call Ratio by Volume
- **Max Pain**: Strike price with maximum pain for option writers
- **Gamma Environment**: Positive (stabilizing) or Negative (amplifying)
- **Flow Bias**: Bullish, Bearish, or Neutral based on volume/OI flows
- **IV Skew**: Implied Volatility skew interpretation

### Trading Recommendations
- **Strong Signals**: High confidence directional moves
- **Moderate Signals**: Medium confidence with caution
- **Neutral Signals**: Range-bound or wait for clearer signals

## 🔧 Configuration Options

### Path Configuration
```python
# Default parquet path
analyzer = OptionsAnalyzerV2("../../../data/options_parquet")

# Custom path
analyzer = OptionsAnalyzerV2("/path/to/your/parquet/files")
```

### Time Configuration
```python
# Default analysis time
result = analyzer.generate_prediction_signal("13:30:00")

# Custom time
result = analyzer.generate_prediction_signal("11:00:00")

# Multiple times
times = ["09:30:00", "11:00:00", "13:30:00", "15:00:00"]
for time in times:
    result = analyzer.generate_prediction_signal(time)
```

### Spot Price Configuration
```python
# Automatic estimation (recommended) - NEW in V2
result = analyzer.generate_prediction_signal("13:30:00")

# Manual override
result = analyzer.generate_prediction_signal("13:30:00", spot_price=24500)
```

## 📊 Data Requirements

### Required Parquet Structure
- **date**: Trading date (datetime.date object)
- **Fetch_Time**: Timestamp of data collection
- **Symbol**: Option symbol (e.g., "NIFTY")
- **Expiry_Date**: Option expiry date
- **Strike_Price**: Option strike prices
- **CALLS_OI**: Call options open interest
- **PUTS_OI**: Put options open interest
- **CALLS_Volume**: Call options volume
- **PUTS_Volume**: Put options volume
- **CALLS_IV**: Call options implied volatility
- **PUTS_IV**: Put options implied volatility

### File Naming Convention
```
NIFTY_YYYYMMDD_HHMMSS.parquet
Example: NIFTY_20250829_152428.parquet
```

### Directory Structure
```
data/options_parquet/
├── 20250829/
│   ├── NIFTY_20250829_093000.parquet
│   ├── NIFTY_20250829_110000.parquet
│   └── NIFTY_20250829_133000.parquet
└── 20250830/
    └── NIFTY_20250830_093000.parquet
```

## 🚨 Troubleshooting

### Common Issues
1. **"No data found for date"**: Check if parquet files exist for the specified date
2. **"Empty snapshot"**: Verify the time format (HH:MM:SS)
3. **Import errors**: Ensure all files are in the same directory
4. **"No NIFTY files found"**: Verify NIFTY_*.parquet files exist in the date folder

### Data Quality Checks
```python
# Validate data quality
from validate_options_analyzer_v2 import validate_options_analysis
validate_options_analysis()

# Check specific date data
from check_sept1_data import check_sept1_data
check_sept1_data()
```

## 📚 Advanced Usage

### Batch Analysis
```python
from batch_predictor_v2 import batch_predict_dates

# Analyze multiple dates
dates = ["20250829", "20250830", "20250901"]
times = ["09:30:00", "11:00:00", "13:30:00", "15:00:00"]

results = batch_predict_dates(dates, times, output_file="batch_results.csv")
```

### Custom Analysis
```python
# Create custom analysis pipeline
class CustomAnalyzer(OptionsAnalyzerV2):
    def custom_signal_generation(self, snapshot):
        # Your custom logic here
        pass

# Use custom analyzer
custom_analyzer = CustomAnalyzer("../../../data/options_parquet")
```

### Integration with Other Systems
```python
# Export results to CSV
import pandas as pd
results = []
for time in ["09:30:00", "11:00:00", "13:30:00"]:
    result = analyzer.generate_prediction_signal(time)
    results.append({
        'time': time,
        'direction': result['direction'],
        'confidence': result['confidence'],
        'signal_score': result['signal_score']
    })

df = pd.DataFrame(results)
df.to_csv('options_analysis_results.csv', index=False)
```

## 🎯 Best Practices

1. **Use Recent Data**: Always use the most recent parquet files
2. **Automatic Spot Price**: Let V2 estimate spot price automatically (more accurate)
3. **Validate Results**: Cross-check predictions with other indicators
4. **Risk Management**: Never rely solely on one signal
5. **Regular Updates**: Update analysis throughout the trading day
6. **Backtesting**: Use historical data to validate the system

## 🔄 V2 Enhancements

### New Features in V2
- **Advanced Spot Price Estimation**: Multi-factor estimation using Max OI, Volume, Price Change, and Market Movement
- **Improved Scoring System**: More balanced scoring with reduced bullish bias
- **Enhanced Flow Analysis**: Better neutral classification with pressure ratio thresholds
- **Granular Max Pain Analysis**: More detailed thresholds (0.8%, 2.0%) for better signal classification
- **PCR Volume Scoring**: Additional PCR analysis based on volume data

### Scoring Weights (V2)
- **PCR (OI)**: 20% - Put-Call Ratio by Open Interest
- **PCR (Volume)**: 10% - Put-Call Ratio by Volume
- **Max Pain**: 15% - Distance from Max Pain strike
- **Flow Dynamics**: 25% - Call vs Put pressure analysis
- **IV Skew**: 20% - Implied Volatility skew interpretation
- **Gamma Environment**: 10% - Positive/Negative gamma exposure

## 📞 Support

For issues or questions:
1. Check the validation tool output
2. Review data quality in parquet files
3. Ensure all dependencies are installed
4. Verify file paths and permissions
5. Use the test suite to validate functionality

---

**Happy Trading! 🚀📈**
