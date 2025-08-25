# Options Data Collection System

This document describes the comprehensive options data collection system for NIFTY and BANKNIFTY options chain data and futures spot prices.

## Overview

The Options Data Collection System automatically collects:
- **Options Chain Data**: Complete options chain for NIFTY and BANKNIFTY
- **Futures Spot Prices**: Real-time spot prices from futures data
- **1-Minute Intervals**: Data collected every minute during market hours
- **Trading Days Only**: Automatically skips weekends and trading holidays

## Features

### Data Collected
- **Options Data**: Strike prices, expiry dates, option types (CE/PE)
- **Price Data**: Last price, bid price, ask price
- **Volume Data**: Volume, open interest, change in OI
- **Greeks**: Implied volatility (IV), delta, gamma, theta, vega
- **Market Data**: Spot price, ATM strike, PCR (Put-Call Ratio)
- **Timestamps**: Precise timestamps for each data point

### Trading Schedule
- **Market Hours**: 9:30 AM - 3:30 PM IST
- **Trading Days**: Monday to Friday (excluding holidays)
- **Collection Interval**: Every 1 minute
- **Holiday Detection**: Automatic detection using NSE trading holidays

## Database Schema

The data is stored in the `options_chain_data` table in `comprehensive_equity.duckdb`:

```sql
CREATE TABLE options_chain_data (
    timestamp TIMESTAMP,
    index_symbol VARCHAR,
    strike_price DOUBLE,
    expiry_date DATE,
    option_type VARCHAR,
    last_price DOUBLE,
    bid_price DOUBLE,
    ask_price DOUBLE,
    volume BIGINT,
    open_interest BIGINT,
    change_in_oi BIGINT,
    implied_volatility DOUBLE,
    delta DOUBLE,
    gamma DOUBLE,
    theta DOUBLE,
    vega DOUBLE,
    spot_price DOUBLE,
    atm_strike DOUBLE,
    pcr DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (timestamp, index_symbol, strike_price, expiry_date, option_type)
);
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- Required packages: `pandas`, `duckdb`, `schedule`, `loguru`, `psutil`

### Installation
```bash
# Install dependencies
pip install pandas duckdb schedule loguru psutil

# Ensure NseUtility is available
# The system uses src/nsedata/NseUtility.py for data collection
```

## Usage

### 1. Quick Start
```bash
# Start the options data collector
python start_options_collector.py
```

### 2. Using the Scheduler Service
```bash
# Start as a service
python src/data/downloaders/options_scheduler.py start

# Start as a daemon (background process)
python src/data/downloaders/options_scheduler.py start --daemon

# Check service status
python src/data/downloaders/options_scheduler.py status

# Stop the service
python src/data/downloaders/options_scheduler.py stop

# Restart the service
python src/data/downloaders/options_scheduler.py restart
```

### 3. Direct Usage
```python
from src.data.downloaders.options_chain_collector import OptionsChainCollector

# Initialize collector
collector = OptionsChainCollector()

# Collect data for all indices
collector.collect_all_data()

# Collect data for specific index
collector.collect_data_for_index('NIFTY')

# Get recent data
recent_data = collector.get_recent_data('NIFTY', minutes=60)

# Get daily summary
summary = collector.get_daily_summary('NIFTY')
```

## Testing

### Run Tests
```bash
# Run comprehensive tests
python src/data/downloaders/test_options_collector.py
```

### Test Components
```python
# Test individual components
from src.data.downloaders.test_options_collector import *

# Test database connection
test_database_connection()

# Test basic functionality
test_options_collector()

# Test single collection
test_single_collection()
```

## Data Analysis

### Query Examples

#### Get Recent Options Data
```sql
SELECT * FROM options_chain_data 
WHERE index_symbol = 'NIFTY' 
AND timestamp >= datetime('now', '-60 minutes')
ORDER BY timestamp DESC, strike_price, option_type;
```

#### Get ATM Options Data
```sql
SELECT * FROM options_chain_data 
WHERE index_symbol = 'NIFTY' 
AND strike_price = atm_strike
AND timestamp >= datetime('now', '-1 hour')
ORDER BY timestamp DESC;
```

#### Get PCR Trends
```sql
SELECT 
    timestamp,
    index_symbol,
    AVG(pcr) as avg_pcr,
    MAX(pcr) as max_pcr,
    MIN(pcr) as min_pcr
FROM options_chain_data 
WHERE timestamp >= datetime('now', '-1 day')
GROUP BY timestamp, index_symbol
ORDER BY timestamp DESC;
```

#### Get High Volume Options
```sql
SELECT 
    timestamp,
    index_symbol,
    strike_price,
    option_type,
    volume,
    open_interest,
    last_price
FROM options_chain_data 
WHERE volume > 1000
AND timestamp >= datetime('now', '-1 hour')
ORDER BY volume DESC;
```

## Monitoring & Logs

### Log Files
- **Location**: `logs/options_collector.log`
- **Rotation**: Daily
- **Retention**: 7 days
- **Level**: DEBUG

### Monitoring Commands
```bash
# Check if service is running
python src/data/downloaders/options_scheduler.py status

# View recent logs
tail -f logs/options_collector.log

# Check database size
duckdb data/comprehensive_equity.duckdb "SELECT COUNT(*) FROM options_chain_data;"
```

## Configuration

### Trading Hours
Edit `src/data/downloaders/options_chain_collector.py`:
```python
# Trading hours (IST)
self.market_open = "09:30"
self.market_close = "15:30"
```

### Indices
Edit the indices list:
```python
# Indices to track
self.indices = ['NIFTY', 'BANKNIFTY']
```

### Database Path
```python
# Database path
collector = OptionsChainCollector(db_path="path/to/your/database.duckdb")
```

## Troubleshooting

### Common Issues

#### 1. No Data Being Collected
- Check if it's a trading day
- Verify market hours
- Check NSE API connectivity
- Review logs for errors

#### 2. Database Errors
- Ensure DuckDB is installed
- Check database file permissions
- Verify table schema exists

#### 3. Service Won't Start
- Check if another instance is running
- Verify PID file permissions
- Review system resources

#### 4. Rate Limiting
- The system includes delays between API calls
- If issues persist, increase delays in the code

### Debug Mode
```python
# Enable debug logging
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Performance Considerations

### Data Volume
- **Per Minute**: ~200-400 records (NIFTY + BANKNIFTY)
- **Per Day**: ~72,000-144,000 records
- **Storage**: ~50-100 MB per day

### Optimization
- Database indexes are created automatically
- Data is compressed in DuckDB
- Old data can be archived periodically

### Resource Usage
- **CPU**: Low (mostly I/O bound)
- **Memory**: ~50-100 MB
- **Network**: Moderate (API calls every minute)

## Integration

### With Backtesting System
```python
# Use collected data for backtesting
from src.data.downloaders.options_chain_collector import OptionsChainCollector

collector = OptionsChainCollector()
data = collector.get_recent_data('NIFTY', minutes=1440)  # Last 24 hours

# Use data in backtesting strategies
# ... your backtesting code here
```

### With Analysis Tools
```python
# Export data for analysis
import pandas as pd

collector = OptionsChainCollector()
data = collector.get_recent_data('NIFTY', minutes=1440)

# Export to CSV
data.to_csv('nifty_options_data.csv', index=False)

# Export to Excel
data.to_excel('nifty_options_data.xlsx', index=False)
```

## Support

For issues and questions:
1. Check the logs in `logs/options_collector.log`
2. Run the test script: `python src/data/downloaders/test_options_collector.py`
3. Review this documentation
4. Check the main project documentation

## Future Enhancements

- **Additional Indices**: Support for more indices
- **Real-time Alerts**: Price and volume alerts
- **Advanced Analytics**: Built-in PCR analysis, volatility analysis
- **Web Interface**: Dashboard for monitoring
- **Data Export**: Automated exports to external systems
- **Machine Learning**: Integration with ML models for prediction
