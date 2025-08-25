# Data Directory - AI Hedge Fund

## Overview

This directory contains all data-related files, databases, and configuration for the AI Hedge Fund system. It includes options chain data, equity data, and various data collection and analysis components.

## 📁 Directory Structure

```
data/
├── README.md                           # This file
├── comprehensive_equity.duckdb         # Main equity database
├── options_chain_data.duckdb          # Options chain data database
├── fundamental_data/                   # Fundamental data storage
├── sqlite_backup/                      # SQLite backup files
└── nse_equity_symbols_complete.txt    # NSE equity symbols list
```

## 🗄️ Database Files

### 1. Comprehensive Equity Database
**File**: `comprehensive_equity.duckdb`  
**Purpose**: Main database for equity data and analysis  

**Tables**:
- `equity_data` - Historical equity price data
- `fundamental_data` - Company fundamental information
- `screening_results` - Stock screening results
- `backtest_results` - Backtesting results
- `ml_predictions` - Machine learning predictions

**Usage**:
```python
import duckdb
conn = duckdb.connect('data/comprehensive_equity.duckdb')
```

---

### 2. Options Chain Data Database
**File**: `options_chain_data.duckdb`  
**Purpose**: Dedicated database for options chain data collection  

**Tables**:
- `options_chain_data` - Real-time options chain data

**Schema**:
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
)
```

**Usage**:
```python
import duckdb
conn = duckdb.connect('data/options_chain_data.duckdb')
```

---

## 📊 Data Collection Schedulers

### Options Data Collection
**Scheduler**: `src/data/downloaders/options_scheduler.py`  
**Database**: `data/options_chain_data.duckdb`  
**Interval**: 1 minute during market hours  
**Market Hours**: 9:30 AM - 3:30 PM IST  

**Commands**:
```bash
# Start data collection
poetry run python src/data/downloaders/options_scheduler.py start

# Stop data collection
poetry run python src/data/downloaders/options_scheduler.py stop

# Check status
poetry run python src/data/downloaders/options_scheduler.py status
```

### Auto Options Collector
**Scheduler**: `src/data/downloaders/auto_options_collector.py`  
**Purpose**: Enhanced data collection with automatic scheduling  

**Features**:
- ✅ Automatic market hours detection
- ✅ Trading day validation
- ✅ Error handling and recovery
- ✅ Performance monitoring

**Usage**:
```bash
poetry run python start_auto_options_collector.py
```

---

## 🔧 Database Management

### Database Manager
**File**: `src/data/database/duckdb_manager.py`  
**Purpose**: Main database management for comprehensive equity data  

**Features**:
- ✅ Connection management
- ✅ Table creation and maintenance
- ✅ Data insertion and retrieval
- ✅ Performance optimization

### Options Database Manager
**File**: `src/data/database/options_db_manager.py`  
**Purpose**: Dedicated management for options chain data  

**Features**:
- ✅ Options-specific table management
- ✅ Real-time data insertion
- ✅ Index optimization
- ✅ Data retrieval utilities

---

## 📈 Data Analysis

### Options Analysis
**Analyzer**: `src/screening/fixed_enhanced_options_analyzer.py`  
**Output**: `results/options_tracker/option_tracker.csv`  

**Analysis Features**:
- ✅ ATM ± 2 strikes analysis
- ✅ PCR (Put-Call Ratio) calculation
- ✅ OI (Open Interest) analysis
- ✅ Signal generation
- ✅ Trade recommendations

### Equity Screening
**Screener**: `src/screening/enhanced_eod_screener.py`  
**Database**: `data/comprehensive_equity.duckdb`  

**Screening Features**:
- ✅ Technical indicators
- ✅ Fundamental metrics
- ✅ Volume analysis
- ✅ Price momentum
- ✅ Risk assessment

---

## 🔍 Monitoring & Maintenance

### Data Quality Monitor
**File**: `src/data/update/data_quality_monitor.py`  
**Purpose**: Monitor data quality and integrity  

**Features**:
- ✅ Data completeness checks
- ✅ Data accuracy validation
- ✅ Missing data detection
- ✅ Performance monitoring

### Maintenance Scheduler
**File**: `src/data/update/maintenance_scheduler.py`  
**Purpose**: Automated database maintenance  

**Tasks**:
- ✅ Database optimization
- ✅ Index rebuilding
- ✅ Data cleanup
- ✅ Backup management

---

## 📊 Data Statistics

### Quick Stats Commands
```python
# Check options data count
import duckdb
conn = duckdb.connect('data/options_chain_data.duckdb')
count = conn.execute('SELECT COUNT(*) FROM options_chain_data').fetchone()[0]
print(f"Options records: {count:,}")

# Check equity data count
conn = duckdb.connect('data/comprehensive_equity.duckdb')
count = conn.execute('SELECT COUNT(*) FROM equity_data').fetchone()[0]
print(f"Equity records: {count:,}")
```

### Database Size Monitoring
```bash
# Check database file sizes
ls -lh data/*.duckdb

# Monitor growth
du -h data/ --max-depth=1
```

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. Database Lock
**Symptom**: `IO Error: Cannot open file... being used by another process`
**Solution**:
```bash
# Check for processes using the database
tasklist | findstr python

# Terminate specific process
taskkill /f /pid <PID>
```

#### 2. Insufficient Space
**Symptom**: Database write failures
**Solution**:
```bash
# Check disk space
df -h

# Clean up old data
poetry run python src/data/update/maintenance_scheduler.py cleanup
```

#### 3. Data Corruption
**Symptom**: Database read errors
**Solution**:
```bash
# Check database integrity
poetry run python -c "
import duckdb
conn = duckdb.connect('data/options_chain_data.duckdb')
print('Database integrity check passed')
"
```

---

## 📋 Data Backup

### Automatic Backup
**Scheduler**: `src/data/update/maintenance_scheduler.py`  
**Location**: `data/sqlite_backup/`  
**Frequency**: Daily  

### Manual Backup
```bash
# Create backup
cp data/options_chain_data.duckdb data/sqlite_backup/options_chain_data_$(date +%Y%m%d).duckdb

# Restore from backup
cp data/sqlite_backup/options_chain_data_20250825.duckdb data/options_chain_data.duckdb
```

---

## 🔐 Security

### Access Control
- **File permissions**: Read/write for application user only
- **Database connections**: Local access only
- **Backup encryption**: Optional encryption for sensitive data

### Data Privacy
- **No PII**: No personally identifiable information stored
- **Market data only**: Only public market data collected
- **Compliance**: Follows financial data regulations

---

## 📞 Support

### Getting Help
1. **Check logs**: Review application logs for errors
2. **Verify paths**: Ensure database paths are correct
3. **Test connections**: Use provided test scripts
4. **Documentation**: Refer to main documentation

### Useful Commands
```bash
# Check database status
poetry run python -c "import duckdb; print('Database connection test')"

# View recent data
poetry run python -c "
import duckdb
conn = duckdb.connect('data/options_chain_data.duckdb')
print(conn.execute('SELECT * FROM options_chain_data ORDER BY timestamp DESC LIMIT 5').fetchall())
"

# Monitor data collection
tail -f logs/options_collector.log
```

---

## 📝 Version History

### v1.0.0 (Current)
- ✅ **Separate options database**
- ✅ **Real-time data collection**
- ✅ **Database optimization**
- ✅ **Maintenance automation**
- ✅ **Data quality monitoring**

### Planned Features
- 🔄 **Data compression**
- 🔄 **Advanced analytics**
- 🔄 **Real-time dashboards**
- 🔄 **Multi-exchange support**
- 🔄 **Cloud backup integration**

---

*Last Updated: August 25, 2025*  
*Version: 1.0.0*
