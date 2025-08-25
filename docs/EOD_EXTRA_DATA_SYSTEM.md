# EOD Extra Data System Documentation

## 📊 **Overview**

The EOD Extra Data System is a comprehensive solution for downloading and managing additional End-of-Day (EOD) data from NSE using NSE Utility methods. This system provides four types of EOD data with automated scheduling and database management.

---

## 🗄️ **Database Tables**

### **1. FNO Bhav Copy (`fno_bhav_copy`)**
**Purpose**: Stores Futures and Options bhav copy data from NSE.

**Schema**:
```sql
CREATE TABLE fno_bhav_copy (
    SYMBOL VARCHAR,
    EXPIRY_DT VARCHAR,
    STRIKE_PRICE DOUBLE,
    OPTION_TYP VARCHAR,
    OPEN DOUBLE,
    HIGH DOUBLE,
    LOW DOUBLE,
    CLOSE DOUBLE,
    SETTLE_PR DOUBLE,
    CONTRACTS BIGINT,
    VAL_INLAKH DOUBLE,
    OPEN_INT BIGINT,
    CHG_IN_OI BIGINT,
    TRADE_DATE DATE,
    last_updated TIMESTAMP,
    PRIMARY KEY (SYMBOL, EXPIRY_DT, STRIKE_PRICE, OPTION_TYP, TRADE_DATE)
);
```

**Data Source**: `nse.fno_bhav_copy(trade_date)`
**Usage**: Options and futures analysis, volatility calculations

---

### **2. Equity Bhav Copy with Delivery (`equity_bhav_copy_delivery`)**
**Purpose**: Stores equity bhav copy data with delivery information.

**Schema**:
```sql
CREATE TABLE equity_bhav_copy_delivery (
    SYMBOL VARCHAR,
    SERIES VARCHAR,
    DATE1 VARCHAR,
    PREV_CLOSE DOUBLE,
    OPEN_PRICE DOUBLE,
    HIGH_PRICE DOUBLE,
    LOW_PRICE DOUBLE,
    LAST_PRICE DOUBLE,
    CLOSE_PRICE DOUBLE,
    AVG_PRICE DOUBLE,
    TTL_TRD_QNTY BIGINT,
    TURNOVER_LACS DOUBLE,
    NO_OF_TRADES BIGINT,
    DELIV_QTY BIGINT,
    DELIV_PER DOUBLE,
    TRADE_DATE DATE,
    last_updated TIMESTAMP,
    PRIMARY KEY (SYMBOL, SERIES, TRADE_DATE)
);
```

**Data Source**: `nse.bhav_copy_with_delivery(trade_date)`
**Usage**: Delivery analysis, volume analysis, trading patterns

---

### **3. Bhav Copy Indices (`bhav_copy_indices`)**
**Purpose**: Stores index bhav copy data.

**Schema**:
```sql
CREATE TABLE bhav_copy_indices (
    Index_Name VARCHAR,
    Index_Date VARCHAR,
    Open_Index_Value DOUBLE,
    High_Index_Value DOUBLE,
    Low_Index_Value DOUBLE,
    Closing_Index_Value DOUBLE,
    Points_Change DOUBLE,
    Change_Percent DOUBLE,
    Volume DOUBLE,
    Turnover_Crs DOUBLE,
    P/E DOUBLE,
    P/B DOUBLE,
    Div_Yield DOUBLE,
    TRADE_DATE DATE,
    last_updated TIMESTAMP,
    PRIMARY KEY (Index_Name, TRADE_DATE)
);
```

**Data Source**: `nse.bhav_copy_indices(trade_date)`
**Usage**: Index analysis, market sentiment, sector performance

---

### **4. FII DII Activity (`fii_dii_activity`)**
**Purpose**: Stores Foreign Institutional Investors and Domestic Institutional Investors activity data.

**Schema**:
```sql
CREATE TABLE fii_dii_activity (
    category VARCHAR,
    buyValue DOUBLE,
    buyQuantity BIGINT,
    sellValue DOUBLE,
    sellQuantity BIGINT,
    netValue DOUBLE,
    netQuantity BIGINT,
    activity_date DATE,
    last_updated TIMESTAMP,
    PRIMARY KEY (category, activity_date)
);
```

**Data Source**: `nse.fii_dii_activity()`
**Usage**: Institutional flow analysis, market sentiment, capital flow tracking

---

## 🔧 **Core Components**

### **1. EOD Extra Data Downloader (`src/data/downloaders/eod_extra_data_downloader.py`)**

**Purpose**: Primary downloader for EOD extra data with bulk and incremental capabilities.

**Key Features**:
- ✅ **Bulk Historical Downloads**: Download 5+ years of data
- ✅ **Incremental Updates**: Daily updates for latest data
- ✅ **Progress Tracking**: JSON-based progress persistence
- ✅ **Error Handling**: Retry logic and failure logging
- ✅ **Database Management**: Automatic table creation and indexing
- ✅ **Weekend Skipping**: Automatically skips non-trading days

**Usage**:
```python
from src.data.downloaders.eod_extra_data_downloader import EODExtraDataDownloader

# Initialize downloader
downloader = EODExtraDataDownloader()

# Download 5 years of data
results = downloader.download_all_eod_data(years=5)

# Download specific date range
fno_result = downloader.download_fno_bhav_copy("2024-01-01", "2024-01-31")

# Get database statistics
stats = downloader.get_database_stats()
```

**Command Line Usage**:
```bash
# Download 5 years of data (default)
poetry run python src/data/downloaders/eod_extra_data_downloader.py

# Download specific number of years
poetry run python src/data/downloaders/eod_extra_data_downloader.py 3
```

---

### **2. Enhanced Daily Updater (`src/data/update/daily_updater.py`)**

**Purpose**: Integrated daily updater that includes EOD extra data updates.

**Key Features**:
- ✅ **Integrated Updates**: Combines ticker updates with EOD extra data
- ✅ **Configuration Driven**: JSON-based configuration
- ✅ **Quality Monitoring**: Data quality checks and reporting
- ✅ **Error Recovery**: Retry logic and failure handling

**Configuration** (`config/tickers.json`):
```json
{
    "indian_stocks": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"],
    "us_stocks": ["AAPL", "MSFT", "GOOGL"],
    "data_types": ["technical", "fundamental"],
    "eod_extra_data": {
        "enabled": true,
        "types": ["fno_bhav_copy", "equity_bhav_copy_delivery", "bhav_copy_indices", "fii_dii_activity"]
    }
}
```

**Usage**:
```python
from src.data.update.daily_updater import DailyDataUpdater

# Initialize updater
updater = DailyDataUpdater()

# Run daily update for yesterday
result = await updater.run_daily_update("2024-01-15")

# Get update status
status = updater.get_update_status()
```

---

### **3. Maintenance Scheduler (`src/data/update/maintenance_scheduler.py`)**

**Purpose**: Automated scheduling system for all data updates and maintenance tasks.

**Key Features**:
- ✅ **Automated Scheduling**: 6 AM daily updates as requested
- ✅ **Multiple Job Types**: Daily, weekly, monthly maintenance
- ✅ **Failure Handling**: Retry logic and failure logging
- ✅ **Performance Optimization**: Database optimization and cleanup
- ✅ **Reporting**: Weekly and monthly reports

**Configuration** (`config/scheduler_config.json`):
```json
{
    "daily_updates": {
        "enabled": true,
        "time": "06:00",
        "timezone": "Asia/Kolkata",
        "include_eod_extra_data": true
    },
    "eod_extra_data": {
        "enabled": true,
        "time": "06:00",
        "timezone": "Asia/Kolkata",
        "data_types": ["fno_bhav_copy", "equity_bhav_copy_delivery", "bhav_copy_indices", "fii_dii_activity"]
    }
}
```

**Usage**:
```python
from src.data.update.maintenance_scheduler import MaintenanceScheduler

# Initialize scheduler
scheduler = MaintenanceScheduler()

# Start scheduler (runs continuously)
scheduler.start()

# Run specific job once
scheduler.run_once("eod_extra_data")
scheduler.run_once("daily")
```

**Command Line Usage**:
```bash
# Start the scheduler
poetry run python src/data/update/maintenance_scheduler.py
```

---

## 📅 **Scheduling System**

### **Daily Schedule (6 AM IST)**
- **06:00 AM**: Daily ticker updates + EOD extra data updates
- **06:00 AM**: EOD extra data updates (separate job for redundancy)

### **Weekly Schedule (Sunday 2 AM IST)**
- **02:00 AM**: Weekly maintenance
  - Database optimization (VACUUM, ANALYZE)
  - Log cleanup (30-day retention)
  - Weekly performance reports

### **Monthly Schedule (1st of month, 3 AM IST)**
- **03:00 AM**: Monthly cleanup
  - Archive old data (2+ years)
  - Update monthly statistics
  - Temporary file cleanup

---

## 🚀 **Quick Start Guide**

### **1. Initial Setup**
```bash
# Install dependencies
poetry install

# Create configuration directories
mkdir -p config logs reports temp
```

### **2. Download Historical Data (5 Years)**
```bash
# Download all EOD extra data for 5 years
poetry run python src/data/downloaders/eod_extra_data_downloader.py 5
```

### **3. Test the System**
```bash
# Run comprehensive tests
poetry run python test_eod_extra_data.py
```

### **4. Start Automated Scheduling**
```bash
# Start the maintenance scheduler
poetry run python src/data/update/maintenance_scheduler.py
```

### **5. Manual Daily Update**
```bash
# Run daily update manually
poetry run python -c "
import asyncio
from src.data.update.daily_updater import DailyDataUpdater
updater = DailyDataUpdater()
result = asyncio.run(updater.run_daily_update())
print(result)
"
```

---

## 📊 **Data Analysis Examples**

### **FNO Analysis**
```sql
-- Get PCR (Put-Call Ratio) for NIFTY
SELECT 
    TRADE_DATE,
    SUM(CASE WHEN OPTION_TYP = 'PE' THEN OPEN_INT ELSE 0 END) as put_oi,
    SUM(CASE WHEN OPTION_TYP = 'CE' THEN OPEN_INT ELSE 0 END) as call_oi,
    SUM(CASE WHEN OPTION_TYP = 'PE' THEN OPEN_INT ELSE 0 END) / 
    SUM(CASE WHEN OPTION_TYP = 'CE' THEN OPEN_INT ELSE 0 END) as pcr
FROM fno_bhav_copy 
WHERE SYMBOL = 'NIFTY' 
GROUP BY TRADE_DATE 
ORDER BY TRADE_DATE DESC;
```

### **Delivery Analysis**
```sql
-- Get high delivery stocks
SELECT 
    SYMBOL,
    TRADE_DATE,
    DELIV_PER,
    TTL_TRD_QNTY,
    DELIV_QTY
FROM equity_bhav_copy_delivery 
WHERE DELIV_PER > 50 
ORDER BY DELIV_PER DESC;
```

### **FII DII Flow Analysis**
```sql
-- Get institutional flow trends
SELECT 
    category,
    activity_date,
    netValue,
    netQuantity
FROM fii_dii_activity 
ORDER BY activity_date DESC;
```

---

## 🔍 **Monitoring and Maintenance**

### **Database Statistics**
```python
from src.data.downloaders.eod_extra_data_downloader import EODExtraDataDownloader

downloader = EODExtraDataDownloader()
stats = downloader.get_database_stats()

for data_type, data_stats in stats.items():
    print(f"{data_type}: {data_stats['total_records']:,} records")
```

### **Progress Tracking**
```python
# Show download progress
downloader.show_progress()

# Progress is saved in: eod_extra_download_progress.json
```

### **Log Files**
- **Main Log**: `eod_extra_data_download.log`
- **Scheduler Log**: `logs/maintenance.log`
- **Failure Logs**: `logs/failures/`

---

## ⚠️ **Important Notes**

### **Data Availability**
- **FNO Data**: Available for trading days only
- **Equity Data**: Available for trading days only
- **Indices Data**: Available for trading days only
- **FII DII Data**: Available daily (including weekends)

### **Rate Limiting**
- **Delay Between Requests**: 0.5 seconds (configurable)
- **Retry Attempts**: 3 attempts per download
- **Weekend Skipping**: Automatically skips non-trading days

### **Database Performance**
- **Indexes**: Created automatically for better query performance
- **Primary Keys**: Enforce data integrity
- **Optimization**: Weekly VACUUM and ANALYZE operations

### **Error Handling**
- **Network Errors**: Automatic retry with exponential backoff
- **Data Errors**: Logged and skipped, download continues
- **Database Errors**: Transaction rollback and retry

---

## 🔧 **Configuration Options**

### **Downloader Configuration**
```python
# In EODExtraDataDownloader.__init__()
self.max_workers = 5          # Concurrent downloads
self.retry_attempts = 3       # Retry attempts
self.delay_between_requests = 0.5  # Delay in seconds
```

### **Scheduler Configuration**
```json
{
    "daily_updates": {
        "time": "06:00",
        "retry_attempts": 3,
        "retry_delay_minutes": 30
    },
    "eod_extra_data": {
        "time": "06:00",
        "retry_attempts": 3,
        "retry_delay_minutes": 15
    }
}
```

---

## 📈 **Performance Metrics**

### **Expected Performance**
- **Historical Download (5 years)**: 2-4 hours
- **Daily Update**: 10-15 minutes
- **Database Size**: ~2-5 GB for 5 years of data
- **Memory Usage**: ~500MB during downloads

### **Monitoring Metrics**
- **Download Success Rate**: >95%
- **Data Completeness**: >98%
- **Update Frequency**: Daily at 6 AM IST
- **Data Freshness**: <24 hours behind

---

## 🆘 **Troubleshooting**

### **Common Issues**

1. **Network Timeout**
   ```bash
   # Increase timeout in NSE Utility
   # Check internet connection
   # Verify NSE website accessibility
   ```

2. **Database Lock**
   ```bash
   # Check for other processes using the database
   # Restart the application
   # Check disk space
   ```

3. **Data Missing**
   ```bash
   # Check trading holidays
   # Verify date format (DD-MM-YYYY)
   # Check NSE data availability
   ```

### **Debug Mode**
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
downloader = EODExtraDataDownloader()
downloader.delay_between_requests = 1.0  # Slower for debugging
```

---

## 📞 **Support**

For issues or questions:
1. Check the log files for error details
2. Verify configuration files
3. Test with smaller date ranges
4. Check NSE website for data availability

---

**Last Updated**: January 2024
**Version**: 1.0.0
**Author**: AI Hedge Fund Team
