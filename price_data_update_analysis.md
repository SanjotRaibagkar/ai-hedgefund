# Price Data Update Analysis - `comprehensive_equity.duckdb`

## 📊 **Overview**

This document analyzes which files and components are responsible for updating the `price_data` table in the `comprehensive_equity.duckdb` database with the latest market data.

---

## 🗄️ **Database Schema**

### **Main Table: `price_data`**
```sql
CREATE TABLE price_data (
    symbol VARCHAR,
    date DATE,
    open_price DOUBLE,
    high_price DOUBLE,
    low_price DOUBLE,
    close_price DOUBLE,
    volume BIGINT,
    turnover DOUBLE,
    last_updated TIMESTAMP,
    PRIMARY KEY (symbol, date)
);
```

---

## 🔄 **Primary Update Files**

### **1. Optimized Equity Data Downloader (`src/data/downloaders/optimized_equity_downloader.py`)**

#### **Purpose:**
- **Primary bulk data downloader** for historical price data
- Downloads data for multiple companies in batches
- Uses DuckDB for fast data storage and processing

#### **Update Methods:**

##### **A. Batch Data Storage:**
```python
def _store_company_data_batch(self, companies_data: List[Tuple[Dict, List[Dict]]]) -> bool:
    # Batch insert price data
    if price_data:
        self.conn.execute('''
            INSERT OR REPLACE INTO price_data 
            (symbol, date, open_price, high_price, low_price, close_price, 
             volume, turnover, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', price_data)
```

##### **B. Delta Data Updates:**
```python
def _store_symbol_delta_data(self, symbol: str, data: List[Dict]):
    # Remove existing data for the date range
    self.conn.execute(
        "DELETE FROM price_data WHERE symbol = ? AND date BETWEEN ? AND ?",
        [symbol, date_range[0], date_range[1]]
    )
    
    # Insert new data
    self.conn.execute("INSERT INTO price_data SELECT * FROM df")
```

#### **Data Sources:**
- **NSE Utility**: `NseUtils.get_historical_data()`
- **Simulated Data**: For testing and development
- **Date Range**: 2024-01-01 to current date

#### **Usage:**
```bash
# Run bulk download
poetry run python src/data/downloaders/optimized_equity_downloader.py [max_companies]

# Example: Download data for 100 companies
poetry run python src/data/downloaders/optimized_equity_downloader.py 100
```

---

### **2. Enhanced Indian Data Manager (`src/data/enhanced_indian_data_manager.py`)**

#### **Purpose:**
- **Real-time data manager** for Indian stocks
- Handles incremental updates and missing data
- Uses NSE API for live data

#### **Update Methods:**

##### **A. Enhanced Price Data Storage:**
```python
def _store_price_data_enhanced(self, symbol: str, data_points: List[Dict]) -> int:
    for data_point in data_points:
        self.db_manager.connection.execute("""
            INSERT OR REPLACE INTO price_data 
            (symbol, date, open_price, high_price, low_price, close_price, 
             volume, turnover, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol,
            data_point['date'],
            float(data_point['open_price']),
            float(data_point['high_price']),
            float(data_point['low_price']),
            float(data_point['close_price']),
            int(data_point['volume']),
            float(data_point['turnover']),
            datetime.now().isoformat()
        ))
```

##### **B. Latest Data Updates:**
```python
async def update_latest_data(self) -> Dict:
    # Get symbols needing updates
    symbols_to_update = await self._get_symbols_needing_update()
    
    # Update in batches
    for batch in symbols_to_update:
        await self._download_and_store_latest_data(symbol)
```

#### **Data Sources:**
- **NSE API**: Real-time price data
- **NSE Utility**: Historical data retrieval
- **Date Range**: From last stored date to current date

#### **Usage:**
```python
from src.data.enhanced_indian_data_manager import EnhancedIndianDataManager

# Initialize manager
manager = EnhancedIndianDataManager()

# Update latest data for all symbols
await manager.update_latest_data()

# Update specific symbol
await manager.download_and_store_data('RELIANCE.NS')
```

---

### **3. Daily Data Updater (`src/data/update/daily_updater.py`)**

#### **Purpose:**
- **Automated daily updates** for configured tickers
- Handles incremental updates from last stored date
- Integrates with data quality monitoring

#### **Update Methods:**

##### **A. Daily Update Process:**
```python
async def run_daily_update(self, target_date: str = None) -> Dict[str, Any]:
    # Determine target date (default: yesterday)
    if target_date is None:
        target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Get all tickers to update
    all_tickers = (self.tickers_config.get("indian_stocks", []) + 
                  self.tickers_config.get("us_stocks", []))
    
    # Run updates for all tickers
    results = await self._update_all_tickers(all_tickers, target_date)
```

##### **B. Single Ticker Update:**
```python
async def _update_single_ticker(self, ticker: str, target_date: str) -> Dict[str, Any]:
    # Check update requirements
    requirements = await self._check_update_requirements(ticker, target_date)
    
    if not requirements['needs_update']:
        return {"success": True, "ticker": ticker, "skipped": True}
    
    # Collect data
    config = DataCollectionConfig(
        ticker=ticker,
        start_date=requirements['start_date'],
        end_date=target_date,
        data_types=self.tickers_config.get("data_types", ["technical"])
    )
    
    results = await self.collector.collect_data(config)
```

#### **Configuration:**
- **Config File**: `config/tickers.json`
- **Default Tickers**: Top 10 Indian stocks + 5 US stocks
- **Update Frequency**: Daily
- **Data Types**: Technical, Fundamental

#### **Usage:**
```python
from src.data.update.daily_updater import DailyDataUpdater

# Initialize updater
updater = DailyDataUpdater()

# Run daily update
await updater.run_daily_update()

# Update for specific date
await updater.run_daily_update('2024-01-15')
```

---

### **4. Maintenance Scheduler (`src/data/update/maintenance_scheduler.py`)**

#### **Purpose:**
- **Automated scheduling** of data updates
- Handles maintenance tasks and quality checks
- Provides monitoring and notifications

#### **Scheduled Tasks:**

##### **A. Daily Update Task:**
```python
async def _run_daily_update(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Run daily data update task."""
    try:
        target_date = parameters.get('target_date')
        result = await self.daily_updater.run_daily_update(target_date)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

##### **B. Task Scheduling:**
```python
def setup_schedules(self):
    """Setup scheduled tasks."""
    if self.config.get("enabled", True):
        # Daily update at 6:00 AM
        schedule.every().day.at("06:00").do(self._run_scheduled_task, 'daily_update')
        
        # Quality check at 7:00 AM
        schedule.every().day.at("07:00").do(self._run_scheduled_task, 'quality_check')
        
        # Weekly maintenance on Sunday
        schedule.every().sunday.at("08:00").do(self._run_scheduled_task, 'fill_missing_data')
```

#### **Configuration:**
- **Config File**: `config/maintenance.json`
- **Default Schedule**: Daily at 6:00 AM
- **Retry Logic**: Up to 3 retries on failure
- **Notifications**: On failure

#### **Usage:**
```python
from src.data.update.maintenance_scheduler import MaintenanceScheduler

# Initialize scheduler
scheduler = MaintenanceScheduler()

# Start automated scheduling
scheduler.start()

# Run manual task
await scheduler.run_task('daily_update')
```

---

## 🔄 **Update Flow Architecture**

### **1. Manual Bulk Update:**
```
OptimizedEquityDataDownloader
    ↓
NSE Utility API
    ↓
DuckDB (price_data table)
```

### **2. Automated Daily Update:**
```
MaintenanceScheduler
    ↓
DailyDataUpdater
    ↓
AsyncDataCollector
    ↓
EnhancedIndianDataManager
    ↓
DuckDB (price_data table)
```

### **3. Real-time Updates:**
```
EnhancedIndianDataManager
    ↓
NSE API (Real-time)
    ↓
DuckDB (price_data table)
```

---

## 📊 **Data Sources and APIs**

### **Primary Data Sources:**

#### **1. NSE Utility (`src/nsedata/NseUtility.py`)**
- **Purpose**: Historical data retrieval
- **Methods**: `get_historical_data()`, `get_spot_price()`
- **Data Type**: Indian stock data
- **Update Frequency**: On-demand

#### **2. NSE API (Direct)**
- **Purpose**: Real-time price data
- **Methods**: Direct API calls
- **Data Type**: Live Indian stock data
- **Update Frequency**: Real-time

#### **3. Yahoo Finance API**
- **Purpose**: US stock data and fallback
- **Methods**: `yfinance` library
- **Data Type**: US and some Indian data
- **Update Frequency**: On-demand

---

## ⚙️ **Configuration Files**

### **1. Ticker Configuration (`config/tickers.json`)**
```json
{
    "indian_stocks": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
        "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS"
    ],
    "us_stocks": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"
    ],
    "data_types": ["technical", "fundamental"],
    "update_frequency": "daily",
    "max_workers": 5,
    "retry_attempts": 3
}
```

### **2. Maintenance Configuration (`config/maintenance.json`)**
```json
{
    "enabled": true,
    "timezone": "UTC",
    "tasks": {
        "daily_update": {
            "enabled": true,
            "schedule": "daily",
            "time": "06:00",
            "retry_on_failure": true,
            "max_retries": 3
        }
    }
}
```

---

## 🚀 **Usage Examples**

### **1. Manual Bulk Download:**
```bash
# Download data for 50 companies
poetry run python src/data/downloaders/optimized_equity_downloader.py 50

# Download data for all available companies
poetry run python src/data/downloaders/optimized_equity_downloader.py
```

### **2. Manual Daily Update:**
```python
from src.data.update.daily_updater import DailyDataUpdater

# Initialize and run daily update
updater = DailyDataUpdater()
await updater.run_daily_update()
```

### **3. Start Automated Scheduler:**
```python
from src.data.update.maintenance_scheduler import MaintenanceScheduler

# Initialize and start scheduler
scheduler = MaintenanceScheduler()
scheduler.start()

# This will automatically run daily updates at 6:00 AM
```

### **4. Real-time Updates:**
```python
from src.data.enhanced_indian_data_manager import EnhancedIndianDataManager

# Initialize manager
manager = EnhancedIndianDataManager()

# Update latest data for all symbols
await manager.update_latest_data()

# Update specific symbol
await manager.download_and_store_data('RELIANCE.NS')
```

---

## 📈 **Monitoring and Quality**

### **1. Data Quality Monitoring:**
- **Completeness**: Check for missing dates
- **Accuracy**: Validate price ranges
- **Timeliness**: Ensure data is current
- **Consistency**: Check for data anomalies

### **2. Update Tracking:**
- **Download Tracker**: Track successful/failed downloads
- **Progress Tracking**: Monitor bulk download progress
- **Error Logging**: Log and handle update failures
- **Performance Metrics**: Track update performance

### **3. Health Checks:**
- **Database Connectivity**: Verify database access
- **API Availability**: Check data source availability
- **Data Freshness**: Ensure data is not stale
- **Storage Space**: Monitor database size

---

## ⚠️ **Current Limitations**

### **1. Data Availability:**
- **Market Hours**: Some APIs work only during market hours
- **Rate Limits**: API rate limiting may affect updates
- **Data Quality**: Some data sources may have gaps

### **2. Update Frequency:**
- **Real-time**: Limited to market hours
- **Daily**: Automated daily updates at 6:00 AM
- **Manual**: On-demand bulk downloads

### **3. Coverage:**
- **Indian Stocks**: Primary focus with NSE data
- **US Stocks**: Limited coverage via Yahoo Finance
- **Other Markets**: Not currently supported

---

## 🎯 **Summary**

The `price_data` table in `comprehensive_equity.duckdb` is updated by **4 main components**:

1. **OptimizedEquityDataDownloader**: Bulk historical data download
2. **EnhancedIndianDataManager**: Real-time incremental updates
3. **DailyDataUpdater**: Automated daily updates
4. **MaintenanceScheduler**: Automated scheduling and monitoring

**Primary Data Sources**: NSE Utility API, NSE Direct API, Yahoo Finance
**Update Frequency**: Daily automated + manual bulk + real-time incremental
**Coverage**: Indian stocks (primary) + US stocks (limited)

The system provides comprehensive data update capabilities with automated scheduling, quality monitoring, and multiple update methods to ensure the `price_data` table remains current and complete.
