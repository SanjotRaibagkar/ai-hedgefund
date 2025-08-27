# Fundamental Data Collection System

## Overview

The Fundamental Data Collection System is designed to download and store comprehensive fundamental data from NSE (National Stock Exchange) for all listed companies. This system collects corporate filings, financial results, and other fundamental information to support advanced financial analysis and screening.

## 🎯 Features

### **Data Sources**
- **NSE Corporate Filings**: All corporate announcements and filings
- **Financial Results**: Quarterly, Half-Yearly, and Annual financial statements
- **Market Data**: Market capitalization, face value, book value
- **Financial Ratios**: PE, PB, ROE, ROA, Debt-to-Equity ratios

### **Data Types Collected**
- **Quarterly Reports**: Q1, Q2, Q3, Q4 financial results
- **Half-Yearly Reports**: H1, H2 financial statements
- **Annual Reports**: Full-year financial performance
- **Corporate Actions**: Dividends, bonuses, splits, mergers

### **Technical Features**
- **Batch Processing**: Efficient processing of large datasets
- **Progress Tracking**: Real-time progress monitoring
- **Error Handling**: Robust error recovery and retry mechanisms
- **Rate Limiting**: Respectful API usage to avoid blocking
- **Database Storage**: DuckDB for fast analytical queries

## 🏗️ Architecture

### **System Components**

```
Fundamental Data Collection System
├── NSEFundamentalCollector     # Main data collection engine
├── FundamentalScheduler        # Automated scheduling system
├── Database Manager           # DuckDB storage and retrieval
└── Progress Tracking          # JSON-based progress persistence
```

### **Data Flow**

```
NSE APIs → Data Collection → Processing → DuckDB Storage → Analysis
    ↓           ↓              ↓            ↓            ↓
Corporate   Financial    Data Cleaning   Structured   Fundamental
Filings     Results      & Validation    Storage      Analysis
```

## 📊 Database Schema

### **fundamental_data Table**

```sql
CREATE TABLE fundamental_data (
    symbol VARCHAR,                    -- Stock symbol (e.g., 'RELIANCE')
    report_date DATE,                  -- Financial period end date
    period_type VARCHAR,               -- 'quarterly', 'half_yearly', 'annual'
    filing_date DATE,                  -- Date when filing was submitted
    filing_type VARCHAR,               -- 'financial_results', 'corporate_filings'
    
    -- Financial Metrics
    revenue DOUBLE,                    -- Total revenue
    net_profit DOUBLE,                 -- Net profit/loss
    total_assets DOUBLE,               -- Total assets
    total_liabilities DOUBLE,          -- Total liabilities
    total_equity DOUBLE,               -- Total equity
    operating_profit DOUBLE,           -- Operating profit
    ebitda DOUBLE,                     -- EBITDA
    
    -- Financial Ratios
    eps DOUBLE,                        -- Earnings per share
    pe_ratio DOUBLE,                   -- Price to earnings ratio
    pb_ratio DOUBLE,                   -- Price to book ratio
    roe DOUBLE,                        -- Return on equity
    roa DOUBLE,                        -- Return on assets
    debt_to_equity DOUBLE,             -- Debt to equity ratio
    
    -- Market Data
    market_cap DOUBLE,                 -- Market capitalization
    face_value DOUBLE,                 -- Face value per share
    book_value DOUBLE,                 -- Book value per share
    
    -- Metadata
    source_url VARCHAR,                -- Source URL for the data
    raw_data JSON,                     -- Raw JSON data from NSE
    created_at TIMESTAMP,              -- Record creation timestamp
    updated_at TIMESTAMP,              -- Record update timestamp
    
    PRIMARY KEY (symbol, report_date, period_type)
);
```

## 🚀 Usage

### **Quick Start**

1. **One-time Download**:
   ```bash
   poetry run python start_fundamental_collection.py
   # Choose option 1 for one-time download
   ```

2. **Start Scheduler**:
   ```bash
   poetry run python start_fundamental_collection.py
   # Choose option 2 for automated scheduler
   ```

3. **Check Status**:
   ```bash
   poetry run python start_fundamental_collection.py
   # Choose option 3 to check current status
   ```

### **Programmatic Usage**

```python
from src.data.collectors.nse_fundamental_collector import NSEFundamentalCollector

# Initialize collector
collector = NSEFundamentalCollector()

# Download all fundamental data
collector.download_all_fundamentals(batch_size=50)

# Check status
status = collector.get_download_status()
print(f"Progress: {status['progress_percentage']:.1f}%")
```

## ⏰ Scheduling

### **Automated Schedule**

- **Full Download**: Every Monday at 6:00 AM
- **Incremental Update**: Daily at 8:00 PM
- **Health Check**: Daily at 10:00 AM

### **Manual Scheduling**

```python
from src.data.collectors.fundamental_scheduler import FundamentalScheduler

scheduler = FundamentalScheduler()
scheduler.run_scheduler()
```

## 📈 Data Analysis Examples

### **Query Recent Financial Results**

```sql
-- Get latest quarterly results for top companies
SELECT 
    symbol,
    report_date,
    revenue,
    net_profit,
    eps,
    pe_ratio
FROM fundamental_data 
WHERE period_type = 'quarterly'
  AND report_date >= '2024-01-01'
ORDER BY report_date DESC, revenue DESC
LIMIT 20;
```

### **Calculate Financial Ratios**

```sql
-- Calculate average PE ratio by sector
SELECT 
    symbol,
    AVG(pe_ratio) as avg_pe_ratio,
    AVG(pb_ratio) as avg_pb_ratio,
    AVG(roe) as avg_roe
FROM fundamental_data 
WHERE period_type = 'annual'
  AND report_date >= '2023-01-01'
GROUP BY symbol
HAVING avg_pe_ratio > 0
ORDER BY avg_pe_ratio;
```

### **Track Revenue Growth**

```sql
-- Calculate year-over-year revenue growth
WITH revenue_data AS (
    SELECT 
        symbol,
        report_date,
        revenue,
        LAG(revenue) OVER (PARTITION BY symbol ORDER BY report_date) as prev_revenue
    FROM fundamental_data 
    WHERE period_type = 'annual'
      AND revenue IS NOT NULL
)
SELECT 
    symbol,
    report_date,
    revenue,
    ((revenue - prev_revenue) / prev_revenue * 100) as revenue_growth_pct
FROM revenue_data 
WHERE prev_revenue IS NOT NULL
ORDER BY revenue_growth_pct DESC;
```

## 🔧 Configuration

### **Performance Settings**

```python
# In NSEFundamentalCollector.__init__()
self.max_workers = 3              # Concurrent requests
self.delay_between_requests = 2.0  # Seconds between requests
self.retry_attempts = 3           # Retry attempts on failure
self.session_timeout = 30         # Request timeout in seconds
```

### **Batch Processing**

```python
# Adjust batch size based on system capacity
collector.download_all_fundamentals(batch_size=50)  # Default
collector.download_all_fundamentals(batch_size=20)  # Conservative
collector.download_all_fundamentals(batch_size=100) # Aggressive
```

## 📁 File Structure

```
src/data/collectors/
├── nse_fundamental_collector.py    # Main collection engine
├── fundamental_scheduler.py        # Automated scheduling
└── ...

data/
├── fundamental_data/               # Raw data storage
├── fundamental_progress.json       # Progress tracking
└── comprehensive_equity.duckdb     # Main database

start_fundamental_collection.py     # Startup script
```

## 🛠️ Troubleshooting

### **Common Issues**

1. **Rate Limiting**:
   - Reduce `max_workers` and increase `delay_between_requests`
   - Check NSE website for any temporary restrictions

2. **Database Errors**:
   - Ensure DuckDB is properly installed
   - Check database file permissions
   - Verify table schema creation

3. **Network Issues**:
   - Check internet connectivity
   - Verify NSE website accessibility
   - Increase `session_timeout` if needed

### **Progress Recovery**

The system automatically saves progress to `data/fundamental_progress.json`. If interrupted, restart the download and it will resume from where it left off.

### **Data Validation**

```python
# Validate data quality
with collector.db_manager.connection as conn:
    # Check for missing data
    missing_data = conn.execute("""
        SELECT symbol, COUNT(*) as record_count
        FROM fundamental_data 
        GROUP BY symbol
        HAVING record_count < 4  -- Should have at least 4 quarters
    """).fetchdf()
    
    print(f"Symbols with incomplete data: {len(missing_data)}")
```

## 📊 Monitoring

### **Progress Tracking**

```python
status = collector.get_download_status()
print(f"Total Symbols: {status['total_symbols']}")
print(f"Completed: {status['completed_symbols']}")
print(f"Failed: {status['failed_symbols']}")
print(f"Progress: {status['progress_percentage']:.1f}%")
```

### **Health Checks**

The scheduler automatically performs health checks and alerts on:
- High failure rates (>10% of symbols)
- Low completion rates (<50% progress)
- Missing recent updates

## 🔄 Integration

### **With Screening System**

```python
# Use fundamental data in stock screening
from src.screening.unified_eod_screener import UnifiedEODScreener

screener = UnifiedEODScreener()

# Add fundamental filters
def fundamental_filter(symbol):
    with db_manager.connection as conn:
        latest_data = conn.execute("""
            SELECT pe_ratio, pb_ratio, roe, debt_to_equity
            FROM fundamental_data 
            WHERE symbol = ? 
            ORDER BY report_date DESC 
            LIMIT 1
        """, [symbol]).fetchdf()
        
        if not latest_data.empty:
            row = latest_data.iloc[0]
            return (row['pe_ratio'] < 25 and 
                   row['pb_ratio'] < 3 and 
                   row['roe'] > 15 and 
                   row['debt_to_equity'] < 1)
        return False
```

### **With Analysis Tools**

```python
# Export data for external analysis
import pandas as pd

with collector.db_manager.connection as conn:
    df = conn.execute("""
        SELECT * FROM fundamental_data 
        WHERE period_type = 'annual'
        ORDER BY report_date DESC
    """).fetchdf()

# Export to CSV
df.to_csv('fundamental_data_export.csv', index=False)
```

## 📝 Best Practices

1. **Start Small**: Begin with a small batch size and increase gradually
2. **Monitor Progress**: Regularly check progress and error logs
3. **Respect Rate Limits**: Don't overwhelm NSE servers
4. **Backup Data**: Regularly backup the DuckDB database
5. **Validate Data**: Check data quality after collection
6. **Update Regularly**: Run incremental updates to stay current

## 🎯 Future Enhancements

- **Real-time Updates**: WebSocket-based real-time data streaming
- **Advanced Analytics**: Built-in financial ratio calculations
- **Data Visualization**: Interactive dashboards for data exploration
- **API Integration**: REST API for external access
- **Machine Learning**: Predictive models using fundamental data
- **Multi-exchange Support**: Extend to BSE and other exchanges

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the logs in the terminal output
3. Check the progress file for detailed status
4. Verify NSE website accessibility

---

**Note**: This system is designed for educational and research purposes. Always comply with NSE's terms of service and data usage policies.
