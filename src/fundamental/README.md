# Fundamental Data Collection Package

This package provides organized collection of fundamental data from NSE (National Stock Exchange) for all listed companies.

## 📁 Package Structure

```
src/fundamental/
├── __init__.py                    # Package initialization
├── README.md                      # This file
├── start_fundamental_collection.py # Main entry point script
├── fundamental_data_collection.md  # Detailed documentation
├── collectors/                    # Data collection modules
│   ├── __init__.py
│   └── nse_fundamental_collector.py # NSE fundamental data collector
├── schedulers/                    # Automated scheduling modules
│   ├── __init__.py
│   └── fundamental_scheduler.py   # Automated scheduler
└── utils/                         # Utility functions
    └── __init__.py
```

## 🚀 Quick Start

### One-time Download
```bash
poetry run python src/fundamental/start_fundamental_collection.py
```

### Start Automated Scheduler
```bash
poetry run python src/fundamental/start_fundamental_collection.py
# Choose option 2: Start scheduler
```

## 📊 Features

- **Comprehensive Data Collection**: Quarterly, half-yearly, and annual financial reports
- **Corporate Filings**: Dividend, bonus, split announcements and other corporate actions
- **Batch Processing**: Efficient processing of 2000+ companies
- **Progress Tracking**: Real-time progress monitoring and resume capability
- **Error Handling**: Robust error handling with retry mechanisms
- **Rate Limiting**: Respectful API usage to avoid blocking
- **Database Storage**: Structured storage in DuckDB with proper indexing

## 🔧 Configuration

### Batch Size
- Default: 50 symbols per batch
- Adjustable for testing (recommend 20 for testing)

### Rate Limiting
- 2 seconds delay between requests
- 3 concurrent workers
- 30-second session timeout

### Database
- Primary table: `fundamental_data`
- Composite primary key: (symbol, report_date, period_type)
- JSON storage for raw data

## 📈 Data Schema

The `fundamental_data` table includes:

### Financial Metrics
- Revenue, Net Profit, Operating Profit, EBITDA
- Total Assets, Liabilities, Equity
- EPS, PE Ratio, PB Ratio, ROE, ROA, Debt-to-Equity

### Market Data
- Market Cap, Face Value, Book Value

### Metadata
- Symbol, Report Date, Period Type, Filing Date
- Source URL, Raw Data (JSON), Timestamps

## 🔄 Scheduling

### Weekly Full Download
- **When**: Every Monday at 6:00 AM
- **What**: Complete refresh of all fundamental data

### Daily Incremental Update
- **When**: Every day at 8:00 PM
- **What**: New filings and updates only

### Health Check
- **When**: Every day at 10:00 AM
- **What**: Verify data integrity and system health

## 🛠️ Troubleshooting

### Common Issues

1. **404 Errors**: NSE API endpoints may change
   - Solution: Update API URLs in collector

2. **Rate Limiting**: Too many requests
   - Solution: Increase delays between requests

3. **Database Locks**: Concurrent access issues
   - Solution: Use smaller batch sizes

4. **Connection Errors**: Network issues
   - Solution: Check internet connection and retry

### Progress Recovery

The system automatically saves progress to `data/fundamental_progress.json`. If interrupted, restart the script and it will resume from where it left off.

## 📝 Logging

All operations are logged with detailed information:
- Progress updates every 50 symbols
- Error details with stack traces
- Performance metrics
- Data quality statistics

## 🔗 Integration

This package integrates with:
- **Database**: `src/data/database/duckdb_manager.py`
- **NSE Utils**: `src/nsedata/NseUtility.py`
- **UI**: Can be integrated into the main web application

## 📚 Documentation

For detailed documentation, see `fundamental_data_collection.md` in this directory.
