# Options Chain Collector V2 Package

This package provides organized collection of options chain data from NSE (National Stock Exchange) for all listed companies.

## 📁 Package Structure

```
src/data/collectors/optionchaincollectorv2/
├── __init__.py                           # Package initialization
├── README.md                             # This file
├── start_options_v2.py                   # Main entry point script
├── options_chain_collector_v2.py         # Main options data collector
├── options_threaded_manager.py           # Threaded manager for collection
├── options_process_manager.py            # Legacy process manager
├── options_scheduler.py                  # Legacy scheduler
├── schedulers/                           # Automated scheduling modules
│   └── __init__.py
└── utils/                                # Utility functions
    └── __init__.py
```

## 🚀 Quick Start

### Start Options Collection
```bash
poetry run python src/data/collectors/optionchaincollectorv2/start_options_v2.py
```

## 📊 Features

- **Two-Thread Architecture**: 
  - Thread 1: Collect 1-minute options data → Parquet files
  - Thread 2: Batch process parquet files → DuckDB every 5 minutes
- **Reduced Database Locks**: Uses parquet files as intermediate storage
- **Better Error Handling**: Robust error recovery and retry mechanisms
- **Rate Limiting**: Respectful API usage to avoid blocking
- **Database Storage**: DuckDB for fast analytical queries

## 🔧 Configuration

### Collection Settings
- **Collection Interval**: 1 minute
- **Batch Processing**: Every 5 minutes
- **Storage**: Parquet files → DuckDB
- **Architecture**: Threaded (no multiprocessing issues)

### Data Storage
- **Raw Parquet**: `data/options_parquet/{YYYYMMDD}/`
- **Processed**: `data/options_processed/`
- **DuckDB**: `options_chain_data.duckdb`

## 📈 Data Flow

```
NSE APIs → Collection Thread → Parquet Files → Batch Thread → DuckDB
    ↓           ↓                ↓              ↓           ↓
Live Data   Real-time      Intermediate    Processing   Analysis
Collection  Updates        Storage         & Cleaning   Ready
```

## 🔄 Threading Architecture

### Collection Thread
- Fetches live options chain data every minute
- Saves to daily parquet files
- Handles rate limiting and errors

### Batch Thread
- Processes parquet files every 5 minutes
- Inserts data into DuckDB database
- Handles data cleaning and validation

## 🛠️ Troubleshooting

### Common Issues

1. **Database Locks**: Resolved with parquet intermediate storage
2. **Pickling Errors**: Resolved with threading instead of multiprocessing
3. **Rate Limiting**: Built-in delays and error handling
4. **File Locks**: Separate threads for collection and processing

### Monitoring

- Check terminal for real-time logs
- Monitor parquet file creation in `data/options_parquet/`
- Check DuckDB database for processed data
- Use `Ctrl+C` to stop all threads gracefully

## 🔗 Integration

This package integrates with:
- **Database**: `src/data/database/duckdb_manager.py`
- **NSE Utils**: `src/nsedata/NseUtility.py`
- **UI**: Can be integrated into the main web application

## 📝 Logging

All operations are logged with detailed information:
- Collection progress and errors
- Batch processing statistics
- Performance metrics
- Data quality information

## 🎯 Usage Examples

### Start Collection System
```python
from src.data.collectors.optionchaincollectorv2 import OptionsThreadedManager

manager = OptionsThreadedManager()
manager.start_all_threads()
```

### Check Status
```python
status = manager.get_thread_status()
print(f"Collection: {status['collection_thread']['alive']}")
print(f"Batch: {status['batch_thread']['alive']}")
```

### Stop System
```python
manager.stop_all_threads()
```
