# AI Hedge Fund - Schedulers Documentation

## Overview

This document provides comprehensive information about all schedulers in the AI Hedge Fund system, their purposes, configurations, and usage instructions.

## 📋 Scheduler Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI HEDGE FUND SCHEDULERS                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🕐 OPTIONS ANALYSIS SCHEDULERS                             │
│  ├── Auto Market Hours Scheduler (Auto-Start)              │
│  ├── Enhanced Options Scheduler (Manual)                   │
│  └── Fixed Options Tracker (Legacy)                        │
│                                                             │
│  📊 DATA COLLECTION SCHEDULERS                              │
│  ├── Options Data Collection Scheduler                     │
│  ├── Auto Options Collector                                 │
│  └── Maintenance Scheduler                                  │
│                                                             │
│  🔧 SETUP & CONFIGURATION                                   │
│  ├── Auto-Start Setup Scripts                               │
│  └── Manual Control Scripts                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🕐 Options Analysis Schedulers

### 1. Auto Market Hours Scheduler (Primary)

**File**: `junk/auto_market_hours_scheduler.py`  
**Purpose**: Main auto-start scheduler for options analysis  
**Auto-Start**: Configured via Windows Task Scheduler  

#### Features
- ✅ **Auto-start at system startup**
- ✅ **Market hours respect**: 9:30 AM - 3:30 PM IST
- ✅ **Trading day detection**: Skips weekends and holidays
- ✅ **Analysis interval**: Every 15 minutes during market hours
- ✅ **Auto-stop**: Stops at 3:30 PM IST
- ✅ **Holiday detection**: Uses `is_nse_trading_holiday()` method

#### Configuration
```python
# Market hours (IST)
market_open = "09:30"
market_close = "15:30"

# Analysis interval
analysis_interval = 15  # minutes

# Trading day detection
- Weekend check: Saturday/Sunday
- Holiday check: NSE trading holidays
```

#### Auto-Start Setup
```powershell
# Run as Administrator
.\setup_auto_start.ps1
```

#### Manual Usage
```bash
# Auto mode (respects market hours)
poetry run python junk/auto_market_hours_scheduler.py --mode auto

# Continuous mode (ignores market hours)
poetry run python junk/auto_market_hours_scheduler.py --mode continuous
```

#### Output
- **CSV File**: `results/options_tracker/option_tracker.csv`
- **Logs**: `logs/auto_options_scheduler.log`
- **Analysis**: Real-time options analysis results

---

### 2. Enhanced Options Scheduler (Manual Control)

**File**: `junk/run_options_scheduler.py`  
**Purpose**: Manual/background options analysis with enhanced features  

#### Features
- ✅ **Enhanced market hours checking**
- ✅ **Trading holiday detection**
- ✅ **Auto-stop at market close**
- ✅ **Manual control and monitoring**
- ✅ **Same analysis logic as auto scheduler**

#### Usage
```bash
# Start enhanced scheduler
poetry run python junk/run_options_scheduler.py
```

#### Behavior
- **Outside market hours**: Skips analysis, logs status
- **Market hours**: Runs analysis every 15 minutes
- **Market close**: Automatically stops scheduler
- **Trading holidays**: Skips analysis

---

### 3. Fixed Options Tracker (Legacy)

**File**: `junk/fixed_options_tracker.py`  
**Purpose**: Legacy options tracker with basic scheduling  

#### Features
- ✅ **Basic 15-minute interval scheduling**
- ✅ **Fixed enhanced options analyzer integration**
- ✅ **CSV output generation**

#### Usage
```bash
poetry run python junk/fixed_options_tracker.py
```

---

## 📊 Data Collection Schedulers

### 1. Options Data Collection Scheduler

**File**: `src/data/downloaders/options_scheduler.py`  
**Purpose**: Collect raw options chain data (not analysis)  

#### Features
- ✅ **1-minute interval data collection**
- ✅ **Market hours respect**: 9:30 AM - 3:30 PM IST
- ✅ **Database storage**: `data/options_chain_data.duckdb`
- ✅ **Service management**: Start/stop/restart/status

#### Database Schema
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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

#### Usage
```bash
# Start data collection service
poetry run python src/data/downloaders/options_scheduler.py start

# Stop service
poetry run python src/data/downloaders/options_scheduler.py stop

# Check status
poetry run python src/data/downloaders/options_scheduler.py status

# Restart service
poetry run python src/data/downloaders/options_scheduler.py restart
```

---

### 2. Auto Options Collector

**File**: `src/data/downloaders/auto_options_collector.py`  
**Purpose**: Enhanced options data collection with automatic scheduling  

#### Features
- ✅ **Automatic market hours detection**
- ✅ **Trading day validation**
- ✅ **Error handling and recovery**
- ✅ **Performance monitoring**

#### Usage
```bash
# Start auto collector
poetry run python start_auto_options_collector.py
```

---

### 3. Maintenance Scheduler

**File**: `src/data/update/maintenance_scheduler.py`  
**Purpose**: System maintenance and data cleanup  

#### Features
- ✅ **Database maintenance**
- ✅ **Data cleanup tasks**
- ✅ **Performance optimization**
- ✅ **Configurable scheduling**

#### Configuration
```python
# Maintenance configuration
maintenance_config = {
    "enabled": True,
    "interval_hours": 24,
    "tasks": ["cleanup", "optimize", "backup"]
}
```

---

## 🔧 Setup & Configuration

### Auto-Start Setup

#### PowerShell Script (Recommended)
**File**: `setup_auto_start.ps1`

```powershell
# Run as Administrator
.\setup_auto_start.ps1
```

**Features**:
- ✅ **Automatic task creation**
- ✅ **Path validation**
- ✅ **Error handling**
- ✅ **Status reporting**

#### Batch Script (Alternative)
**File**: `scripts/setup_auto_options_scheduler.bat`

```cmd
# Run as Administrator
scripts\setup_auto_options_scheduler.bat
```

### Manual Control Scripts

#### Start Auto Options Collector
**File**: `start_auto_options_collector.py`

```bash
poetry run python start_auto_options_collector.py
```

---

## 📁 File Structure

```
ai-hedge-fund/
├── junk/
│   ├── auto_market_hours_scheduler.py    # Main auto-start scheduler
│   ├── run_options_scheduler.py          # Enhanced manual scheduler
│   └── fixed_options_tracker.py          # Legacy tracker
├── src/
│   └── data/
│       ├── downloaders/
│       │   ├── options_scheduler.py      # Data collection scheduler
│       │   └── auto_options_collector.py # Auto data collector
│       └── update/
│           └── maintenance_scheduler.py  # System maintenance
├── scripts/
│   └── setup_auto_options_scheduler.bat  # Auto-start setup
├── setup_auto_start.ps1                  # PowerShell auto-start
└── start_auto_options_collector.py       # Data collector starter
```

---

## 🎯 Usage Scenarios

### Scenario 1: Daily Trading Setup
1. **Auto-start setup**: Run `setup_auto_start.ps1` as Administrator
2. **Data collection**: Start `start_auto_options_collector.py`
3. **Automatic operation**: System runs from 9:30 AM to 3:30 PM IST

### Scenario 2: Manual Testing
1. **Enhanced scheduler**: Run `junk/run_options_scheduler.py`
2. **Continuous mode**: Use `--mode continuous` for testing
3. **Manual control**: Start/stop as needed

### Scenario 3: Data Collection Only
1. **Data collector**: Run `src/data/downloaders/options_scheduler.py start`
2. **Monitor**: Check status with `status` command
3. **Database**: Data stored in `data/options_chain_data.duckdb`

---

## 🔍 Monitoring & Troubleshooting

### Check Running Processes
```bash
# Check all Python processes
tasklist | findstr python

# Check specific scheduler
Get-ScheduledTask -TaskName "AIHedgeFundOptionsAnalyzer"
```

### View Logs
```bash
# Auto scheduler logs
tail -f logs/auto_options_scheduler.log

# Data collector logs
tail -f logs/options_collector.log
```

### Database Status
```bash
# Check options data
poetry run python -c "
import duckdb
conn = duckdb.connect('data/options_chain_data.duckdb')
print(conn.execute('SELECT COUNT(*) FROM options_chain_data').fetchone())
"
```

### Common Issues

#### 1. Database Lock
**Symptom**: `IO Error: Cannot open file... being used by another process`
**Solution**: 
- Check for multiple processes: `tasklist | findstr python`
- Terminate redundant processes: `taskkill /f /pid <PID>`

#### 2. Auto-Start Not Working
**Symptom**: Scheduler doesn't start at system startup
**Solution**:
- Run setup script as Administrator
- Check Windows Task Scheduler
- Verify Python path in task configuration

#### 3. Market Hours Issues
**Symptom**: Scheduler runs outside market hours
**Solution**:
- Check system time zone (should be IST)
- Verify market hours configuration
- Check trading holiday detection

---

## 📊 Performance & Optimization

### Database Optimization
- **Separate databases**: Options data in dedicated `options_chain_data.duckdb`
- **Indexing**: Automatic index creation for performance
- **Cleanup**: Regular maintenance tasks

### Memory Management
- **Connection pooling**: Efficient database connections
- **Data streaming**: Batch processing for large datasets
- **Error recovery**: Automatic retry mechanisms

### Monitoring Metrics
- **Data collection rate**: Records per minute
- **Analysis performance**: Processing time per analysis
- **Error rates**: Failed requests and recovery
- **Database size**: Storage usage and growth

---

## 🔄 Maintenance

### Daily Tasks
- **Log rotation**: Automatic log file management
- **Database cleanup**: Remove old data
- **Performance monitoring**: Check system health

### Weekly Tasks
- **Database optimization**: Rebuild indexes
- **Log analysis**: Review error patterns
- **Configuration review**: Update settings as needed

### Monthly Tasks
- **Data archival**: Move old data to archive
- **Performance tuning**: Optimize queries and processes
- **Security review**: Update access controls

---

## 📞 Support

### Getting Help
1. **Check logs**: Review log files for error messages
2. **Verify configuration**: Ensure paths and settings are correct
3. **Test components**: Run individual components to isolate issues
4. **Documentation**: Refer to this README and other docs

### Common Commands
```bash
# Check scheduler status
Get-ScheduledTask -TaskName "AIHedgeFundOptionsAnalyzer"

# View recent logs
Get-Content logs/auto_options_scheduler.log -Tail 50

# Test database connection
poetry run python -c "import duckdb; conn = duckdb.connect('data/options_chain_data.duckdb'); print('Connected')"

# Check Python environment
poetry env info
```

---

## 📝 Version History

### v1.0.0 (Current)
- ✅ **Auto market hours scheduler**
- ✅ **Enhanced options scheduler**
- ✅ **Options data collection scheduler**
- ✅ **Auto-start configuration**
- ✅ **Database separation**
- ✅ **Trading holiday detection**

### Planned Features
- 🔄 **Web dashboard for monitoring**
- 🔄 **Email notifications**
- 🔄 **Advanced error handling**
- 🔄 **Performance analytics**
- 🔄 **Multi-exchange support**

---

*Last Updated: August 25, 2025*  
*Version: 1.0.0*
