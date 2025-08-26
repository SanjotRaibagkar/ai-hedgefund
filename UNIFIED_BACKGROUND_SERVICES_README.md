# 🚀 Unified Background Services Manager

## Overview

The `start_all_background_services.py` script is a unified manager that intelligently manages all 3 background services for the AI Hedge Fund system:

1. **Options Analysis Scheduler** - Generates trading strategies every 15 minutes
2. **Intraday Data Collection** - Collects intraday options data every 15 minutes  
3. **Options Chain Collection** - Collects options chain data every 3 minutes

## 🎯 Key Features

### ✅ **Intelligent Process Management**
- **No Duplicates**: Automatically detects existing processes and won't start duplicates
- **Auto-Restart**: Monitors processes and restarts them if they stop
- **Health Monitoring**: Continuous health checks every 30 seconds
- **Graceful Shutdown**: Proper cleanup when stopping

### ✅ **Comprehensive Logging**
- **Console Output**: Real-time status updates
- **File Logging**: Detailed logs in `logs/unified_background_manager.log`
- **Status Reports**: Automatic status reports every 5 minutes

### ✅ **Easy Management**
- **Single Command**: Start all services with one command
- **Status Checking**: Quick status check with `check_background_status.py`
- **Process Control**: Automatic process lifecycle management

## 🚀 Usage

### Start All Services
```bash
poetry run python start_all_background_services.py
```

### Check Service Status
```bash
poetry run python check_background_status.py
```

### Stop All Services
Press `Ctrl+C` in the terminal running the unified manager.

## 📊 Expected Output

### Starting Services
```
🚀 Unified Background Services Manager
==================================================
Managing: Options Analysis, Intraday Collection, Options Collection
==================================================
🚀 Starting Unified Background Services Manager
============================================================
🔍 Found existing Options Analysis Scheduler process: PID 19088
✅ Options Analysis Scheduler already running (PID: 19088)
🔄 Starting Intraday Data Collection...
✅ Intraday Data Collection started successfully with PID: 21844
🔄 Starting Options Chain Collection...
✅ Options Chain Collection started successfully with PID: 9444
============================================================
✅ All services initialized
```

### Status Report (Every 5 minutes)
```
================================================================================
📊 UNIFIED BACKGROUND SERVICES STATUS
================================================================================
✅ Options Analysis Scheduler | PID: 19088        | Uptime: 1:45:32
✅ Intraday Data Collection  | PID: 21844        | Uptime: 0:12:45
✅ Options Chain Collection  | PID: 9444         | Uptime: 0:08:23
================================================================================
```

## 🔧 How It Works

### 1. **Process Detection**
- Scans for existing processes by script name
- Identifies running services and their PIDs
- Prevents duplicate process creation

### 2. **Service Management**
- Starts only missing services
- Monitors process health continuously
- Auto-restarts failed services

### 3. **Health Monitoring**
- Checks process status every 30 seconds
- Validates process responsiveness
- Logs health status and issues

### 4. **Graceful Shutdown**
- Handles SIGINT and SIGTERM signals
- Terminates processes gracefully
- Cleans up resources properly

## 📁 File Structure

```
ai-hedge-fund/
├── start_all_background_services.py    # Main unified manager
├── check_background_status.py          # Status checker
├── logs/
│   ├── unified_background_manager.log  # Manager logs
│   ├── options_analysis_background.log # Options analysis logs
│   ├── intraday_data_collection_background.log # Intraday logs
│   └── options_collector_background.log # Options collection logs
└── UNIFIED_BACKGROUND_SERVICES_README.md # This file
```

## 🎯 Benefits

### ✅ **Simplified Management**
- **One Command**: Start all services with a single command
- **No Manual Monitoring**: Automatic process management
- **No Duplicates**: Intelligent duplicate prevention

### ✅ **Reliability**
- **Auto-Restart**: Services restart automatically if they fail
- **Health Monitoring**: Continuous health checks
- **Error Recovery**: Graceful error handling and recovery

### ✅ **Visibility**
- **Real-time Status**: Live status updates
- **Comprehensive Logging**: Detailed logs for troubleshooting
- **Status Reports**: Regular status summaries

## 🚨 Troubleshooting

### Service Not Starting
1. Check if Poetry environment is activated
2. Verify all dependencies are installed
3. Check individual service logs in `logs/` directory

### Process Duplicates
- The unified manager automatically prevents duplicates
- If you see duplicates, stop all processes and restart the unified manager

### Service Crashes
- The manager automatically restarts crashed services
- Check logs for specific error messages
- Verify market hours and data availability

## 📈 Monitoring

### Log Files
- **Manager Logs**: `logs/unified_background_manager.log`
- **Service Logs**: Individual service logs in `logs/` directory

### Status Checking
- **Quick Check**: `poetry run python check_background_status.py`
- **Real-time**: Watch the unified manager console output
- **Log Analysis**: Review log files for detailed information

## 🎉 Success Indicators

✅ **All Services Running**: Status shows all 3 services as "✅ Running"  
✅ **Regular Status Reports**: Status reports every 5 minutes  
✅ **No Error Messages**: Clean console output without errors  
✅ **Data Collection**: Database files being updated regularly  
✅ **Log Activity**: Active logging in all log files  

---

**🎯 The unified manager provides a single, reliable way to manage all background services with intelligent process management and comprehensive monitoring.**
