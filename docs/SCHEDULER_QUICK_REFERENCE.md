# Scheduler Quick Reference Guide

## 🚀 Quick Start Commands

### Auto-Start Setup (One-time)
```powershell
# Run as Administrator
.\setup_auto_start.ps1
```

### Daily Operations
```bash
# Start data collection
poetry run python start_auto_options_collector.py

# Manual options analysis
poetry run python junk/run_options_scheduler.py

# Check status
poetry run python src/data/downloaders/options_scheduler.py status
```

---

## 📋 Scheduler Summary

| Scheduler | File | Purpose | Auto-Start | Market Hours |
|-----------|------|---------|------------|--------------|
| **Auto Market Hours** | `junk/auto_market_hours_scheduler.py` | Options Analysis | ✅ Yes | 9:30 AM - 3:30 PM |
| **Enhanced Options** | `junk/run_options_scheduler.py` | Manual Analysis | ❌ No | 9:30 AM - 3:30 PM |
| **Data Collection** | `src/data/downloaders/options_scheduler.py` | Collect Data | ❌ No | 9:30 AM - 3:30 PM |
| **Auto Collector** | `src/data/downloaders/auto_options_collector.py` | Auto Data Collection | ❌ No | 9:30 AM - 3:30 PM |

---

## 🎯 Common Scenarios

### Scenario 1: Tomorrow Morning Setup
```bash
# 1. Set up auto-start (run once as Administrator)
.\setup_auto_start.ps1

# 2. Start data collection
poetry run python start_auto_options_collector.py

# 3. System will automatically:
#    - Start analysis at 9:30 AM
#    - Run every 15 minutes
#    - Stop at 3:30 PM
```

### Scenario 2: Manual Testing
```bash
# Start enhanced scheduler (ignores market hours for testing)
poetry run python junk/run_options_scheduler.py

# Or continuous mode
poetry run python junk/auto_market_hours_scheduler.py --mode continuous
```

### Scenario 3: Data Collection Only
```bash
# Start data collection service
poetry run python src/data/downloaders/options_scheduler.py start

# Check status
poetry run python src/data/downloaders/options_scheduler.py status

# Stop service
poetry run python src/data/downloaders/options_scheduler.py stop
```

---

## 🔍 Monitoring Commands

### Check Running Processes
```bash
# All Python processes
tasklist | findstr python

# Specific scheduler
Get-ScheduledTask -TaskName "AIHedgeFundOptionsAnalyzer"
```

### View Logs
```bash
# Auto scheduler logs
Get-Content logs/auto_options_scheduler.log -Tail 20

# Data collector logs
Get-Content logs/options_collector.log -Tail 20
```

### Database Status
```bash
# Check options data count
poetry run python -c "
import duckdb
conn = duckdb.connect('data/options_chain_data.duckdb')
print(f'Options records: {conn.execute(\"SELECT COUNT(*) FROM options_chain_data\").fetchone()[0]:,}')
"
```

---

## 🛠️ Troubleshooting

### Database Lock
```bash
# Check processes
tasklist | findstr python

# Kill specific process
taskkill /f /pid <PID>
```

### Auto-Start Issues
```bash
# Check task exists
Get-ScheduledTask -TaskName "AIHedgeFundOptionsAnalyzer"

# Run manually
Start-ScheduledTask -TaskName "AIHedgeFundOptionsAnalyzer"
```

### Market Hours Issues
```bash
# Check current time
Get-Date

# Verify timezone (should be IST)
[System.TimeZoneInfo]::Local
```

---

## 📊 Output Files

### Analysis Results
- **CSV**: `results/options_tracker/option_tracker.csv`
- **Logs**: `logs/auto_options_scheduler.log`

### Data Collection
- **Database**: `data/options_chain_data.duckdb`
- **Logs**: `logs/options_collector.log`

---

## ⚡ Quick Commands

### Start Everything
```bash
# Auto-start setup (one-time)
.\setup_auto_start.ps1

# Data collection
poetry run python start_auto_options_collector.py
```

### Stop Everything
```bash
# Stop data collection
poetry run python src/data/downloaders/options_scheduler.py stop

# Kill all Python processes
taskkill /f /im python.exe
```

### Check Status
```bash
# Data collection status
poetry run python src/data/downloaders/options_scheduler.py status

# Auto-start task status
Get-ScheduledTask -TaskName "AIHedgeFundOptionsAnalyzer"
```

---

## 📞 Emergency Commands

### Force Stop All
```bash
# Kill all Python processes
taskkill /f /im python.exe

# Stop scheduled task
Stop-ScheduledTask -TaskName "AIHedgeFundOptionsAnalyzer"
```

### Restart Everything
```bash
# Restart data collection
poetry run python src/data/downloaders/options_scheduler.py restart

# Run enhanced scheduler
poetry run python junk/run_options_scheduler.py
```

### Database Reset
```bash
# Backup current database
cp data/options_chain_data.duckdb data/options_chain_data_backup.duckdb

# Create new database
rm data/options_chain_data.duckdb
```

---

*Last Updated: August 25, 2025*
