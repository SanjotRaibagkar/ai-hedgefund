# Auto Scheduler Tomorrow Setup Guide

## 🎯 **How the Scheduler Will Work Tomorrow**

The auto market hours scheduler will automatically:
1. **Start at system startup** (when you turn on your computer)
2. **Wait for market hours** (9:30 AM IST)
3. **Run analysis every 15 minutes** during market hours
4. **Stop at market close** (3:30 PM IST)
5. **Repeat daily** on trading days only (excludes weekends and holidays)

## 🚀 **Setup Instructions for Tomorrow**

### **Step 1: Set Up Automatic Startup**

**Run as Administrator:**
```bash
scripts\setup_auto_options_scheduler.bat
```

This will:
- ✅ Create a Windows Task Scheduler task
- ✅ Set it to start automatically at system startup
- ✅ Configure it to run the auto market hours scheduler
- ✅ Use the correct Python environment and script

### **Step 2: Verify Setup**

**Check if task was created:**
```bash
schtasks /query /tn AIHedgeFundAutoOptionsScheduler
```

**Test the task manually:**
```bash
schtasks /run /tn AIHedgeFundAutoOptionsScheduler
```

### **Step 3: Monitor Tomorrow**

**Check if it's running:**
```bash
tasklist | findstr python
```

**View logs:**
```bash
# Check scheduler logs
type logs\auto_options_scheduler.log

# Check CSV file
dir results\options_tracker\option_tracker.csv
```

## 📊 **What Will Happen Tomorrow**

### **9:30 AM - Market Opens**
- ✅ Scheduler detects market open
- ✅ Runs initial analysis for NIFTY and BANKNIFTY
- ✅ Creates first records in `option_tracker.csv`

### **Every 15 Minutes During Market Hours**
- ✅ 9:30 AM - Initial analysis
- ✅ 9:45 AM - Second analysis
- ✅ 10:00 AM - Third analysis
- ✅ ... continues every 15 minutes
- ✅ 3:15 PM - Last analysis
- ✅ 3:30 PM - Market closes, analysis stops

### **Outside Market Hours**
- ✅ 3:30 PM - 9:30 AM next day: No analysis
- ✅ Weekends: No analysis
- ✅ Trading holidays: No analysis

## 🔧 **Files and Locations**

### **Scheduler Files**
- **Main Scheduler**: `junk/auto_market_hours_scheduler.py`
- **Setup Script**: `scripts/setup_auto_options_scheduler.bat`
- **Test Script**: `test_auto_scheduler.py`

### **Output Files**
- **CSV Data**: `results/options_tracker/option_tracker.csv`
- **Logs**: `logs/auto_options_scheduler.log`

### **Windows Task**
- **Task Name**: `AIHedgeFundAutoOptionsScheduler`
- **Trigger**: At system startup
- **Action**: Run auto market hours scheduler

## 📈 **Expected Results Tomorrow**

### **CSV File Growth**
- **9:30 AM**: 2 records (NIFTY + BANKNIFTY)
- **9:45 AM**: 4 records
- **10:00 AM**: 6 records
- **...**
- **3:15 PM**: ~48 records (24 intervals × 2 indices)

### **Data Quality**
- ✅ **Accurate spot prices** from futures data
- ✅ **Correct ATM strikes** based on current prices
- ✅ **Real-time PCR** and OI data
- ✅ **Proper signals** (BULLISH/BEARISH/NEUTRAL/RANGE)
- ✅ **Performance tracking** with accuracy status

## 🎯 **Key Features**

### **Smart Scheduling**
- ✅ **Market hours aware**: Only runs 9:30 AM - 3:30 PM
- ✅ **Trading day aware**: Excludes weekends and holidays
- ✅ **Automatic start/stop**: No manual intervention needed
- ✅ **Error handling**: Continues running even if individual analyses fail

### **Data Accuracy**
- ✅ **Uses Fixed Enhanced Options Analyzer**: No database dependencies
- ✅ **Futures data for spot prices**: Most accurate method
- ✅ **ATM ± 2 strikes strategy**: Professional options analysis
- ✅ **Performance tracking**: Monitors signal accuracy

### **Reliability**
- ✅ **Windows Task Scheduler**: Built-in Windows reliability
- ✅ **Automatic restart**: If system reboots, scheduler restarts
- ✅ **Logging**: Complete audit trail of all activities
- ✅ **Error recovery**: Handles network issues gracefully

## 🚨 **Troubleshooting**

### **If Scheduler Doesn't Start**
1. **Check Windows Task Scheduler:**
   ```bash
   schtasks /query /tn AIHedgeFundAutoOptionsScheduler
   ```

2. **Run manually to test:**
   ```bash
   poetry run python junk/auto_market_hours_scheduler.py --mode auto
   ```

3. **Check logs:**
   ```bash
   type logs\auto_options_scheduler.log
   ```

### **If Analysis Fails**
1. **Check network connection**
2. **Verify NSE API access**
3. **Check Python environment:**
   ```bash
   poetry run python -c "from src.nsedata.NseUtility import NseUtils; print('NSE API working')"
   ```

### **If CSV File Not Created**
1. **Check directory permissions**
2. **Verify working directory**
3. **Run test script:**
   ```bash
   poetry run python test_auto_scheduler.py
   ```

## 📋 **Commands Reference**

### **Setup Commands**
```bash
# Set up automatic startup
scripts\setup_auto_options_scheduler.bat

# Test the scheduler
poetry run python test_auto_scheduler.py

# Run manually (continuous mode)
poetry run python junk/auto_market_hours_scheduler.py --mode continuous

# Run manually (auto mode)
poetry run python junk/auto_market_hours_scheduler.py --mode auto
```

### **Monitoring Commands**
```bash
# Check if task exists
schtasks /query /tn AIHedgeFundAutoOptionsScheduler

# Run task manually
schtasks /run /tn AIHedgeFundAutoOptionsScheduler

# Stop task
schtasks /end /tn AIHedgeFundAutoOptionsScheduler

# Delete task
schtasks /delete /tn AIHedgeFundAutoOptionsScheduler /f

# Check running processes
tasklist | findstr python

# View logs
type logs\auto_options_scheduler.log

# Check CSV file
dir results\options_tracker\option_tracker.csv
```

## 🎉 **Success Indicators**

### **Tomorrow Morning (9:30 AM)**
- ✅ Windows Task Scheduler shows task is running
- ✅ `logs/auto_options_scheduler.log` shows "Market opened - starting options analysis"
- ✅ `results/options_tracker/option_tracker.csv` has new records with 9:30 AM timestamps

### **Throughout the Day**
- ✅ New records added every 15 minutes
- ✅ Logs show successful analysis for both NIFTY and BANKNIFTY
- ✅ CSV file grows with accurate data

### **Market Close (3:30 PM)**
- ✅ Logs show "Market closed - stopping options analysis"
- ✅ No new records added after 3:30 PM
- ✅ Scheduler continues running but waits for next market open

## 🚀 **Ready for Tomorrow!**

The auto market hours scheduler is now ready to:
- ✅ **Start automatically** when you turn on your computer
- ✅ **Run during market hours** (9:30 AM - 3:30 PM)
- ✅ **Analyze options every 15 minutes** with accurate data
- ✅ **Create comprehensive CSV file** for strategy development
- ✅ **Work reliably** day after day on trading days

**Just run `scripts\setup_auto_options_scheduler.bat` as Administrator and you're all set for tomorrow!** 🎯
