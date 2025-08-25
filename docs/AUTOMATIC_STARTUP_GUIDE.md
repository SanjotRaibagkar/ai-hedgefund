# Automatic Startup Guide for Options Data Collection

This guide explains how to set up automatic startup for the options data collection service so it starts automatically at 9:30 AM every trading day.

## 🎯 **Current Status**

✅ **Options Data Collection**: Currently running (PID: 13752)  
✅ **Market Hours**: Active (9:30 AM - 3:30 PM IST)  
✅ **Data Collection**: 1,181+ records per minute  
✅ **Database**: 2,362+ options records collected  

## 🚀 **Setup Options**

### **Option 1: Windows Task Scheduler (Recommended)**

#### **Step 1: Create the Task**
```bash
# Run as Administrator
scripts\setup_task_scheduler.bat
```

#### **Step 2: Verify the Task**
```bash
# Check if task was created
schtasks /query /tn AIHedgeFundOptionsCollector
```

#### **Step 3: Test the Task**
```bash
# Run the task manually to test
schtasks /run /tn AIHedgeFundOptionsCollector
```

### **Option 2: Windows Service**

#### **Step 1: Install as Service**
```bash
# Run as Administrator
scripts\install_options_service.bat
```

#### **Step 2: Start the Service**
```bash
# Start the service
sc start AIHedgeFundOptionsCollector

# Check service status
sc query AIHedgeFundOptionsCollector
```

### **Option 3: Auto Options Collector (Advanced)**

#### **Step 1: Start Auto Collector**
```bash
# Start with automatic scheduling
poetry run python start_auto_options_collector.py

# Or start with immediate collection
poetry run python src/data/downloaders/auto_options_collector.py --start-now
```

## 📊 **Monitoring Tools**

### **1. Quick Status Check**
```bash
# Check current status
poetry run python check_options_status.py

# Continuous monitoring
poetry run python check_options_status.py --continuous --interval 30
```

### **2. Comprehensive Dashboard**
```bash
# Full monitoring dashboard
poetry run python monitor_options_dashboard.py

# Continuous dashboard
poetry run python monitor_options_dashboard.py --continuous
```

### **3. Database Monitoring**
```bash
# Check database statistics
poetry run python monitor_options_data.py
```

## ⏰ **Schedule Configuration**

### **Trading Hours**
- **Market Open**: 9:30 AM IST
- **Market Close**: 3:30 PM IST
- **Collection Interval**: 1 minute
- **Trading Days**: Monday to Friday (excluding holidays)

### **Automatic Behavior**
- ✅ **Auto Start**: 9:30 AM daily
- ✅ **Auto Stop**: 3:30 PM daily
- ✅ **Holiday Detection**: Automatically skips trading holidays
- ✅ **Weekend Detection**: Automatically skips weekends
- ✅ **Error Recovery**: Automatic restart on failures

## 🔧 **Manual Control**

### **Start Collection**
```bash
# Start immediately
poetry run python start_options_collector.py

# Start with auto scheduling
poetry run python start_auto_options_collector.py
```

### **Stop Collection**
```bash
# Find the process
poetry run python check_options_status.py

# Stop the process (replace PID with actual PID)
taskkill /PID <PID> /F
```

### **Restart Collection**
```bash
# Stop and restart
taskkill /PID <PID> /F
poetry run python start_options_collector.py
```

## 📈 **Data Collection Statistics**

### **Current Performance**
- **NIFTY**: 816 records per collection
- **BANKNIFTY**: 365 records per collection
- **Total**: 1,181 records per minute
- **Daily Total**: 2,362+ records collected today

### **Database Growth**
- **Options Records**: 2,362+ and growing
- **Storage**: Efficient DuckDB compression
- **Query Performance**: Sub-second response times

## 🚨 **Troubleshooting**

### **Common Issues**

#### **1. Service Won't Start**
```bash
# Check service status
sc query AIHedgeFundOptionsCollector

# Check service logs
eventvwr.msc
```

#### **2. Task Scheduler Issues**
```bash
# Check task status
schtasks /query /tn AIHedgeFundOptionsCollector

# Delete and recreate task
schtasks /delete /tn AIHedgeFundOptionsCollector /f
scripts\setup_task_scheduler.bat
```

#### **3. Database Locked**
```bash
# Check if collector is running
poetry run python check_options_status.py

# Stop all Python processes (use with caution)
taskkill /IM python.exe /F
```

#### **4. Memory Issues**
```bash
# Check memory usage
poetry run python check_options_status.py

# Restart if memory usage is high
taskkill /PID <PID> /F
poetry run python start_options_collector.py
```

### **Log Files**
- **Application Logs**: Console output
- **Status File**: `data/options_collector_status.json`
- **Database Logs**: DuckDB internal logging

## 🔄 **Maintenance**

### **Daily Checks**
1. **Morning**: Verify service started at 9:30 AM
2. **Midday**: Check data collection is active
3. **Evening**: Verify service stopped at 3:30 PM

### **Weekly Checks**
1. **Monday**: Verify weekend restart
2. **Friday**: Check weekly data totals
3. **Holidays**: Verify holiday detection

### **Monthly Checks**
1. **Database Size**: Monitor storage growth
2. **Performance**: Check collection rates
3. **Errors**: Review error logs

## 📋 **Setup Checklist**

### **Before Market Open**
- [ ] Verify system is running
- [ ] Check internet connection
- [ ] Ensure database is accessible
- [ ] Monitor service startup

### **During Market Hours**
- [ ] Monitor data collection rates
- [ ] Check system resources
- [ ] Verify data quality
- [ ] Monitor for errors

### **After Market Close**
- [ ] Verify service stopped
- [ ] Check daily data totals
- [ ] Review error logs
- [ ] Backup status files

## 🎉 **Success Indicators**

### **✅ Service Running**
- Process visible in task manager
- Status shows "Running"
- Memory usage stable

### **✅ Data Collection Active**
- Records being added to database
- Collection rate: 1,181+ records/minute
- No error messages in logs

### **✅ Market Hours Detection**
- Service starts at 9:30 AM
- Service stops at 3:30 PM
- Skips weekends and holidays

## 📞 **Support**

If you encounter issues:

1. **Check Status**: `poetry run python check_options_status.py`
2. **Review Logs**: Check console output for errors
3. **Restart Service**: Stop and restart the collector
4. **Check System**: Verify system resources and connectivity

---

**🎯 Your options data collection is now set up for automatic operation!**
