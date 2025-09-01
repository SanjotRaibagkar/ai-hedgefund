# 📊 Options Parquet Data Analysis Summary
## Date: August 29, 2025 (20250829)

---

## 📁 **Data Overview**

### **Available Dates**
- `20250827` - August 27, 2025
- `20250829` - August 29, 2025 ⭐ **Current Analysis**
- `20250901` - September 1, 2025

---

## 🎯 **NIFTY Analysis**

### **Data Summary**
- **Total Files:** 306 parquet files
- **Time Range:** 09:24:16 AM to 15:28:56 PM
- **Records per file:** ~763-764 records
- **Columns:** 29 data columns

### **File Distribution by Hour**
- **09:00** - 32 files (09:24:16 to 09:59:02)
- **10:00** - 32 files (10:01:26 to 10:58:54)
- **11:00** - 55 files (11:00:14 to 11:59:21)
- **12:00** - 54 files (12:00:26 to 12:59:23)
- **13:00** - 54 files (13:00:34 to 13:59:47)
- **14:00** - 53 files (14:00:57 to 14:59:44)
- **15:00** - 26 files (15:00:52 to 15:28:56)

### **Strike Price Range**
- **Minimum:** ₹17,000
- **Maximum:** ₹31,000
- **Unique Strikes:** 114
- **ATM Range:** Around ₹19,500 ± ₹500

### **Expiry Dates Available**
- **September 2025:** 02-Sep, 09-Sep, 16-Sep, 23-Sep, 30-Sep
- **October 2025:** 28-Oct
- **November 2025:** 25-Nov
- **December 2025:** 30-Dec
- **March 2026:** 31-Mar
- **June 2026:** 30-Jun
- **December 2026:** 29-Dec
- **June 2027:** 29-Jun
- **December 2027:** 28-Dec
- **June 2028:** 27-Jun
- **December 2028:** 26-Dec
- **June 2029:** 26-Jun
- **December 2029:** 24-Dec
- **June 2030:** 25-Jun

### **Key Metrics (September 30, 2025 Expiry)**
- **Total Strikes:** 106
- **Put-Call Ratio (PCR):** 1.176 (Bullish sentiment)
- **Most Active Strikes:**
  - ₹25,000: 76,618 CALLS OI, 58,824 PUTS OI
  - ₹24,500: 29,432 CALLS OI, 63,299 PUTS OI
  - ₹26,000: 53,660 CALLS OI, 20,760 PUTS OI

---

## 🏦 **BANKNIFTY Analysis**

### **Data Summary**
- **Total Files:** 305 parquet files
- **Time Range:** 09:24:16 AM to 15:28:56 PM
- **Records per file:** ~394-428 records
- **Columns:** 29 data columns

### **File Distribution by Hour**
- **09:00** - 32 files (09:24:16 to 09:59:02)
- **10:00** - 31 files (10:02:38 to 10:58:54)
- **11:00** - 55 files (11:00:14 to 11:59:21)
- **12:00** - 54 files (12:00:26 to 12:59:23)
- **13:00** - 54 files (13:00:34 to 13:59:47)
- **14:00** - 53 files (14:00:57 to 14:59:44)
- **15:00** - 26 files (15:00:52 to 15:28:56)

### **Strike Price Range**
- **Minimum:** ₹40,500
- **Maximum:** ₹65,000
- **Unique Strikes:** 152
- **ATM Range:** Around ₹47,000 ± ₹1,000

### **Expiry Dates Available**
- **September 2025:** 30-Sep
- **October 2025:** 28-Oct
- **November 2025:** 25-Nov
- **December 2025:** 30-Dec
- **March 2026:** 31-Mar
- **June 2026:** 30-Jun

### **Key Metrics (September 30, 2025 Expiry)**
- **Total Strikes:** 152
- **Put-Call Ratio (PCR):** 0.857 (Slightly bearish sentiment)
- **Most Active Strikes:**
  - ₹54,000: 25,902 CALLS OI, 31,478 PUTS OI
  - ₹55,000: 30,088 CALLS OI, 20,262 PUTS OI
  - ₹56,000: 34,877 CALLS OI, 12,631 PUTS OI
  - ₹57,000: 52,772 CALLS OI, 29,413 PUTS OI

---

## 📈 **Intraday Movement Analysis**

### **Data Collection Frequency**
- **Collection Interval:** Approximately every 1 minute
- **Market Hours Coverage:** 9:15 AM to 3:30 PM IST
- **Total Snapshots:** 306 for NIFTY, 305 for BANKNIFTY

### **Key Observations**
1. **High Frequency Data:** 1-minute intervals provide granular intraday analysis
2. **Complete Market Coverage:** From market open to close
3. **Strike Price Granularity:** Fine-grained strike prices for precise analysis
4. **Multiple Expiries:** Coverage across multiple expiry cycles

---

## 🔍 **Data Quality Assessment**

### **Strengths**
- ✅ **High Frequency:** 1-minute snapshots for detailed analysis
- ✅ **Complete Coverage:** Full trading day data
- ✅ **Rich Data:** 29 columns including OI, Volume, IV, LTP
- ✅ **Multiple Expiries:** Various expiry cycles available
- ✅ **Consistent Format:** Standardized parquet format

### **Data Fields Available**
- **Basic Info:** Fetch_Time, Symbol, Expiry_Date, Strike_Price
- **CALLS Data:** OI, Change_in_OI, Volume, IV, LTP, Net_Change, Bid/Ask
- **PUTS Data:** OI, Change_in_OI, Volume, IV, LTP, Net_Change, Bid/Ask
- **Metadata:** collection_timestamp, index_name, date

---

## 💡 **Backtesting Applications**

### **Intraday Strategies**
1. **Delta Hedging:** Real-time position adjustments
2. **Gamma Scalping:** Volatility-based trading
3. **Theta Decay Analysis:** Time decay studies
4. **IV Skew Analysis:** Implied volatility patterns

### **Risk Management**
1. **Portfolio Greeks:** Real-time risk metrics
2. **Position Sizing:** Based on current market conditions
3. **Stop Loss Optimization:** Using historical patterns

### **Research Opportunities**
1. **Market Microstructure:** Order flow analysis
2. **Volatility Patterns:** Intraday volatility studies
3. **Option Flow Analysis:** Institutional vs retail activity
4. **Cross-Asset Correlation:** NIFTY vs BANKNIFTY relationships

---

## 📊 **Summary Statistics**

| Metric | NIFTY | BANKNIFTY |
|--------|-------|------------|
| **Total Files** | 306 | 305 |
| **Records per File** | 763-764 | 394-428 |
| **Strike Range** | ₹17,000 - ₹31,000 | ₹40,500 - ₹65,000 |
| **Unique Strikes** | 114 | 152 |
| **PCR (Sep 30)** | 1.176 | 0.857 |
| **Data Columns** | 29 | 29 |
| **Collection Frequency** | 1 minute | 1 minute |

---

## 🚀 **Next Steps for Backtesting**

1. **Data Preprocessing:** Clean and validate parquet data
2. **Feature Engineering:** Calculate derived metrics (Greeks, ratios)
3. **Strategy Development:** Implement intraday options strategies
4. **Backtesting Framework:** Build comprehensive testing system
5. **Performance Analysis:** Risk-adjusted returns, drawdown analysis
6. **Optimization:** Parameter tuning and strategy refinement

---

*Generated on: September 1, 2025*  
*Data Source: Options Parquet Files*  
*Analysis Tool: ParquetDataAnalyzer & OptionsDataTable*
