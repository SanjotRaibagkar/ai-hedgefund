# Comprehensive Test Results

**Timestamp**: 2025-08-23T20:48:01.225757
**Status**: FAIL
**Success Rate**: 9.1%

## Summary

- Total Tests: 11
- Passed: 1
- Failed: 8
- Warnings: 2

## Test Script Results

### Data Infrastructure
- Status: FAIL
- Duration: 67.51s

### Llm Analysis
- Status: FAIL
- Duration: 4.35s

### Strategies
- Status: FAIL
- Duration: 18.93s

### Backtesting
- Status: FAIL
- Duration: 10.54s

## Additional Test Results

### Documentation
- Tests: 4
- Passed: 0
- Success Rate: 0.0%

### Github Integration
- Tests: 3
- Passed: 1
- Success Rate: 33.3%

## Detailed Results

```json
{
  "data_infrastructure": {
    "script_name": "data_infrastructure",
    "script_path": "test_scripts/test_data_infrastructure.py",
    "start_time": "2025-08-23T20:46:19.546282",
    "end_time": "2025-08-23T20:47:27.055337",
    "duration": 67.50889682769775,
    "exit_code": 1,
    "output": "\n============================================================\nData Infrastructure Test Results\n============================================================\nTotal Tests: 25\nPassed: 11\nFailed: 6\nWarnings: 8\nSuccess Rate: 44.0%\nStatus: FAIL\n============================================================\nResults saved to: C:\\sanjot\\virattt\\ai-hedge-fund\\tests\\comprehensive_test_suite\\test_scripts\\../test_results/data_infrastructure_results.md\n",
    "error": "2025-08-23T20:46:26.779441+0530 | INFO | SQLite database initialized successfully at data/ai_hedge_fund.db\n2025-08-23T20:46:26.779837+0530 | INFO | Database manager initialized successfully\n2025-08-23T20:46:26.780093+0530 | INFO | Data Infrastructure Tester initialized\n2025-08-23T20:46:26.780292+0530 | INFO | Starting Data Infrastructure Test Suite...\n2025-08-23T20:46:26.780482+0530 | INFO | Testing data retrieval...\n2025-08-23T20:46:26.780658+0530 | INFO | Testing US stock data retrieval: AAPL\n2025-08-23T20:46:42.291647+0530 | ERROR | \\u274c Price data error for AAPL: 'list' object has no attribute 'empty'\nError fetching financial metrics for AAPL: HTTPSConnectionPool(host='api.financialdatasets.ai', port=443): Max retries exceeded with url: /financial-metrics/?ticker=AAPL&report_period_lte=2023-12-31&limit=10&period=ttm (Caused by NameResolutionError(\"<urllib3.connection.HTTPSConnection object at 0x00000230B96C0190>: Failed to resolve 'api.financialdatasets.ai' ([Errno 11004] getaddrinfo failed)\"))\n2025-08-23T20:46:48.321759+0530 | INFO | \\u2705 Financial metrics retrieved for AAPL\nError fetching market cap for AAPL: Error fetching data: AAPL - 400 - {\"error\":\"Invalid period\",\"message\":\"Please provide a valid period: 'annual', 'quarterly', or 'ttm'\"}\n2025-08-23T20:46:51.017946+0530 | WARNING | \\u26a0\\ufe0f No market cap for AAPL\n2025-08-23T20:46:51.018331+0530 | INFO | Testing US stock data retrieval: MSFT\n2025-08-23T20:46:51.524305+0530 | ERROR | \\u274c Price data error for MSFT: 'list' object has no attribute 'empty'\n2025-08-23T20:46:53.450937+0530 | INFO | \\u2705 Financial metrics retrieved for MSFT\nError fetching market cap for MSFT: Error fetching data: MSFT - 400 - {\"error\":\"Invalid period\",\"message\":\"Please provide a valid period: 'annual', 'quarterly', or 'ttm'\"}\n2025-08-23T20:46:55.063162+0530 | WARNING | \\u26a0\\ufe0f No market cap for MSFT\n2025-08-23T20:46:55.063422+0530 | INFO | Testing US stock data retrieval: GOOGL\n2025-08-23T20:46:55.557942+0530 | ERROR | \\u274c Price data error for GOOGL: 'list' object has no attribute 'empty'\n2025-08-23T20:46:57.316106+0530 | INFO | \\u2705 Financial metrics retrieved for GOOGL\nError fetching market cap for GOOGL: Error fetching data: GOOGL - 400 - {\"error\":\"Invalid period\",\"message\":\"Please provide a valid period: 'annual', 'quarterly', or 'ttm'\"}\n2025-08-23T20:46:58.862312+0530 | WARNING | \\u26a0\\ufe0f No market cap for GOOGL\n2025-08-23T20:46:58.862581+0530 | INFO | Testing Indian stock data retrieval: RELIANCE.NS\nError fetching prices as DataFrame for RELIANCE.NS: HTTPSConnectionPool(host='www.nseindia.com', port=443): Max retries exceeded with url: /api/quote-equity?symbol=RELIANCE (Caused by NameResolutionError(\"<urllib3.connection.HTTPSConnection object at 0x00000230B96C2990>: Failed to resolve 'www.nseindia.com' ([Errno 11001] getaddrinfo failed)\"))\n2025-08-23T20:47:00.444013+0530 | WARNING | \\u26a0\\ufe0f No price data for RELIANCE.NS (may be expected)\nNo financial metrics found for Indian ticker: RELIANCE.NS\n2025-08-23T20:47:05.940270+0530 | INFO | \\u2705 Financial metrics retrieved for RELIANCE.NS\n2025-08-23T20:47:09.318755+0530 | WARNING | \\u26a0\\ufe0f No market cap for RELIANCE.NS\n2025-08-23T20:47:09.319098+0530 | INFO | Testing Indian stock data retrieval: TCS.NS\n2025-08-23T20:47:12.638921+0530 | INFO | \\u2705 Price data retrieved for TCS.NS: 1 records\nNo financial metrics found for Indian ticker: TCS.NS\n2025-08-23T20:47:15.890702+0530 | INFO | \\u2705 Financial metrics retrieved for TCS.NS\n2025-08-23T20:47:18.461759+0530 | WARNING | \\u26a0\\ufe0f No market cap for TCS.NS\n2025-08-23T20:47:18.462018+0530 | INFO | Testing Indian stock data retrieval: INFY.NS\n2025-08-23T20:47:20.554006+0530 | INFO | \\u2705 Price data retrieved for INFY.NS: 1 records\nNo financial metrics found for Indian ticker: INFY.NS\n2025-08-23T20:47:23.707121+0530 | INFO | \\u2705 Financial metrics retrieved for INFY.NS\n2025-08-23T20:47:26.061686+0530 | WARNING | \\u26a0\\ufe0f No market cap for INFY.NS\n2025-08-23T20:47:26.061966+0530 | INFO | Testing data storage...\n2025-08-23T20:47:26.062154+0530 | ERROR | \\u274c Database table creation failed: 'DatabaseManager' object has no attribute 'create_tables'\n2025-08-23T20:47:26.358964+0530 | ERROR | \\u274c Data storage test failed: 'list' object has no attribute 'empty'\n2025-08-23T20:47:26.359190+0530 | WARNING | \\u26a0\\ufe0f Data quality metrics warning: DatabaseManager.get_data_quality_metrics() missing 1 required positional argument: 'ticker'\n2025-08-23T20:47:26.359341+0530 | INFO | Testing data quality...\n2025-08-23T20:47:26.496806+0530 | ERROR | \\u274c Data quality test failed: 'list' object has no attribute 'empty'\n2025-08-23T20:47:26.497123+0530 | INFO | Testing update mechanisms...\n2025-08-23T20:47:26.502571+0530 | INFO | SQLite database initialized successfully at data/ai_hedge_fund.db\n2025-08-23T20:47:26.508311+0530 | INFO | \\u2705 Update manager initialized successfully\n2025-08-23T20:47:26.508603+0530 | INFO | \\u2705 Data collector initialized successfully\n2025-08-23T20:47:26.510197+0530 | INFO | \\u2705 Missing data detection: 22 missing dates\n2025-08-23T20:47:26.510489+0530 | INFO | Data Infrastructure Test Suite completed in 59.73 seconds\n2025-08-23T20:47:26.510662+0530 | INFO | Results: 11/25 tests passed (44.0%)\n",
    "status": "FAIL"
  },
  "llm_analysis": {
    "script_name": "llm_analysis",
    "script_path": "test_scripts/test_llm_analysis.py",
    "start_time": "2025-08-23T20:47:27.057173",
    "end_time": "2025-08-23T20:47:31.404500",
    "duration": 4.347147703170776,
    "exit_code": 1,
    "output": "",
    "error": "Traceback (most recent call last):\n  File \"C:\\sanjot\\virattt\\ai-hedge-fund\\tests\\comprehensive_test_suite\\test_scripts/test_llm_analysis.py\", line 19, in <module>\n    from src.agents import (\n    ...<4 lines>...\n    )\nImportError: cannot import name 'WarrenBuffettAgent' from 'src.agents' (C:\\sanjot\\virattt\\ai-hedge-fund\\tests\\comprehensive_test_suite\\test_scripts\\..\\..\\..\\src\\agents\\__init__.py)\n",
    "status": "FAIL"
  },
  "strategies": {
    "script_name": "strategies",
    "script_path": "test_scripts/test_strategies.py",
    "start_time": "2025-08-23T20:47:31.408123",
    "end_time": "2025-08-23T20:47:50.339031",
    "duration": 18.93074369430542,
    "exit_code": 1,
    "output": "",
    "error": "2025-08-23T20:47:45.608273+0530 | INFO | Strategy Tester initialized\n2025-08-23T20:47:45.608495+0530 | INFO | Starting Strategy Test Suite...\nFailed to initialize NSEUtility provider: HTTPConnectionPool(host='nseindia.com', port=80): Max retries exceeded with url: / (Caused by NameResolutionError(\"<urllib3.connection.HTTPConnection object at 0x000001621B870C20>: Failed to resolve 'nseindia.com' ([Errno 11001] getaddrinfo failed)\"))\nError fetching price data for AAPL: NSEUtility initialization failed: HTTPConnectionPool(host='nseindia.com', port=80): Max retries exceeded with url: / (Caused by NameResolutionError(\"<urllib3.connection.HTTPConnection object at 0x000001621B870C20>: Failed to resolve 'nseindia.com' ([Errno 11001] getaddrinfo failed)\"))\n2025-08-23T20:47:49.652644+0530 | ERROR | \\u274c Failed to load test data: 'list' object has no attribute 'empty'\n2025-08-23T20:47:49.652818+0530 | INFO | Testing intraday strategies...\n2025-08-23T20:47:49.652961+0530 | INFO | Testing Momentum Breakout strategy\n2025-08-23T20:47:49.653057+0530 | INFO | \\u2705 Momentum Breakout initialized successfully\nTraceback (most recent call last):\n  File \"C:\\sanjot\\virattt\\ai-hedge-fund\\tests\\comprehensive_test_suite\\test_scripts/test_strategies.py\", line 487, in <module>\n    success = main()\n  File \"C:\\sanjot\\virattt\\ai-hedge-fund\\tests\\comprehensive_test_suite\\test_scripts/test_strategies.py\", line 443, in main\n    results = tester.run_all_tests()\n  File \"C:\\sanjot\\virattt\\ai-hedge-fund\\tests\\comprehensive_test_suite\\test_scripts/test_strategies.py\", line 418, in run_all_tests\n    self.test_intraday_strategies()\n    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^\n  File \"C:\\sanjot\\virattt\\ai-hedge-fund\\tests\\comprehensive_test_suite\\test_scripts/test_strategies.py\", line 113, in test_intraday_strategies\n    if self.test_data is not None and not self.test_data.empty:\n                                          ^^^^^^^^^^^^^^^^^^^^\nAttributeError: 'list' object has no attribute 'empty'\n",
    "status": "FAIL"
  },
  "backtesting": {
    "script_name": "backtesting",
    "script_path": "test_scripts/test_backtesting.py",
    "start_time": "2025-08-23T20:47:50.340161",
    "end_time": "2025-08-23T20:48:00.881737",
    "duration": 10.541444301605225,
    "exit_code": 1,
    "output": "",
    "error": "Traceback (most recent call last):\n  File \"C:\\sanjot\\virattt\\ai-hedge-fund\\tests\\comprehensive_test_suite\\test_scripts/test_backtesting.py\", line 19, in <module>\n    from src.ml.backtesting import MLBacktester\nImportError: cannot import name 'MLBacktester' from 'src.ml.backtesting' (C:\\sanjot\\virattt\\ai-hedge-fund\\tests\\comprehensive_test_suite\\test_scripts\\..\\..\\..\\src\\ml\\backtesting.py)\n",
    "status": "FAIL"
  },
  "documentation": {
    "total_tests": 4,
    "passed": 0,
    "failed": 4,
    "warnings": 0,
    "details": {}
  },
  "github_integration": {
    "total_tests": 3,
    "passed": 1,
    "failed": 0,
    "warnings": 2,
    "details": {}
  },
  "summary": {
    "timestamp": "2025-08-23T20:48:01.225757",
    "total_tests": 11,
    "passed": 1,
    "failed": 8,
    "warnings": 2,
    "success_rate": 9.090909090909092,
    "status": "FAIL",
    "test_scripts": {
      "data_infrastructure": {
        "status": "FAIL",
        "duration": 67.50889682769775
      },
      "llm_analysis": {
        "status": "FAIL",
        "duration": 4.347147703170776
      },
      "strategies": {
        "status": "FAIL",
        "duration": 18.93074369430542
      },
      "backtesting": {
        "status": "FAIL",
        "duration": 10.541444301605225
      }
    },
    "additional_tests": {
      "documentation": {
        "tests": 4,
        "passed": 0,
        "success_rate": 0.0
      },
      "github_integration": {
        "tests": 3,
        "passed": 1,
        "success_rate": 33.33333333333333
      }
    }
  },
  "errors": [],
  "warnings": []
}
```
