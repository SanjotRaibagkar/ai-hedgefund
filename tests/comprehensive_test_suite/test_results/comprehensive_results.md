# Comprehensive Test Results

**Timestamp**: 2025-08-22T16:34:55.076410
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
- Duration: 0.65s

### Llm Analysis
- Status: FAIL
- Duration: 0.65s

### Strategies
- Status: FAIL
- Duration: 0.66s

### Backtesting
- Status: FAIL
- Duration: 0.63s

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
    "start_time": "2025-08-22T16:34:52.263619",
    "end_time": "2025-08-22T16:34:52.918063",
    "duration": 0.6543078422546387,
    "exit_code": 1,
    "output": "",
    "error": "Traceback (most recent call last):\n  File \"C:\\sanjot\\virattt\\ai-hedge-fund\\tests\\comprehensive_test_suite\\test_scripts/test_data_infrastructure.py\", line 18, in <module>\n    from src.tools.enhanced_api import get_prices, get_financial_metrics, get_market_cap\nModuleNotFoundError: No module named 'src'\n",
    "status": "FAIL"
  },
  "llm_analysis": {
    "script_name": "llm_analysis",
    "script_path": "test_scripts/test_llm_analysis.py",
    "start_time": "2025-08-22T16:34:52.919798",
    "end_time": "2025-08-22T16:34:53.572511",
    "duration": 0.6525158882141113,
    "exit_code": 1,
    "output": "",
    "error": "Traceback (most recent call last):\n  File \"C:\\sanjot\\virattt\\ai-hedge-fund\\tests\\comprehensive_test_suite\\test_scripts/test_llm_analysis.py\", line 18, in <module>\n    from src.tools.enhanced_api import get_prices, get_financial_metrics, get_market_cap\nModuleNotFoundError: No module named 'src'\n",
    "status": "FAIL"
  },
  "strategies": {
    "script_name": "strategies",
    "script_path": "test_scripts/test_strategies.py",
    "start_time": "2025-08-22T16:34:53.573792",
    "end_time": "2025-08-22T16:34:54.232789",
    "duration": 0.6588420867919922,
    "exit_code": 1,
    "output": "",
    "error": "Traceback (most recent call last):\n  File \"C:\\sanjot\\virattt\\ai-hedge-fund\\tests\\comprehensive_test_suite\\test_scripts/test_strategies.py\", line 18, in <module>\n    from src.tools.enhanced_api import get_prices\nModuleNotFoundError: No module named 'src'\n",
    "status": "FAIL"
  },
  "backtesting": {
    "script_name": "backtesting",
    "script_path": "test_scripts/test_backtesting.py",
    "start_time": "2025-08-22T16:34:54.233739",
    "end_time": "2025-08-22T16:34:54.866971",
    "duration": 0.6330838203430176,
    "exit_code": 1,
    "output": "",
    "error": "Traceback (most recent call last):\n  File \"C:\\sanjot\\virattt\\ai-hedge-fund\\tests\\comprehensive_test_suite\\test_scripts/test_backtesting.py\", line 18, in <module>\n    from src.tools.enhanced_api import get_prices\nModuleNotFoundError: No module named 'src'\n",
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
    "timestamp": "2025-08-22T16:34:55.076410",
    "total_tests": 11,
    "passed": 1,
    "failed": 8,
    "warnings": 2,
    "success_rate": 9.090909090909092,
    "status": "FAIL",
    "test_scripts": {
      "data_infrastructure": {
        "status": "FAIL",
        "duration": 0.6543078422546387
      },
      "llm_analysis": {
        "status": "FAIL",
        "duration": 0.6525158882141113
      },
      "strategies": {
        "status": "FAIL",
        "duration": 0.6588420867919922
      },
      "backtesting": {
        "status": "FAIL",
        "duration": 0.6330838203430176
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
