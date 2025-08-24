# Data Infrastructure Test Results

**Timestamp**: 2025-08-23T20:50:06.605482
**Status**: FAIL
**Success Rate**: 48.0%

## Summary

- Total Tests: 25
- Passed: 12
- Failed: 6
- Warnings: 7

## Category Results

### Data Retrieval
- Tests: 18
- Passed: 9
- Success Rate: 50.0%

### Data Storage
- Tests: 3
- Passed: 0
- Success Rate: 0.0%

### Data Quality
- Tests: 1
- Passed: 0
- Success Rate: 0.0%

### Update Mechanisms
- Tests: 3
- Passed: 3
- Success Rate: 100.0%

## Detailed Results

```json
{
  "data_retrieval": {
    "total_tests": 18,
    "passed": 9,
    "failed": 3,
    "warnings": 6,
    "details": {
      "AAPL": {
        "prices": false,
        "financial_metrics": true,
        "market_cap": false
      },
      "MSFT": {
        "prices": false,
        "financial_metrics": true,
        "market_cap": false
      },
      "GOOGL": {
        "prices": false,
        "financial_metrics": true,
        "market_cap": false
      },
      "RELIANCE.NS": {
        "prices": true,
        "financial_metrics": true,
        "market_cap": false
      },
      "TCS.NS": {
        "prices": true,
        "financial_metrics": true,
        "market_cap": false
      },
      "INFY.NS": {
        "prices": true,
        "financial_metrics": true,
        "market_cap": false
      }
    }
  },
  "data_storage": {
    "total_tests": 3,
    "passed": 0,
    "failed": 2,
    "warnings": 1,
    "details": {}
  },
  "data_quality": {
    "total_tests": 1,
    "passed": 0,
    "failed": 1,
    "warnings": 0,
    "details": {}
  },
  "update_mechanisms": {
    "total_tests": 3,
    "passed": 3,
    "failed": 0,
    "warnings": 0,
    "details": {}
  },
  "summary": {
    "timestamp": "2025-08-23T20:50:06.605482",
    "total_tests": 25,
    "passed": 12,
    "failed": 6,
    "warnings": 7,
    "success_rate": 48.0,
    "status": "FAIL",
    "categories": {
      "data_retrieval": {
        "tests": 18,
        "passed": 9,
        "success_rate": 50.0
      },
      "data_storage": {
        "tests": 3,
        "passed": 0,
        "success_rate": 0.0
      },
      "data_quality": {
        "tests": 1,
        "passed": 0,
        "success_rate": 0.0
      },
      "update_mechanisms": {
        "tests": 3,
        "passed": 3,
        "success_rate": 100.0
      }
    }
  },
  "errors": [],
  "warnings": []
}
```
