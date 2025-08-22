# Comprehensive Test Suite

This directory contains a comprehensive test suite for the AI Hedge Fund system, designed to verify all functionality from data infrastructure to advanced ML strategies.

## Overview

The comprehensive test suite is organized into multiple test scripts that cover different aspects of the system:

1. **Data Infrastructure Testing** - Tests data retrieval, storage, quality, and update mechanisms
2. **LLM Analysis Testing** - Tests LLM agent functionality for both US and Indian stocks
3. **Strategy Testing** - Tests all trading strategies (intraday, options, EOD, ML)
4. **Backtesting Testing** - Tests backtesting functionality and performance metrics
5. **Documentation Testing** - Verifies documentation completeness
6. **GitHub Integration Testing** - Checks code repository status

## Directory Structure

```
tests/comprehensive_test_suite/
├── README.md                           # This file
├── run_comprehensive_tests.py          # Main test runner
├── test_scripts/                       # Individual test scripts
│   ├── data_infrastructure_test.py     # Data infrastructure tests
│   ├── test_llm_analysis.py           # LLM analysis tests
│   ├── test_strategies.py             # Strategy tests
│   └── test_backtesting.py            # Backtesting tests
└── test_results/                       # Test results (generated)
    ├── comprehensive_results.md        # Overall results
    ├── data_infrastructure_results.md  # Data infrastructure results
    ├── llm_analysis_results.md        # LLM analysis results
    ├── strategy_results.md            # Strategy results
    └── backtesting_results.md         # Backtesting results
```

## Prerequisites

Before running the test suite, ensure you have:

1. **Python Environment**: Python 3.9+ with all dependencies installed
2. **Dependencies**: Run `poetry install --with ml` to install all required packages
3. **Data Access**: Internet connection for data retrieval tests
4. **Git Repository**: The system should be in a git repository with proper remote configuration

## Running the Tests

### Option 1: Run All Tests (Recommended)

To run the complete comprehensive test suite:

```bash
cd tests/comprehensive_test_suite
python run_comprehensive_tests.py
```

This will:
- Run all individual test scripts
- Generate detailed reports for each test category
- Create a comprehensive summary report
- Save all results to the `test_results/` directory

### Option 2: Run Individual Tests

You can also run individual test scripts:

```bash
# Data infrastructure tests
python test_scripts/data_infrastructure_test.py

# LLM analysis tests
python test_scripts/test_llm_analysis.py

# Strategy tests
python test_scripts/test_strategies.py

# Backtesting tests
python test_scripts/test_backtesting.py
```

## Test Categories

### 1. Data Infrastructure Tests

**Purpose**: Verify data retrieval, storage, quality, and update mechanisms

**Tests Include**:
- US stock data retrieval (AAPL, MSFT, GOOGL)
- Indian stock data retrieval (RELIANCE.NS, TCS.NS, INFY.NS)
- Database initialization and table creation
- Data insertion and retrieval
- Data quality validation
- Update mechanisms and missing data detection

**Expected Results**: 
- Success rate: ≥70%
- All core data functionality should work
- Some warnings expected for Indian data (availability dependent)

### 2. LLM Analysis Tests

**Purpose**: Test LLM agent functionality for stock analysis

**Tests Include**:
- Agent initialization (15 different agents)
- US stock analysis with key agents
- Indian stock analysis with key agents
- Data integration with LLM analysis
- Agent method validation

**Expected Results**:
- Success rate: ≥70%
- All agents should initialize properly
- Analysis results should be in expected format
- Some warnings expected for data availability

### 3. Strategy Tests

**Purpose**: Test all trading strategies

**Tests Include**:
- Intraday strategies (5 strategies)
- Options strategies (5 strategies)
- EOD momentum strategies (3 strategies)
- ML-enhanced strategies
- Strategy manager functionality

**Expected Results**:
- Success rate: ≥80%
- All strategies should initialize properly
- Strategy analysis should return valid results
- Strategy manager should provide summary information

### 4. Backtesting Tests

**Purpose**: Test backtesting functionality and performance metrics

**Tests Include**:
- ML backtesting execution
- Strategy backtesting
- MLflow integration
- Performance metrics calculation
- Backtesting configuration validation

**Expected Results**:
- Success rate: ≥75%
- Backtesting should execute without errors
- Performance metrics should be calculated correctly
- MLflow integration should work properly

### 5. Documentation Tests

**Purpose**: Verify documentation completeness

**Tests Include**:
- Required documentation files (README.md, USAGE.md, etc.)
- Source code package structure
- Documentation file existence

**Expected Results**:
- Success rate: ≥90%
- All required documentation should be present
- Package structure should be properly organized

### 6. GitHub Integration Tests

**Purpose**: Check code repository status

**Tests Include**:
- Git availability
- Repository status
- Remote origin configuration

**Expected Results**:
- Success rate: ≥80%
- Git should be available
- Repository should be properly configured

## Understanding Test Results

### Success Criteria

- **PASS**: Success rate ≥80% for comprehensive suite
- **WARNING**: Success rate 60-79%
- **FAIL**: Success rate <60%

### Result Categories

- **✅ PASSED**: Test completed successfully
- **❌ FAILED**: Test failed with errors
- **⚠️ WARNING**: Test completed with warnings or partial success

### Result Files

Each test generates detailed results in markdown format:

1. **comprehensive_results.md**: Overall summary and all test results
2. **data_infrastructure_results.md**: Data infrastructure test details
3. **llm_analysis_results.md**: LLM analysis test details
4. **strategy_results.md**: Strategy test details
5. **backtesting_results.md**: Backtesting test details

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `poetry install --with ml`
2. **Data Retrieval Failures**: Check internet connection and API availability
3. **MLflow Errors**: Ensure MLflow is properly configured
4. **Git Errors**: Verify git repository configuration

### Debug Mode

To run tests with more verbose output:

```bash
# Set debug logging
export LOG_LEVEL=DEBUG
python run_comprehensive_tests.py
```

### Individual Test Debugging

If a specific test fails, you can run it individually with debug output:

```bash
python -u test_scripts/data_infrastructure_test.py 2>&1 | tee debug_output.log
```

## Test Data

The tests use the following data sources:

- **US Stocks**: AAPL, MSFT, GOOGL (reliable data sources)
- **Indian Stocks**: RELIANCE.NS, TCS.NS, INFY.NS (availability dependent)
- **Test Period**: 2022-2023 for backtesting, 2023 for other tests

## Performance Expectations

- **Total Runtime**: 5-15 minutes depending on data availability
- **Individual Scripts**: 1-3 minutes each
- **Memory Usage**: <500MB
- **Network Usage**: Moderate (data retrieval)

## Continuous Integration

The test suite is designed to be run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Comprehensive Tests
  run: |
    cd tests/comprehensive_test_suite
    python run_comprehensive_tests.py
```

## Contributing

When adding new functionality to the system:

1. **Add Tests**: Create corresponding test cases in the appropriate test script
2. **Update Documentation**: Ensure documentation reflects new features
3. **Run Suite**: Verify all tests pass before committing
4. **Update This README**: Add information about new test categories

## Support

For issues with the test suite:

1. Check the troubleshooting section above
2. Review individual test result files for specific error details
3. Run tests in debug mode for more information
4. Check system logs for additional context

## Version History

- **v1.0**: Initial comprehensive test suite
- **v1.1**: Added ML strategy testing
- **v1.2**: Enhanced backtesting validation
- **v1.3**: Added documentation and GitHub integration tests

---

**Last Updated**: December 2024
**Test Suite Version**: 1.3
**Compatible System Version**: Phase 4 Complete 