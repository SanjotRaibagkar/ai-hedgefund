# Detailed Test Instructions

## Test Execution Order

### 1. LLM Analysis Tests
**Purpose**: Verify LLM-powered stock analysis for both US and Indian stocks

**Test Script**: `test_llm_analysis.py`

**What it tests**:
- US stock analysis using various AI agents
- Indian stock analysis with market-specific considerations
- Error handling for API failures
- Response quality and format validation

**Expected Duration**: 3-5 minutes

**Success Criteria**:
- ✅ All AI agents can analyze US stocks
- ✅ All AI agents can analyze Indian stocks
- ✅ Proper error handling for API failures
- ✅ Response format is consistent

### 2. Data Infrastructure Tests
**Purpose**: Verify data retrieval and storage capabilities

**Test Script**: `test_data_infrastructure.py`

**What it tests**:
- Indian data retrieval from multiple sources
- Data storage in SQLite database
- Data quality and validation
- Daily update mechanisms

**Expected Duration**: 2-3 minutes

**Success Criteria**:
- ✅ Data can be retrieved from all sources
- ✅ Data is properly stored in database
- ✅ Data quality checks pass
- ✅ Update mechanisms work correctly

### 3. Strategy Tests
**Purpose**: Verify all trading strategies are functional

**Test Script**: `test_strategies.py`

**What it tests**:
- Intraday strategies (5 strategies)
- Options strategies (5 strategies)
- EOD momentum strategies (2 strategies)
- ML-enhanced strategies (Phase 4)

**Expected Duration**: 5-8 minutes

**Success Criteria**:
- ✅ All strategies can be initialized
- ✅ All strategies can analyze stocks
- ✅ Strategy outputs are valid
- ✅ ML strategies can train and predict

### 4. Backtesting Tests
**Purpose**: Verify backtesting framework functionality

**Test Script**: `test_backtesting.py`

**What it tests**:
- Backtesting for traditional strategies
- Backtesting for ML-enhanced strategies
- Performance metrics calculation
- Report generation

**Expected Duration**: 3-5 minutes

**Success Criteria**:
- ✅ Backtesting can run for all strategies
- ✅ Performance metrics are calculated correctly
- ✅ Reports are generated properly
- ✅ No critical errors during backtesting

### 5. Integration Tests
**Purpose**: Verify end-to-end system functionality

**Test Script**: `test_integration.py`

**What it tests**:
- Complete workflows from data to analysis
- System performance under load
- Error recovery mechanisms
- Cross-component integration

**Expected Duration**: 2-3 minutes

**Success Criteria**:
- ✅ End-to-end workflows complete successfully
- ✅ System performance is acceptable
- ✅ Error recovery works properly
- ✅ All components integrate correctly

## Test Execution Commands

### Individual Test Execution
```bash
# LLM Analysis Tests
python test_scripts/test_llm_analysis.py

# Data Infrastructure Tests
python test_scripts/test_data_infrastructure.py

# Strategy Tests
python test_scripts/test_strategies.py

# Backtesting Tests
python test_scripts/test_backtesting.py

# Integration Tests
python test_scripts/test_integration.py
```

### Full Test Suite Execution
```bash
# Run all tests
python run_all_tests.py

# Run with verbose output
python run_all_tests.py --verbose

# Run with specific test categories
python run_all_tests.py --categories llm,data,strategies
```

## Test Data

### Test Tickers
- **US Stocks**: AAPL, MSFT, GOOGL
- **Indian Stocks**: RELIANCE.NS, TCS.NS, INFY.NS

### Test Date Ranges
- **Historical Data**: 2023-01-01 to 2023-12-31
- **Recent Data**: Last 30 days
- **Real-time Data**: Current market data

### Test Parameters
- **Data Points**: Minimum 100 data points per ticker
- **Strategies**: All available strategies
- **Timeouts**: 30 seconds per API call
- **Retries**: 3 attempts for failed calls

## Error Handling

### Expected Errors (Non-Critical)
- API rate limiting
- Network timeouts
- Data source unavailability
- Missing historical data

### Critical Errors (Test Failures)
- System initialization failures
- Database connection errors
- Strategy execution crashes
- Data corruption

### Error Recovery
- Automatic retry for transient failures
- Fallback to alternative data sources
- Graceful degradation for missing features
- Detailed error logging

## Performance Benchmarks

### Response Times
- **API Calls**: < 5 seconds
- **Data Processing**: < 30 seconds for 1 year
- **Strategy Analysis**: < 10 seconds per strategy
- **Backtesting**: < 60 seconds per strategy

### Resource Usage
- **Memory**: < 500MB for full test suite
- **CPU**: < 50% average usage
- **Disk**: < 100MB temporary files
- **Network**: < 50MB data transfer

## Test Validation

### Data Validation
- Check data completeness
- Verify data format consistency
- Validate data ranges and types
- Confirm data source attribution

### Strategy Validation
- Verify strategy initialization
- Check strategy parameter validation
- Test strategy output formats
- Validate strategy logic

### Integration Validation
- Test component interactions
- Verify data flow between components
- Check error propagation
- Validate system state consistency

## Reporting

### Test Reports
- Summary of all test results
- Detailed logs for failed tests
- Performance metrics
- Recommendations for improvements

### Result Storage
- Results saved in `test_results/` directory
- Timestamped result files
- Historical result tracking
- Performance trend analysis

## Maintenance

### Regular Updates
- Update test data periodically
- Refresh test parameters
- Add tests for new features
- Remove obsolete tests

### Performance Monitoring
- Track test execution times
- Monitor resource usage
- Identify performance bottlenecks
- Optimize slow tests

### Quality Assurance
- Review test coverage
- Validate test accuracy
- Update test documentation
- Train team on test procedures 