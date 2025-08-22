#!/usr/bin/env python3
"""
LLM Analysis Test Suite
Tests LLM analysis functionality for both US and Indian stocks.
"""

import sys
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..', 'src'))

from loguru import logger
from src.tools.enhanced_api import get_prices, get_financial_metrics, get_market_cap
from src.agents import (
    WarrenBuffettAgent, PeterLynchAgent, PhilFisherAgent, BenjaminGrahamAgent,
    JohnTempletonAgent, GeorgeSorosAgent, RayDalioAgent, CharlieMungerAgent,
    SethKlarmanAgent, HowardMarksAgent, JoelGreenblattAgent, MohnishPabraiAgent,
    LiLuAgent, BillAckmanAgent, ValuationAnalystAgent
)


class LLMAnalysisTester:
    """Test LLM analysis capabilities."""
    
    def __init__(self):
        """Initialize the LLM analysis tester."""
        self.test_results = {
            'us_stock_analysis': {},
            'indian_stock_analysis': {},
            'agent_functionality': {},
            'data_integration': {},
            'summary': {},
            'errors': [],
            'warnings': []
        }
        
        # Test tickers
        self.us_tickers = ['AAPL', 'MSFT', 'GOOGL']
        self.indian_tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
        
        # Test agents
        self.agents = {
            'Warren Buffett': WarrenBuffettAgent(),
            'Peter Lynch': PeterLynchAgent(),
            'Phil Fisher': PhilFisherAgent(),
            'Benjamin Graham': BenjaminGrahamAgent(),
            'John Templeton': JohnTempletonAgent(),
            'George Soros': GeorgeSorosAgent(),
            'Ray Dalio': RayDalioAgent(),
            'Charlie Munger': CharlieMungerAgent(),
            'Seth Klarman': SethKlarmanAgent(),
            'Howard Marks': HowardMarksAgent(),
            'Joel Greenblatt': JoelGreenblattAgent(),
            'Mohnish Pabrai': MohnishPabraiAgent(),
            'Li Lu': LiLuAgent(),
            'Bill Ackman': BillAckmanAgent(),
            'Valuation Analyst': ValuationAnalystAgent()
        }
        
        logger.info("LLM Analysis Tester initialized")
    
    def test_agent_initialization(self) -> Dict[str, Any]:
        """Test agent initialization."""
        logger.info("Testing agent initialization...")
        
        results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': {}
        }
        
        for agent_name, agent in self.agents.items():
            logger.info(f"Testing {agent_name} agent initialization")
            
            try:
                if agent is not None:
                    results['passed'] += 1
                    logger.info(f"✅ {agent_name} agent initialized successfully")
                    
                    # Test basic agent attributes
                    if hasattr(agent, 'name'):
                        results['passed'] += 1
                        logger.info(f"✅ {agent_name} has name attribute")
                    else:
                        results['warnings'] += 1
                        logger.warning(f"⚠️ {agent_name} missing name attribute")
                    results['total_tests'] += 1
                    
                    if hasattr(agent, 'analyze_stock'):
                        results['passed'] += 1
                        logger.info(f"✅ {agent_name} has analyze_stock method")
                    else:
                        results['warnings'] += 1
                        logger.warning(f"⚠️ {agent_name} missing analyze_stock method")
                    results['total_tests'] += 1
                    
                else:
                    results['failed'] += 1
                    logger.error(f"❌ {agent_name} agent initialization failed")
                results['total_tests'] += 1
                
            except Exception as e:
                results['failed'] += 1
                results['total_tests'] += 1
                logger.error(f"❌ {agent_name} agent initialization error: {e}")
            
            results['details'][agent_name] = {
                'initialized': agent is not None,
                'has_name': hasattr(agent, 'name') if agent else False,
                'has_analyze_method': hasattr(agent, 'analyze_stock') if agent else False
            }
        
        self.test_results['agent_functionality'] = results
        return results
    
    def test_us_stock_analysis(self) -> Dict[str, Any]:
        """Test US stock analysis with LLM agents."""
        logger.info("Testing US stock analysis...")
        
        results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': {}
        }
        
        # Test with a subset of agents to avoid overwhelming the system
        test_agents = {
            'Warren Buffett': self.agents['Warren Buffett'],
            'Peter Lynch': self.agents['Peter Lynch'],
            'Valuation Analyst': self.agents['Valuation Analyst']
        }
        
        for ticker in self.us_tickers:
            logger.info(f"Testing US stock analysis for {ticker}")
            ticker_results = {
                'data_retrieval': False,
                'agent_analysis': {},
                'overall_success': False
            }
            
            # Test data retrieval
            try:
                prices = get_prices(ticker, '2023-01-01', '2023-12-31')
                if prices is not None and not prices.empty:
                    ticker_results['data_retrieval'] = True
                    results['passed'] += 1
                    logger.info(f"✅ Data retrieved for {ticker}: {len(prices)} records")
                else:
                    results['warnings'] += 1
                    logger.warning(f"⚠️ No data for {ticker}")
                results['total_tests'] += 1
            except Exception as e:
                results['warnings'] += 1
                results['total_tests'] += 1
                logger.warning(f"⚠️ Data retrieval warning for {ticker}: {e}")
            
            # Test agent analysis (if data available)
            if ticker_results['data_retrieval']:
                for agent_name, agent in test_agents.items():
                    logger.info(f"Testing {agent_name} analysis for {ticker}")
                    
                    try:
                        analysis = agent.analyze_stock(ticker)
                        if analysis and isinstance(analysis, dict):
                            ticker_results['agent_analysis'][agent_name] = True
                            results['passed'] += 1
                            logger.info(f"✅ {agent_name} analysis successful for {ticker}")
                        else:
                            ticker_results['agent_analysis'][agent_name] = False
                            results['warnings'] += 1
                            logger.warning(f"⚠️ {agent_name} analysis returned invalid format for {ticker}")
                        results['total_tests'] += 1
                    except Exception as e:
                        ticker_results['agent_analysis'][agent_name] = False
                        results['warnings'] += 1
                        results['total_tests'] += 1
                        logger.warning(f"⚠️ {agent_name} analysis warning for {ticker}: {e}")
                
                # Check overall success
                successful_analyses = sum(ticker_results['agent_analysis'].values())
                if successful_analyses > 0:
                    ticker_results['overall_success'] = True
                    results['passed'] += 1
                    logger.info(f"✅ Overall analysis successful for {ticker}")
                else:
                    results['warnings'] += 1
                    logger.warning(f"⚠️ No successful analyses for {ticker}")
                results['total_tests'] += 1
            
            results['details'][ticker] = ticker_results
        
        self.test_results['us_stock_analysis'] = results
        return results
    
    def test_indian_stock_analysis(self) -> Dict[str, Any]:
        """Test Indian stock analysis with LLM agents."""
        logger.info("Testing Indian stock analysis...")
        
        results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': {}
        }
        
        # Test with a subset of agents
        test_agents = {
            'Warren Buffett': self.agents['Warren Buffett'],
            'Peter Lynch': self.agents['Peter Lynch'],
            'Valuation Analyst': self.agents['Valuation Analyst']
        }
        
        for ticker in self.indian_tickers:
            logger.info(f"Testing Indian stock analysis for {ticker}")
            ticker_results = {
                'data_retrieval': False,
                'agent_analysis': {},
                'overall_success': False
            }
            
            # Test data retrieval
            try:
                prices = get_prices(ticker, '2023-01-01', '2023-12-31')
                if prices is not None and not prices.empty:
                    ticker_results['data_retrieval'] = True
                    results['passed'] += 1
                    logger.info(f"✅ Data retrieved for {ticker}: {len(prices)} records")
                else:
                    results['warnings'] += 1
                    logger.warning(f"⚠️ No data for {ticker} (may be expected)")
                results['total_tests'] += 1
            except Exception as e:
                results['warnings'] += 1
                results['total_tests'] += 1
                logger.warning(f"⚠️ Data retrieval warning for {ticker}: {e}")
            
            # Test agent analysis (if data available)
            if ticker_results['data_retrieval']:
                for agent_name, agent in test_agents.items():
                    logger.info(f"Testing {agent_name} analysis for {ticker}")
                    
                    try:
                        analysis = agent.analyze_stock(ticker)
                        if analysis and isinstance(analysis, dict):
                            ticker_results['agent_analysis'][agent_name] = True
                            results['passed'] += 1
                            logger.info(f"✅ {agent_name} analysis successful for {ticker}")
                        else:
                            ticker_results['agent_analysis'][agent_name] = False
                            results['warnings'] += 1
                            logger.warning(f"⚠️ {agent_name} analysis returned invalid format for {ticker}")
                        results['total_tests'] += 1
                    except Exception as e:
                        ticker_results['agent_analysis'][agent_name] = False
                        results['warnings'] += 1
                        results['total_tests'] += 1
                        logger.warning(f"⚠️ {agent_name} analysis warning for {ticker}: {e}")
                
                # Check overall success
                successful_analyses = sum(ticker_results['agent_analysis'].values())
                if successful_analyses > 0:
                    ticker_results['overall_success'] = True
                    results['passed'] += 1
                    logger.info(f"✅ Overall analysis successful for {ticker}")
                else:
                    results['warnings'] += 1
                    logger.warning(f"⚠️ No successful analyses for {ticker}")
                results['total_tests'] += 1
            
            results['details'][ticker] = ticker_results
        
        self.test_results['indian_stock_analysis'] = results
        return results
    
    def test_data_integration(self) -> Dict[str, Any]:
        """Test data integration with LLM analysis."""
        logger.info("Testing data integration...")
        
        results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': {}
        }
        
        test_ticker = 'AAPL'
        
        # Test financial metrics retrieval
        try:
            metrics = get_financial_metrics(test_ticker, '2023-12-31')
            if metrics is not None:
                results['passed'] += 1
                logger.info(f"✅ Financial metrics retrieved for {test_ticker}")
            else:
                results['warnings'] += 1
                logger.warning(f"⚠️ No financial metrics for {test_ticker}")
            results['total_tests'] += 1
        except Exception as e:
            results['warnings'] += 1
            results['total_tests'] += 1
            logger.warning(f"⚠️ Financial metrics warning for {test_ticker}: {e}")
        
        # Test market cap retrieval
        try:
            market_cap = get_market_cap(test_ticker, '2023-12-31')
            if market_cap is not None:
                results['passed'] += 1
                logger.info(f"✅ Market cap retrieved for {test_ticker}: {market_cap}")
            else:
                results['warnings'] += 1
                logger.warning(f"⚠️ No market cap for {test_ticker}")
            results['total_tests'] += 1
        except Exception as e:
            results['warnings'] += 1
            results['total_tests'] += 1
            logger.warning(f"⚠️ Market cap warning for {test_ticker}: {e}")
        
        # Test agent analysis with enhanced data
        try:
            agent = self.agents['Valuation Analyst']
            analysis = agent.analyze_stock(test_ticker)
            if analysis and isinstance(analysis, dict):
                results['passed'] += 1
                logger.info(f"✅ Enhanced analysis successful for {test_ticker}")
            else:
                results['warnings'] += 1
                logger.warning(f"⚠️ Enhanced analysis returned invalid format for {test_ticker}")
            results['total_tests'] += 1
        except Exception as e:
            results['warnings'] += 1
            results['total_tests'] += 1
            logger.warning(f"⚠️ Enhanced analysis warning for {test_ticker}: {e}")
        
        self.test_results['data_integration'] = results
        return results
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate test summary."""
        agent_results = self.test_results['agent_functionality']
        us_results = self.test_results['us_stock_analysis']
        indian_results = self.test_results['indian_stock_analysis']
        integration_results = self.test_results['data_integration']
        
        total_tests = (
            agent_results.get('total_tests', 0) +
            us_results.get('total_tests', 0) +
            indian_results.get('total_tests', 0) +
            integration_results.get('total_tests', 0)
        )
        
        total_passed = (
            agent_results.get('passed', 0) +
            us_results.get('passed', 0) +
            indian_results.get('passed', 0) +
            integration_results.get('passed', 0)
        )
        
        total_failed = (
            agent_results.get('failed', 0) +
            us_results.get('failed', 0) +
            indian_results.get('failed', 0) +
            integration_results.get('failed', 0)
        )
        
        total_warnings = (
            agent_results.get('warnings', 0) +
            us_results.get('warnings', 0) +
            indian_results.get('warnings', 0) +
            integration_results.get('warnings', 0)
        )
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed': total_passed,
            'failed': total_failed,
            'warnings': total_warnings,
            'success_rate': success_rate,
            'status': 'PASS' if success_rate >= 70 else 'FAIL',
            'categories': {
                'agent_functionality': {
                    'tests': agent_results.get('total_tests', 0),
                    'passed': agent_results.get('passed', 0),
                    'success_rate': (agent_results.get('passed', 0) / agent_results.get('total_tests', 1) * 100) if agent_results.get('total_tests', 0) > 0 else 0
                },
                'us_stock_analysis': {
                    'tests': us_results.get('total_tests', 0),
                    'passed': us_results.get('passed', 0),
                    'success_rate': (us_results.get('passed', 0) / us_results.get('total_tests', 1) * 100) if us_results.get('total_tests', 0) > 0 else 0
                },
                'indian_stock_analysis': {
                    'tests': indian_results.get('total_tests', 0),
                    'passed': indian_results.get('passed', 0),
                    'success_rate': (indian_results.get('passed', 0) / indian_results.get('total_tests', 1) * 100) if indian_results.get('total_tests', 0) > 0 else 0
                },
                'data_integration': {
                    'tests': integration_results.get('total_tests', 0),
                    'passed': integration_results.get('passed', 0),
                    'success_rate': (integration_results.get('passed', 0) / integration_results.get('total_tests', 1) * 100) if integration_results.get('total_tests', 0) > 0 else 0
                }
            }
        }
        
        self.test_results['summary'] = summary
        return summary
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all LLM analysis tests."""
        logger.info("Starting LLM Analysis Test Suite...")
        start_time = time.time()
        
        # Run tests
        self.test_agent_initialization()
        self.test_us_stock_analysis()
        self.test_indian_stock_analysis()
        self.test_data_integration()
        
        # Generate summary
        summary = self.generate_summary()
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"LLM Analysis Test Suite completed in {duration:.2f} seconds")
        logger.info(f"Results: {summary['passed']}/{summary['total_tests']} tests passed ({summary['success_rate']:.1f}%)")
        
        return self.test_results


def main():
    """Main test execution function."""
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")
    
    # Run tests
    tester = LLMAnalysisTester()
    results = tester.run_all_tests()
    
    # Print summary
    summary = results['summary']
    print("\n" + "="*60)
    print("LLM Analysis Test Results")
    print("="*60)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Status: {summary['status']}")
    print("="*60)
    
    # Save results
    results_file = os.path.join(os.path.dirname(__file__), '../test_results/llm_analysis_results.md')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write(f"# LLM Analysis Test Results\n\n")
        f.write(f"**Timestamp**: {summary['timestamp']}\n")
        f.write(f"**Status**: {summary['status']}\n")
        f.write(f"**Success Rate**: {summary['success_rate']:.1f}%\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- Total Tests: {summary['total_tests']}\n")
        f.write(f"- Passed: {summary['passed']}\n")
        f.write(f"- Failed: {summary['failed']}\n")
        f.write(f"- Warnings: {summary['warnings']}\n\n")
        f.write(f"## Category Results\n\n")
        for category, stats in summary['categories'].items():
            f.write(f"### {category.replace('_', ' ').title()}\n")
            f.write(f"- Tests: {stats['tests']}\n")
            f.write(f"- Passed: {stats['passed']}\n")
            f.write(f"- Success Rate: {stats['success_rate']:.1f}%\n\n")
        f.write(f"## Detailed Results\n\n")
        f.write(f"```json\n{json.dumps(results, indent=2)}\n```\n")
    
    print(f"Results saved to: {results_file}")
    
    return summary['status'] == 'PASS'


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 