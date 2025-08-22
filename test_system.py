#!/usr/bin/env python3
"""
Comprehensive System Test for AI Hedge Fund with Indian Market Integration
"""

import sys
import traceback
from datetime import datetime, timedelta

def test_imports():
    """Test all critical imports."""
    print("🧪 Testing Imports...")
    
    try:
        from src.tools.enhanced_api import get_prices, get_financial_metrics
        print("✅ Enhanced API imported successfully")
    except Exception as e:
        print(f"❌ Enhanced API import failed: {e}")
        return False
    
    try:
        from src.strategies.strategy_manager import get_strategy_summary
        print("✅ Strategy Manager imported successfully")
    except Exception as e:
        print(f"❌ Strategy Manager import failed: {e}")
        return False
    
    try:
        from src.data.providers.provider_factory import get_provider_factory
        print("✅ Provider Factory imported successfully")
    except Exception as e:
        print(f"❌ Provider Factory import failed: {e}")
        return False
    
    return True

def test_strategy_framework():
    """Test strategy framework."""
    print("\n🧪 Testing Strategy Framework...")
    
    try:
        from src.strategies.strategy_manager import get_strategy_summary
        summary = get_strategy_summary()
        
        print(f"✅ Total Strategies: {summary['total_strategies']}")
        print(f"✅ Active Strategies: {summary['active_strategies']}")
        print(f"✅ Categories: {summary['categories']}")
        
        return True
    except Exception as e:
        print(f"❌ Strategy Framework test failed: {e}")
        return False

def test_data_providers():
    """Test data providers."""
    print("\n🧪 Testing Data Providers...")
    
    try:
        from src.data.providers.provider_factory import get_provider_factory
        factory = get_provider_factory()
        
        print(f"✅ Provider Factory initialized with {len(factory.providers)} providers")
        
        # Test NSEUtility provider
        try:
            nse_provider = factory.get_nse_utility_provider()
            print("✅ NSEUtility Provider available")
        except Exception as e:
            print(f"⚠️ NSEUtility Provider: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Data Providers test failed: {e}")
        return False

def test_enhanced_api():
    """Test enhanced API functionality."""
    print("\n🧪 Testing Enhanced API...")
    
    try:
        from src.tools.enhanced_api import get_prices, get_financial_metrics
        
        # Test with a US stock (should work)
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        prices = get_prices('AAPL', start_date, end_date)
        print(f"✅ US Stock Test: {len(prices)} price records for AAPL")
        
        # Test with Indian stock
        try:
            prices = get_prices('RELIANCE.NS', start_date, end_date)
            print(f"✅ Indian Stock Test: {len(prices)} price records for RELIANCE.NS")
        except Exception as e:
            print(f"⚠️ Indian Stock Test: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Enhanced API test failed: {e}")
        return False

def test_nse_utility():
    """Test NSEUtility integration."""
    print("\n🧪 Testing NSEUtility...")
    
    try:
        from src.nsedata.NseUtility import NseUtils
        nse = NseUtils()
        
        # Test basic functionality
        info = nse.get_quote('RELIANCE')
        if info and 'lastPrice' in info:
            print(f"✅ NSEUtility: RELIANCE price ₹{info['lastPrice']}")
            return True
        else:
            print("⚠️ NSEUtility: No price data returned")
            return False
            
    except Exception as e:
        print(f"⚠️ NSEUtility test failed: {e}")
        return False

def test_ai_agents():
    """Test AI agents."""
    print("\n🧪 Testing AI Agents...")
    
    try:
        # Test a few key agents
        from src.agents.warren_buffett import warren_buffett_agent
        from src.agents.phil_fisher import phil_fisher_agent
        
        print("✅ Warren Buffett Agent imported")
        print("✅ Phil Fisher Agent imported")
        
        return True
    except Exception as e:
        print(f"❌ AI Agents test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 AI Hedge Fund System Test")
    print("=" * 50)
    print(f"Test started at: {datetime.now()}")
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Strategy Framework", test_strategy_framework),
        ("Data Providers", test_data_providers),
        ("Enhanced API", test_enhanced_api),
        ("NSEUtility", test_nse_utility),
        ("AI Agents", test_ai_agents),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    print(f"Test completed at: {datetime.now()}")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 