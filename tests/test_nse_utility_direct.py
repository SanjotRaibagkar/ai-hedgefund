#!/usr/bin/env python3
"""
Direct NSEUtility Test
Tests the NSEUtility module directly to identify issues.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from loguru import logger

def test_nse_utility_import():
    """Test if NSEUtility can be imported."""
    try:
        from src.nsedata.NseUtility import NseUtils
        logger.info("✅ NSEUtility import successful")
        return True
    except Exception as e:
        logger.error(f"❌ NSEUtility import failed: {e}")
        return False

def test_nse_utility_initialization():
    """Test if NSEUtility can be initialized."""
    try:
        from src.nsedata.NseUtility import NseUtils
        nse = NseUtils()
        logger.info("✅ NSEUtility initialization successful")
        return nse
    except Exception as e:
        logger.error(f"❌ NSEUtility initialization failed: {e}")
        return None

def test_nse_utility_price_info():
    """Test NSEUtility price info functionality."""
    try:
        from src.nsedata.NseUtility import NseUtils
        nse = NseUtils()
        
        # Test with a known Indian stock
        test_stocks = ['RELIANCE', 'TCS', 'HDFCBANK']
        
        for stock in test_stocks:
            try:
                logger.info(f"Testing price info for {stock}...")
                price_info = nse.price_info(stock)
                if price_info:
                    logger.info(f"✅ {stock}: Price info available")
                    logger.info(f"   Data: {price_info}")
                else:
                    logger.warning(f"⚠️ {stock}: No price info")
            except Exception as e:
                logger.error(f"❌ {stock}: Error - {e}")
                
        return True
        
    except Exception as e:
        logger.error(f"❌ NSEUtility price info test failed: {e}")
        return False

def test_nse_utility_equity_info():
    """Test NSEUtility equity info functionality."""
    try:
        from src.nsedata.NseUtility import NseUtils
        nse = NseUtils()
        
        # Test with a known Indian stock
        test_stock = 'RELIANCE'
        
        logger.info(f"Testing equity info for {test_stock}...")
        equity_info = nse.equity_info(test_stock)
        if equity_info:
            logger.info(f"✅ {test_stock}: Equity info available")
            logger.info(f"   Keys: {list(equity_info.keys())}")
        else:
            logger.warning(f"⚠️ {test_stock}: No equity info")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ NSEUtility equity info test failed: {e}")
        return False

def main():
    """Main test execution."""
    logger.info("🔧 Testing NSEUtility Directly")
    logger.info("=" * 40)
    
    tests = [
        ("Import Test", test_nse_utility_import),
        ("Initialization Test", test_nse_utility_initialization),
        ("Price Info Test", test_nse_utility_price_info),
        ("Equity Info Test", test_nse_utility_equity_info)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*30}")
        logger.info(f"Running {test_name}...")
        
        try:
            success = test_func()
            if success:
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
                failed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            failed += 1
    
    # Summary
    logger.info(f"\n{'='*40}")
    logger.info("🎯 NSEUtility Direct Test Results")
    logger.info(f"{'='*40}")
    logger.info(f"Total Tests: {len(tests)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {(passed/len(tests)*100):.1f}%")
    
    if failed == 0:
        logger.info("🎉 NSEUtility is working correctly!")
    else:
        logger.warning(f"⚠️ {failed} test(s) failed. NSEUtility has issues.")
    
    return failed == 0

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")
    
    success = main()
    sys.exit(0 if success else 1) 