#!/usr/bin/env python3
"""
Test script to check fundamental data availability
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from loguru import logger
from src.nsedata.NseUtility import NseUtils
from src.tools.enhanced_api import get_financial_metrics, get_market_cap


def test_nse_fundamental_data():
    """Test fundamental data from NSEUtility."""
    logger.info("🧪 Testing NSEUtility fundamental data...")
    
    nse = NseUtils()
    test_symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
    
    for symbol in test_symbols:
        try:
            logger.info(f"Testing {symbol}...")
            
            # Get equity info
            equity_info = nse.equity_info(symbol)
            if equity_info:
                logger.info(f"✅ {symbol}: Equity info available")
                
                # Check price info
                if 'priceInfo' in equity_info:
                    price_info = equity_info['priceInfo']
                    logger.info(f"   📊 Price info keys: {list(price_info.keys())}")
                    
                    # Check for fundamental data
                    fundamental_keys = ['marketCap', 'faceValue', 'bookValue', 'eps', 'pe', 'pb', 'dividendYield']
                    available_fundamentals = []
                    
                    for key in fundamental_keys:
                        if key in price_info and price_info[key] is not None:
                            available_fundamentals.append(f"{key}: {price_info[key]}")
                    
                    if available_fundamentals:
                        logger.info(f"   📈 Available fundamentals: {available_fundamentals}")
                    else:
                        logger.warning(f"   ⚠️ No fundamental data found")
                else:
                    logger.warning(f"   ⚠️ No price info found")
            else:
                logger.error(f"   ❌ No equity info available")
                
        except Exception as e:
            logger.error(f"   ❌ Error testing {symbol}: {e}")
        
        print()


def test_enhanced_api_fundamental_data():
    """Test fundamental data from enhanced API."""
    logger.info("🧪 Testing Enhanced API fundamental data...")
    
    test_symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
    
    for symbol in test_symbols:
        try:
            logger.info(f"Testing {symbol}...")
            
            # Get financial metrics
            metrics = get_financial_metrics(symbol, datetime.now().strftime('%Y-%m-%d'), limit=1)
            if metrics:
                metric = metrics[0]
                logger.info(f"   ✅ Financial metrics available")
                
                # Check available fields
                available_fields = []
                for field_name, field_value in metric.model_dump().items():
                    if field_value is not None:
                        available_fields.append(f"{field_name}: {field_value}")
                
                if available_fields:
                    logger.info(f"   📈 Available fields: {available_fields[:5]}...")  # Show first 5
                else:
                    logger.warning(f"   ⚠️ No fundamental data found")
            else:
                logger.warning(f"   ⚠️ No financial metrics available")
                
        except Exception as e:
            logger.error(f"   ❌ Error testing {symbol}: {e}")
        
        print()


def main():
    """Main test function."""
    logger.info("🚀 STARTING FUNDAMENTAL DATA AVAILABILITY TEST")
    logger.info("=" * 60)
    
    # Test NSEUtility
    test_nse_fundamental_data()
    
    logger.info("=" * 60)
    
    # Test Enhanced API
    test_enhanced_api_fundamental_data()
    
    logger.info("=" * 60)
    logger.info("🎉 FUNDAMENTAL DATA AVAILABILITY TEST COMPLETED")


if __name__ == "__main__":
    main()
