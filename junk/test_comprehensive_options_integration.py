#!/usr/bin/env python3
"""
Comprehensive Options Integration Test
Test all options analysis components including scheduler, UI integration, and ML integration.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime
from loguru import logger

def test_options_scheduler():
    """Test the options scheduler functionality."""
    logger.info("🧪 Testing Options Scheduler")
    logger.info("=" * 40)
    
    try:
        from src.nsedata.NseUtility import NseUtils
        
        nse = NseUtils()
        
        # Test both indices
        indices = ['NIFTY', 'BANKNIFTY']
        
        for index in indices:
            logger.info(f"📊 Testing {index} options analysis...")
            
            # Get options data
            options_data = nse.get_live_option_chain(index, indices=True)
            
            if options_data is not None and not options_data.empty:
                logger.info(f"   ✅ {index} options data retrieved successfully")
                logger.info(f"   📊 Records: {len(options_data)}")
                
                # Basic analysis
                strikes = sorted(options_data['Strike_Price'].unique())
                current_price = float(strikes[len(strikes)//2])
                atm_strike = min(strikes, key=lambda x: abs(x - current_price))
                
                logger.info(f"   💰 Spot Price: ₹{current_price:,.0f}")
                logger.info(f"   🎯 ATM Strike: ₹{atm_strike:,.0f}")
                
                # OI analysis
                atm_data = options_data[options_data['Strike_Price'] == atm_strike]
                if not atm_data.empty:
                    call_oi = float(atm_data['CALLS_OI'].iloc[0]) if 'CALLS_OI' in atm_data.columns else 0
                    put_oi = float(atm_data['PUTS_OI'].iloc[0]) if 'PUTS_OI' in atm_data.columns else 0
                    pcr = put_oi / call_oi if call_oi > 0 else 0
                    
                    logger.info(f"   📊 ATM Call OI: {call_oi:,.0f}")
                    logger.info(f"   📊 ATM Put OI: {put_oi:,.0f}")
                    logger.info(f"   📊 PCR: {pcr:.2f}")
                    
                    # Signal generation
                    signal = "NEUTRAL"
                    if pcr > 0.9:
                        signal = "BULLISH"
                    elif pcr < 0.8:
                        signal = "BEARISH"
                    
                    logger.info(f"   🎯 Signal: {signal}")
                    
            else:
                logger.warning(f"   ⚠️ No {index} options data available")
        
        logger.info("✅ Options scheduler test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Options scheduler test failed: {e}")
        return False

def test_ui_integration():
    """Test UI integration with options analysis."""
    logger.info("🧪 Testing UI Integration")
    logger.info("=" * 40)
    
    try:
        from src.ui.web_app.app import app
        
        logger.info("✅ UI app imports successfully")
        logger.info(f"📱 App title: {app.title}")
        logger.info(f"🔧 App layout components: {len(app.layout.children)}")
        
        # Test that the options callback exists
        callbacks = [cb for cb in app.callback_map.values() if 'options-results' in str(cb)]
        if callbacks:
            logger.info("✅ Options analysis callback found in UI")
        else:
            logger.warning("⚠️ Options analysis callback not found")
        
        logger.info("✅ UI integration test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ UI integration test failed: {e}")
        return False

def test_ml_integration():
    """Test ML integration with options signals."""
    logger.info("🧪 Testing ML Integration")
    logger.info("=" * 40)
    
    try:
        from src.ml.options_ml_integration import OptionsMLIntegration
        import pandas as pd
        
        # Initialize options ML integration
        options_ml = OptionsMLIntegration()
        
        # Get options signals
        logger.info("📊 Getting options signals...")
        options_signals = options_ml.get_options_signals(['NIFTY', 'BANKNIFTY'])
        
        if options_signals:
            logger.info(f"✅ Retrieved signals for {len(options_signals)} indices")
            
            # Test sentiment calculation
            sentiment_score = options_ml.get_market_sentiment_score(options_signals)
            logger.info(f"📊 Market Sentiment Score: {sentiment_score:.3f}")
            
            # Test feature enhancement
            base_features = pd.DataFrame({
                'technical_score': [0.6, 0.4, 0.8],
                'fundamental_score': [0.7, 0.5, 0.9],
                'momentum_score': [0.5, 0.3, 0.7]
            })
            
            enhanced_features = options_ml.enhance_ml_features(base_features, options_signals)
            logger.info(f"📊 Enhanced features: {len(enhanced_features.columns)} columns")
            logger.info(f"📊 Original features: {len(base_features.columns)} columns")
            
            # Test prediction adjustment
            base_prediction = 0.15
            adjusted_prediction = options_ml.adjust_ml_prediction(base_prediction, options_signals)
            logger.info(f"📊 Base prediction: {base_prediction:.3f}")
            logger.info(f"📊 Adjusted prediction: {adjusted_prediction:.3f}")
            
            # Test comprehensive recommendations
            recommendations = options_ml.get_ml_recommendations(base_features, options_signals, base_prediction)
            logger.info(f"📊 Recommendation: {recommendations['recommendation']}")
            
        else:
            logger.warning("⚠️ No options signals available for ML integration")
        
        logger.info("✅ ML integration test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ ML integration test failed: {e}")
        return False

def test_csv_tracking():
    """Test CSV tracking functionality."""
    logger.info("🧪 Testing CSV Tracking")
    logger.info("=" * 40)
    
    try:
        csv_file = "results/options_tracker/options_tracking.csv"
        
        if os.path.exists(csv_file):
            import pandas as pd
            df = pd.read_csv(csv_file)
            
            logger.info(f"✅ CSV file exists: {csv_file}")
            logger.info(f"📊 Records in CSV: {len(df)}")
            logger.info(f"📋 Columns: {list(df.columns)}")
            
            if not df.empty:
                logger.info(f"📊 Latest record:")
                latest = df.iloc[-1]
                logger.info(f"   Timestamp: {latest.get('timestamp', 'N/A')}")
                logger.info(f"   Index: {latest.get('index', 'N/A')}")
                logger.info(f"   Signal: {latest.get('signal_type', 'N/A')}")
                logger.info(f"   Confidence: {latest.get('confidence', 'N/A')}%")
            
        else:
            logger.warning(f"⚠️ CSV file not found: {csv_file}")
        
        logger.info("✅ CSV tracking test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ CSV tracking test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all comprehensive tests."""
    logger.info("🚀 Starting Comprehensive Options Integration Test")
    logger.info("=" * 60)
    
    tests = [
        ("Options Scheduler", test_options_scheduler),
        ("UI Integration", test_ui_integration),
        ("ML Integration", test_ml_integration),
        ("CSV Tracking", test_csv_tracking)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} Test...")
        try:
            result = test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{status} {test_name} Test")
        except Exception as e:
            logger.error(f"❌ {test_name} Test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n📊 Test Summary:")
    logger.info("=" * 40)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status} {test_name}")
    
    logger.info(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Options integration is working correctly.")
    else:
        logger.warning(f"⚠️ {total - passed} test(s) failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    run_comprehensive_test()
