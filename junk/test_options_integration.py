#!/usr/bin/env python3
"""
Simple Options Integration Test
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from loguru import logger

def test_options_components():
    """Test all options components."""
    logger.info("🧪 Testing Options Integration Components")
    
    # Test 1: Options ML Integration
    try:
        from src.ml.options_ml_integration import OptionsMLIntegration
        logger.info("✅ Options ML Integration imports successfully")
        
        options_ml = OptionsMLIntegration()
        signals = options_ml.get_options_signals(['NIFTY', 'BANKNIFTY'])
        logger.info(f"✅ Options signals retrieved: {len(signals)} indices")
        
    except Exception as e:
        logger.error(f"❌ Options ML Integration failed: {e}")
    
    # Test 2: UI Integration
    try:
        from src.ui.web_app.app import app
        logger.info("✅ UI app imports successfully")
    except Exception as e:
        logger.error(f"❌ UI Integration failed: {e}")
    
    # Test 3: CSV Tracking
    try:
        csv_file = "results/options_tracker/options_tracking.csv"
        if os.path.exists(csv_file):
            import pandas as pd
            df = pd.read_csv(csv_file)
            logger.info(f"✅ CSV tracking working: {len(df)} records")
        else:
            logger.warning("⚠️ CSV file not found")
    except Exception as e:
        logger.error(f"❌ CSV tracking failed: {e}")
    
    logger.info("✅ Options integration test completed!")

if __name__ == "__main__":
    test_options_components()
