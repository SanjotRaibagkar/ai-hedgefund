#!/usr/bin/env python3
"""
Test Unified Options Analyzer
Test the new unified options analyzer to ensure it's working correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.screening.enhanced_options_analyzer import get_latest_analysis, run_analysis_and_save

def test_unified_analyzer():
    """Test the unified options analyzer."""
    print("🧪 Testing Unified Options Analyzer")
    print("=" * 50)
    
    # Test NIFTY analysis
    print("\n📊 Testing NIFTY Analysis:")
    try:
        result = get_latest_analysis('NIFTY')
        if result:
            print(f"✅ NIFTY Analysis Successful:")
            print(f"   Spot Price: ₹{result['current_price']:,.2f}")
            print(f"   ATM Strike: ₹{result['atm_strike']:,.2f}")
            print(f"   PCR: {result['pcr']:.2f}")
            print(f"   Signal: {result['signal']} ({result['confidence']:.1f}% confidence)")
            print(f"   Trade: {result['suggested_trade']}")
        else:
            print("❌ NIFTY Analysis Failed")
    except Exception as e:
        print(f"❌ NIFTY Analysis Error: {e}")
    
    # Test BANKNIFTY analysis
    print("\n📊 Testing BANKNIFTY Analysis:")
    try:
        result = get_latest_analysis('BANKNIFTY')
        if result:
            print(f"✅ BANKNIFTY Analysis Successful:")
            print(f"   Spot Price: ₹{result['current_price']:,.2f}")
            print(f"   ATM Strike: ₹{result['atm_strike']:,.2f}")
            print(f"   PCR: {result['pcr']:.2f}")
            print(f"   Signal: {result['signal']} ({result['confidence']:.1f}% confidence)")
            print(f"   Trade: {result['suggested_trade']}")
        else:
            print("❌ BANKNIFTY Analysis Failed")
    except Exception as e:
        print(f"❌ BANKNIFTY Analysis Error: {e}")
    
    # Test CSV saving
    print("\n💾 Testing CSV Saving:")
    try:
        success = run_analysis_and_save('NIFTY')
        if success:
            print("✅ CSV Save Successful")
        else:
            print("❌ CSV Save Failed")
    except Exception as e:
        print(f"❌ CSV Save Error: {e}")

if __name__ == "__main__":
    test_unified_analyzer()

