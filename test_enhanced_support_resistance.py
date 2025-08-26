#!/usr/bin/env python3
"""
Test Enhanced Support/Resistance Calculation
Tests the new OI-based support and resistance calculation.
"""

import sys
import os
from datetime import datetime

# Add the project root to Python path
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.screening.fixed_enhanced_options_analyzer import FixedEnhancedOptionsAnalyzer

def test_support_resistance_calculation():
    """Test the enhanced support/resistance calculation."""
    print("🧪 Testing Enhanced OI-based Support/Resistance Calculation")
    print("=" * 60)
    
    try:
        # Initialize the analyzer
        analyzer = FixedEnhancedOptionsAnalyzer()
        
        # Test for NIFTY
        print("📊 Testing NIFTY Options Analysis...")
        result = analyzer.analyze_index_options('NIFTY')
        
        if result:
            print("✅ Analysis completed successfully!")
            print()
            
            # Display support/resistance results
            support_resistance = result.get('support_resistance', {})
            current_price = result.get('current_price', 0)
            
            print("🎯 Support/Resistance Analysis Results:")
            print(f"   Current Price: ₹{current_price:,.2f}")
            print(f"   Support Level: ₹{support_resistance.get('support', 'N/A'):,.2f}")
            print(f"   Resistance Level: ₹{support_resistance.get('resistance', 'N/A'):,.2f}")
            print(f"   Support Distance: {support_resistance.get('support_distance_pct', 'N/A')}%")
            print(f"   Resistance Distance: {support_resistance.get('resistance_distance_pct', 'N/A')}%")
            print()
            
            # Display signal information
            signal = result.get('signal', {})
            print("📈 Signal Analysis:")
            print(f"   Signal: {signal.get('signal', 'N/A')}")
            print(f"   Confidence: {signal.get('confidence', 'N/A')}%")
            print(f"   Suggested Trade: {signal.get('suggested_trade', 'N/A')}")
            print()
            
            # Display OI analysis
            oi_analysis = result.get('oi_analysis', {})
            print("📊 OI Analysis:")
            print(f"   PCR: {oi_analysis.get('pcr', 'N/A'):.3f}")
            print(f"   ATM Call OI: {oi_analysis.get('atm_call_oi', 'N/A'):,}")
            print(f"   ATM Put OI: {oi_analysis.get('atm_put_oi', 'N/A'):,}")
            print(f"   ATM Call OI Change: {oi_analysis.get('atm_call_oi_change', 'N/A'):,}")
            print(f"   ATM Put OI Change: {oi_analysis.get('atm_put_oi_change', 'N/A'):,}")
            print(f"   Analyzed Strikes: {oi_analysis.get('analyzed_strikes', [])}")
            
        else:
            print("❌ Analysis failed - no result returned")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_support_resistance_calculation()
