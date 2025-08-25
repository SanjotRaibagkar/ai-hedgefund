#!/usr/bin/env python3
"""
Test Fixed Enhanced Options Analyzer
Test script to verify the fixed options analyzer is working correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import logging
from loguru import logger

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_fixed_options_analyzer():
    """Test the fixed enhanced options analyzer."""
    logger.info("🎯 Testing Fixed Enhanced Options Analyzer")
    
    try:
        from src.screening.fixed_enhanced_options_analyzer import FixedEnhancedOptionsAnalyzer
        
        # Initialize analyzer
        analyzer = FixedEnhancedOptionsAnalyzer()
        
        # Test both indices
        indices = ['NIFTY', 'BANKNIFTY']
        
        for index in indices:
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 Testing {index}")
            logger.info(f"{'='*60}")
            
            # Test analysis
            result = analyzer.analyze_index_options(index)
            if result:
                logger.info(f"✅ Analysis successful for {index}")
                logger.info(f"   Current Price: ₹{result['current_price']:,.2f}")
                logger.info(f"   ATM Strike: ₹{result['atm_strike']:,.2f}")
                logger.info(f"   PCR: {result['oi_analysis']['pcr']:.2f}")
                logger.info(f"   Signal: {result['signal']['signal']} ({result['signal']['confidence']:.1f}% confidence)")
                logger.info(f"   Trade: {result['signal']['suggested_trade']}")
                logger.info(f"   Call OI: {result['oi_analysis']['atm_call_oi']:,.0f}")
                logger.info(f"   Put OI: {result['oi_analysis']['atm_put_oi']:,.0f}")
                logger.info(f"   Call ΔOI: {result['oi_analysis']['atm_call_oi_change']:+,.0f}")
                logger.info(f"   Put ΔOI: {result['oi_analysis']['atm_put_oi_change']:+,.0f}")
            else:
                logger.error(f"❌ Analysis failed for {index}")
            
            # Test save to CSV
            success = analyzer.run_analysis_and_save(index)
            if success:
                logger.info(f"✅ CSV save successful for {index}")
            else:
                logger.error(f"❌ CSV save failed for {index}")
            
            # Test get latest analysis (for UI)
            ui_result = analyzer.get_latest_analysis(index)
            if ui_result:
                logger.info(f"✅ UI analysis successful for {index}")
                logger.info(f"   UI Current Price: ₹{ui_result['current_price']:,.2f}")
                logger.info(f"   UI Signal: {ui_result['signal']}")
            else:
                logger.error(f"❌ UI analysis failed for {index}")
        
        # Check CSV file
        import pandas as pd
        csv_file = "results/options_tracker/option_tracker.csv"
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            logger.info(f"\n{'='*60}")
            logger.info(f"📄 CSV File Status")
            logger.info(f"{'='*60}")
            logger.info(f"   File: {csv_file}")
            logger.info(f"   Records: {len(df)}")
            logger.info(f"   Columns: {list(df.columns)}")
            logger.info(f"   Latest NIFTY: {df[df['index']=='NIFTY']['signal'].iloc[-1] if len(df[df['index']=='NIFTY']) > 0 else 'N/A'}")
            logger.info(f"   Latest BANKNIFTY: {df[df['index']=='BANKNIFTY']['signal'].iloc[-1] if len(df[df['index']=='BANKNIFTY']) > 0 else 'N/A'}")
        else:
            logger.error(f"❌ CSV file not found: {csv_file}")
        
        logger.info(f"\n{'='*60}")
        logger.info("🎉 Fixed Enhanced Options Analyzer Test Complete!")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_fixed_options_analyzer()
