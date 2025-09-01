#!/usr/bin/env python3
"""
Validate Nifty Options Analysis against actual market data
"""

import pandas as pd
from options_analyzer_v2 import OptionsAnalyzerV2
from pathlib import Path

def validate_nifty_analysis():
    print("🔍 VALIDATING NIFTY ANALYSIS AGAINST ACTUAL DATA")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = OptionsAnalyzerV2("../../../data/options_parquet")
    
    try:
        # Load data for August 29, 2025
        analyzer.load_data_for_date("20250829")
        print("✅ Data loaded successfully")
        
        # Check actual Nifty spot price from options data
        print("\n📊 ACTUAL NIFTY DATA ANALYSIS:")
        print("-" * 50)
        
        # Get snapshot at different times
        test_times = ["09:30:00", "10:00:00", "11:00:00", "12:00:00", "13:30:00", "14:00:00", "15:00:00"]
        
        for time in test_times:
            try:
                snapshot = analyzer.get_snapshot_at_time(time)
                if len(snapshot) > 0:
                    # Find ATM options (closest to actual spot)
                    strikes = sorted(snapshot['Strike_Price'].unique())
                    
                    # Get actual spot from options data (approximate)
                    # Look for strikes with highest total OI around the middle
                    mid_strike = strikes[len(strikes)//2]
                    
                    # Get PCR for this time
                    pcr_data = analyzer.calculate_pcr_indicators(snapshot)
                    
                    print(f"\n⏰ {time}:")
                    print(f"   Available Strikes: {len(strikes)} (Range: {strikes[0]:,} - {strikes[-1]:,})")
                    print(f"   Mid Strike: {mid_strike:,}")
                    print(f"   PCR (OI): {pcr_data.get('pcr_oi', 0):.3f}")
                    print(f"   PCR Sentiment: {pcr_data.get('sentiment', 'N/A')}")
                    
                    # Show actual OI distribution
                    current_expiry = snapshot['Expiry_Date'].min()
                    current_exp_data = snapshot[snapshot['Expiry_Date'] == current_expiry]
                    
                    total_call_oi = current_exp_data['CALLS_OI'].sum()
                    total_put_oi = current_exp_data['PUTS_OI'].sum()
                    
                    print(f"   Total Call OI: {total_call_oi:,.0f}")
                    print(f"   Total Put OI: {total_put_oi:,.0f}")
                    print(f"   Actual PCR: {total_put_oi/total_call_oi:.3f}")
                    
            except Exception as e:
                print(f"❌ Error at {time}: {e}")
                continue
        
        # Check for data quality issues
        print(f"\n🔍 DATA QUALITY CHECK:")
        print("-" * 50)
        
        # Check if we have multiple expiries
        all_expiries = analyzer.df['Expiry_Date'].unique()
        print(f"Available Expiries: {len(all_expiries)}")
        for exp in sorted(all_expiries):
            exp_data = analyzer.df[analyzer.df['Expiry_Date'] == exp]
            print(f"   {exp}: {len(exp_data)} records")
        
        # Check time range
        time_range = analyzer.df['Fetch_Time'].agg(['min', 'max'])
        print(f"\nTime Range: {time_range['min']} to {time_range['max']}")
        
        # Check for missing data
        print(f"\n📋 DATA COMPLETENESS:")
        print("-" * 50)
        
        # Sample a few strikes to see actual data
        sample_snapshot = analyzer.get_snapshot_at_time("13:30:00")
        if len(sample_snapshot) > 0:
            current_expiry = sample_snapshot['Expiry_Date'].min()
            current_exp_data = sample_snapshot[sample_snapshot['Expiry_Date'] == current_expiry]
            
            print(f"Sample data at 13:30 for {current_expiry}:")
            print(current_exp_data[['Strike_Price', 'CALLS_OI', 'PUTS_OI', 'CALLS_Volume', 'PUTS_Volume']].head(10))
            
            # Check for zero OI
            zero_call_oi = (current_exp_data['CALLS_OI'] == 0).sum()
            zero_put_oi = (current_exp_data['PUTS_OI'] == 0).sum()
            total_strikes = len(current_exp_data)
            
            print(f"\nZero OI Analysis:")
            print(f"   Strikes with Zero Call OI: {zero_call_oi}/{total_strikes} ({zero_call_oi/total_strikes*100:.1f}%)")
            print(f"   Strikes with Zero Put OI: {zero_put_oi}/{total_strikes} ({zero_put_oi/total_strikes*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    validate_nifty_analysis()
