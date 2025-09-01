#!/usr/bin/env python3
"""
Test Script for Options Analyzer V2
Tests the analyzer at different times and shows results
"""

import pandas as pd
from options_analyzer_v2 import OptionsAnalyzerV2

def test_nifty_analyzer():
    """
    Test the Nifty Options Analyzer at different times
    """
    print("🚀 Testing Options Analyzer V2")
    print("=" * 80)
    
    # Initialize analyzer with correct path
    analyzer = OptionsAnalyzerV2("../../../data/options_parquet")
    
    # Load data for August 29, 2025
    try:
        analyzer.load_data_for_date("20250829")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Test times to analyze
    test_times = [
        "11:00:00",  # 11 AM
        "11:30:00",  # 11:30 AM
        "12:00:00",  # 12 PM
        "13:30:00",  # 1:30 PM
        "14:00:00",  # 2:00 PM
    ]
    
    # Store results for comparison
    results = []
    
    for test_time in test_times:
        print(f"\n{'='*60}")
        print(f"📊 ANALYSIS FOR {test_time}")
        print(f"{'='*60}")
        
        try:
            # Generate prediction signal
            result = analyzer.generate_prediction_signal(test_time)  # Let analyzer estimate spot price
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                continue
            
            # Print main results
            print(f"🎯 Direction: {result['direction']}")
            print(f"🔒 Confidence: {result['confidence']}")
            print(f"📈 Signal Score: {result['signal_score']:.3f}")
            print(f"📋 Key Factors: {', '.join(result['components'])}")
            
            # Print detailed metrics
            print(f"\n📊 DETAILED METRICS:")
            
            # PCR data
            pcr_data = result['detailed_metrics']['pcr']
            if 'error' not in pcr_data:
                print(f"   PCR (OI): {pcr_data.get('pcr_oi', 0):.3f}")
                print(f"   PCR (Volume): {pcr_data.get('pcr_volume', 0):.3f}")
                print(f"   PCR Sentiment: {pcr_data.get('sentiment', 'N/A')}")
            
            # Max Pain
            max_pain = result['detailed_metrics']['max_pain']
            if max_pain is not None:
                print(f"   Max Pain: {max_pain:,}")
                pain_distance = (result['spot_estimate'] - max_pain) / result['spot_estimate'] * 100
                print(f"   Distance from Max Pain: {pain_distance:.2f}%")
            
            # Gamma environment
            gamma_data = result['detailed_metrics']['gamma']
            print(f"   Gamma data type: {type(gamma_data)}")
            print(f"   Gamma data keys: {list(gamma_data.keys()) if isinstance(gamma_data, dict) else 'Not a dict'}")
            if 'error' not in gamma_data:
                print(f"   Gamma Environment: {gamma_data.get('gamma_interpretation', 'N/A')}")
                print(f"   Net Gamma Exposure: {gamma_data.get('net_gamma_exposure', 0):.2f}")
            else:
                print(f"   Gamma Error: {gamma_data['error']}")
            
            # Flow analysis
            flow_data = result['detailed_metrics']['flows']
            if 'error' not in flow_data:
                print(f"   Flow Bias: {flow_data.get('flow_bias', 'N/A')}")
                print(f"   Flow Intensity: {flow_data.get('flow_intensity', 0):.0f}")
            
            # IV metrics
            iv_data = result['detailed_metrics']['iv_metrics']
            if 'error' not in iv_data:
                print(f"   IV Skew: {iv_data.get('iv_skew', 0):.2f}")
                print(f"   Skew Interpretation: {iv_data.get('skew_interpretation', 'N/A')}")
            
            # Get momentum analysis
            momentum = analyzer.get_intraday_momentum(test_time)
            if 'error' not in momentum:
                print(f"\n📈 MOMENTUM:")
                print(f"   Trend: {momentum.get('momentum', 'N/A')}")
                print(f"   PCR Trend: {momentum.get('pcr_trend', 0):.3f}")
            
            # Store result for comparison
            results.append({
                'time': test_time,
                'direction': result['direction'],
                'confidence': result['confidence'],
                'signal_score': result['signal_score'],
                'components': result['components']
            })
            
        except Exception as e:
            print(f"❌ Error analyzing {test_time}: {e}")
            continue
    
    # Print summary comparison
    print(f"\n{'='*80}")
    print(f"📋 SUMMARY COMPARISON")
    print(f"{'='*80}")
    
    if results:
        summary_df = pd.DataFrame(results)
        print(summary_df.to_string(index=False))
        
        # Count directions
        direction_counts = summary_df['direction'].value_counts()
        print(f"\n🎯 DIRECTION BREAKDOWN:")
        for direction, count in direction_counts.items():
            print(f"   {direction}: {count}")
        
        # Average signal scores
        avg_score = summary_df['signal_score'].mean()
        print(f"\n📊 AVERAGE SIGNAL SCORE: {avg_score:.3f}")
        
        # Most common components
        all_components = []
        for comp_list in summary_df['components']:
            all_components.extend(comp_list)
        
        component_counts = pd.Series(all_components).value_counts()
        print(f"\n🔍 MOST COMMON SIGNAL COMPONENTS:")
        for component, count in component_counts.head(5).items():
            print(f"   {component}: {count}")
    
    print(f"\n✅ Analysis Complete!")

def test_specific_time_detailed(time_str="13:30:00"):
    """
    Test a specific time with detailed analysis
    """
    print(f"🔍 DETAILED ANALYSIS FOR {time_str}")
    print("=" * 80)
    
    # Initialize analyzer with correct path
    analyzer = OptionsAnalyzerV2("../../../data/options_parquet")
    
    try:
        # Load data
        analyzer.load_data_for_date("20250829")
        
        # Get snapshot
        snapshot = analyzer.get_snapshot_at_time(time_str)
        print(f"📊 Snapshot records: {len(snapshot)}")
        
        if len(snapshot) > 0:
            # Show unusual activity
            unusual = analyzer.detect_unusual_activity(snapshot)
            if 'error' not in unusual:
                print(f"\n🚨 UNUSUAL CALL ACTIVITY:")
                print(unusual['unusual_calls'].to_string(index=False))
                
                print(f"\n🚨 UNUSUAL PUT ACTIVITY:")
                print(unusual['unusual_puts'].to_string(index=False))
            
            # Show support/resistance levels
            levels = analyzer.identify_support_resistance(snapshot)
            if 'error' not in levels:
                print(f"\n🏗️ KEY SUPPORT/RESISTANCE LEVELS:")
                print("Top 5 by Total OI:")
                print(levels['key_levels'].to_string(index=False))
                
                print(f"\n🔴 RESISTANCE LEVELS (High Call OI):")
                print(levels['resistance'].to_string(index=False))
                
                print(f"\n🟢 SUPPORT LEVELS (High Put OI):")
                print(levels['support'].to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Run main test
    test_nifty_analyzer()
    
    # Run detailed test for 1:30 PM
    print(f"\n{'='*80}")
    test_specific_time_detailed("13:30:00")
