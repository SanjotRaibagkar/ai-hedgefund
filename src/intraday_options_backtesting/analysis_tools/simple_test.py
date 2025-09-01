#!/usr/bin/env python3
"""
Simple test to isolate the gamma calculation issue
"""

from options_analyzer_v2 import OptionsAnalyzerV2

def simple_test():
    print("🧪 Simple Test for Gamma Calculation")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = OptionsAnalyzerV2("../../../data/options_parquet")
    
    try:
        # Load data
        analyzer.load_data_for_date("20250829")
        print("✅ Data loaded successfully")
        
        # Get snapshot at 13:30
        snapshot = analyzer.get_snapshot_at_time("13:30:00")
        print(f"✅ Snapshot created: {len(snapshot)} records")
        
        # Test gamma calculation directly
        print("\n🔍 Testing gamma calculation...")
        gamma_data = analyzer.calculate_gamma_exposure(snapshot, 19500)
        print(f"Gamma data type: {type(gamma_data)}")
        print(f"Gamma data: {gamma_data}")
        
        # Test PCR calculation
        print("\n🔍 Testing PCR calculation...")
        pcr_data = analyzer.calculate_pcr_indicators(snapshot)
        print(f"PCR data: {pcr_data}")
        
        # Test max pain calculation
        print("\n🔍 Testing max pain calculation...")
        max_pain = analyzer.find_max_pain(snapshot)
        print(f"Max pain: {max_pain}")
        
        print("\n✅ All individual tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test()
