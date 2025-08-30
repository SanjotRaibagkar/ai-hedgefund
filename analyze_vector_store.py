#!/usr/bin/env python3
"""
Analyze Enhanced Vector Store
Analyze the enhanced vector store data and explain if 26,816 snapshots is good or needs improvement.
"""

import sys
import os
from datetime import datetime, timedelta
from collections import Counter

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from build_enhanced_vector_store import EnhancedFNOVectorStore
from loguru import logger

def analyze_vector_store():
    """Analyze the enhanced vector store data."""
    
    print("🔍 Enhanced Vector Store Analysis")
    print("=" * 60)
    
    try:
        # Load vector store
        print("1. Loading Enhanced Vector Store...")
        vector_store = EnhancedFNOVectorStore()
        
        if not vector_store.load_vector_store():
            print("❌ Failed to load vector store")
            return
        
        metadata = vector_store.metadata
        if not metadata:
            print("❌ No metadata found in vector store")
            return
        
        total_snapshots = len(metadata)
        print(f"✅ Loaded {total_snapshots:,} snapshots")
        
        # Analyze date range
        print("\n2. Date Range Analysis:")
        dates = [case['date'] for case in metadata]
        start_date = min(dates)
        end_date = max(dates)
        
        print(f"   📅 Start Date: {start_date}")
        print(f"   📅 End Date: {end_date}")
        
        # Calculate date range
        if isinstance(start_date, str):
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            start_dt = start_date
            end_dt = end_date
        
        date_range_days = (end_dt - start_dt).days
        print(f"   📊 Total Days: {date_range_days:,} days")
        print(f"   📊 Average snapshots per day: {total_snapshots / date_range_days:.1f}")
        
        # Analyze symbols
        print("\n3. Symbol Analysis:")
        symbols = [case['symbol'] for case in metadata]
        symbol_counts = Counter(symbols)
        
        print(f"   📈 Total Unique Symbols: {len(symbol_counts)}")
        print(f"   📊 Top 10 Symbols by Volume:")
        
        for i, (symbol, count) in enumerate(symbol_counts.most_common(10), 1):
            percentage = (count / total_snapshots) * 100
            print(f"      {i:2d}. {symbol:12s}: {count:5,} snapshots ({percentage:5.1f}%)")
        
        # Analyze data quality
        print("\n4. Data Quality Analysis:")
        
        # Check for missing values
        missing_pcr = sum(1 for case in metadata if case.get('pcr') is None or case['pcr'] == 0)
        missing_iv = sum(1 for case in metadata if case.get('implied_volatility') is None or case['implied_volatility'] == 0)
        
        print(f"   ✅ Snapshots with PCR data: {total_snapshots - missing_pcr:,} ({(total_snapshots - missing_pcr)/total_snapshots*100:.1f}%)")
        print(f"   ✅ Snapshots with IV data: {total_snapshots - missing_iv:,} ({(total_snapshots - missing_iv)/total_snapshots*100:.1f}%)")
        
        # Analyze outcomes
        print("\n5. Outcome Analysis:")
        outcomes = [case.get('next_day_direction', 'UNKNOWN') for case in metadata]
        outcome_counts = Counter(outcomes)
        
        print(f"   📊 Outcome Distribution:")
        for outcome, count in outcome_counts.most_common():
            percentage = (count / total_snapshots) * 100
            print(f"      {outcome:8s}: {count:5,} cases ({percentage:5.1f}%)")
        
        # Assess if 26,816 snapshots is good
        print("\n6. Assessment: Is 26,816 Snapshots Good?")
        print("=" * 60)
        
        # Calculate metrics
        avg_snapshots_per_symbol = total_snapshots / len(symbol_counts)
        avg_snapshots_per_day = total_snapshots / date_range_days
        
        print(f"📊 Current Metrics:")
        print(f"   • Total Snapshots: {total_snapshots:,}")
        print(f"   • Unique Symbols: {len(symbol_counts)}")
        print(f"   • Date Range: {date_range_days:,} days")
        print(f"   • Avg per Symbol: {avg_snapshots_per_symbol:.1f}")
        print(f"   • Avg per Day: {avg_snapshots_per_day:.1f}")
        
        # Assessment criteria
        print(f"\n📈 Assessment Criteria:")
        
        # 1. Coverage
        if len(symbol_counts) >= 50:
            print(f"   ✅ Symbol Coverage: EXCELLENT ({len(symbol_counts)} symbols)")
        elif len(symbol_counts) >= 20:
            print(f"   ✅ Symbol Coverage: GOOD ({len(symbol_counts)} symbols)")
        else:
            print(f"   ⚠️ Symbol Coverage: LIMITED ({len(symbol_counts)} symbols)")
        
        # 2. Time coverage
        if date_range_days >= 365:
            print(f"   ✅ Time Coverage: EXCELLENT ({date_range_days:,} days = {date_range_days/365:.1f} years)")
        elif date_range_days >= 180:
            print(f"   ✅ Time Coverage: GOOD ({date_range_days:,} days = {date_range_days/365:.1f} years)")
        else:
            print(f"   ⚠️ Time Coverage: LIMITED ({date_range_days:,} days = {date_range_days/365:.1f} years)")
        
        # 3. Data density
        if avg_snapshots_per_day >= 100:
            print(f"   ✅ Data Density: EXCELLENT ({avg_snapshots_per_day:.1f} snapshots/day)")
        elif avg_snapshots_per_day >= 50:
            print(f"   ✅ Data Density: GOOD ({avg_snapshots_per_day:.1f} snapshots/day)")
        else:
            print(f"   ⚠️ Data Density: LIMITED ({avg_snapshots_per_day:.1f} snapshots/day)")
        
        # 4. Overall assessment
        print(f"\n🎯 Overall Assessment:")
        
        score = 0
        if len(symbol_counts) >= 20: score += 1
        if date_range_days >= 180: score += 1
        if avg_snapshots_per_day >= 50: score += 1
        if total_snapshots >= 20000: score += 1
        
        if score == 4:
            print(f"   🚀 EXCELLENT: Vector store is well-populated and comprehensive")
            print(f"   💡 Recommendation: Ready for production use")
        elif score >= 3:
            print(f"   ✅ GOOD: Vector store has sufficient data for most use cases")
            print(f"   💡 Recommendation: Suitable for production, consider expanding for better coverage")
        elif score >= 2:
            print(f"   ⚠️ ADEQUATE: Vector store has basic coverage but could be improved")
            print(f"   💡 Recommendation: Expand data collection for better results")
        else:
            print(f"   ❌ LIMITED: Vector store needs significant expansion")
            print(f"   💡 Recommendation: Collect more data before production use")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        if date_range_days < 365:
            print(f"   📅 Extend data collection to at least 1 year for better pattern recognition")
        
        if len(symbol_counts) < 50:
            print(f"   📈 Add more symbols to improve market coverage")
        
        if avg_snapshots_per_day < 100:
            print(f"   📊 Increase daily data collection for higher density")
        
        # Compare with original vector store
        print(f"\n7. Comparison with Original Vector Store:")
        try:
            from src.fno_rag.rag.vector_store import FNOVectorStore
            original_vs = FNOVectorStore()
            original_vs.load_vector_store()
            
            if hasattr(original_vs, 'market_conditions'):
                original_count = len(original_vs.market_conditions)
                print(f"   📊 Original Vector Store: {original_count:,} conditions")
                print(f"   📊 Enhanced Vector Store: {total_snapshots:,} snapshots")
                print(f"   📈 Improvement: {((total_snapshots - original_count) / original_count * 100):+.1f}%")
                
                if total_snapshots > original_count:
                    print(f"   ✅ Enhanced store has more data than original")
                else:
                    print(f"   ⚠️ Enhanced store has less data than original")
            else:
                print(f"   ⚠️ Cannot compare with original store")
                
        except Exception as e:
            print(f"   ⚠️ Cannot load original vector store for comparison: {e}")
        
        print(f"\n🎉 Analysis Complete!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    analyze_vector_store()
