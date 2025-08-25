#!/usr/bin/env python3
"""
Test Screening System with Comprehensive Data
Tests the EOD screener with the large dataset we've downloaded.
"""

import pandas as pd
import sqlite3
import time
from datetime import datetime
from src.screening.duckdb_eod_screener import DuckDBEODScreener

def test_screening_with_comprehensive_data():
    """Test the screening system with comprehensive data."""
    print("🧪 TESTING SCREENING SYSTEM WITH COMPREHENSIVE DATA")
    print("=" * 60)
    
    # Initialize screener
    screener = DuckDBEODScreener()
    
    # Get database stats
    print("📊 Database Statistics:")
    conn = sqlite3.connect("data/comprehensive_equity.db")
    
    # Get total symbols
    total_symbols = conn.execute('SELECT COUNT(*) FROM securities').fetchone()[0]
    fno_symbols = conn.execute('SELECT COUNT(*) FROM securities WHERE is_fno = 1').fetchone()[0]
    total_records = conn.execute('SELECT COUNT(*) FROM price_data').fetchone()[0]
    
    print(f"   📈 Total Symbols: {total_symbols}")
    print(f"   🔥 FNO Symbols: {fno_symbols}")
    print(f"   📊 Total Records: {total_records:,}")
    
    # Get sample symbols for testing
    print("\n🎯 Testing with Sample Symbols:")
    
    # Get some major stocks
    major_stocks = conn.execute('''
        SELECT symbol, company_name, is_fno 
        FROM securities 
        WHERE symbol IN ('RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK')
        LIMIT 10
    ''').fetchall()
    
    # Get some FNO stocks
    fno_stocks = conn.execute('''
        SELECT symbol, company_name, is_fno 
        FROM securities 
        WHERE is_fno = 1 
        LIMIT 5
    ''').fetchall()
    
    # Get some random stocks
    random_stocks = conn.execute('''
        SELECT symbol, company_name, is_fno 
        FROM securities 
        WHERE is_fno = 0 
        ORDER BY RANDOM() 
        LIMIT 5
    ''').fetchall()
    
    test_symbols = major_stocks + fno_stocks + random_stocks
    
    print(f"   🎯 Testing {len(test_symbols)} symbols:")
    for symbol, company, is_fno in test_symbols:
        fno_flag = "🔥" if is_fno else "📈"
        print(f"      {fno_flag} {symbol}: {company}")
    
    conn.close()
    
    # Test screening
    print(f"\n🔍 Running EOD Screening...")
    start_time = time.time()
    
    try:
        # Run screening
        results = screener.screen_stocks(test_symbols)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Screening completed in {duration:.2f} seconds")
        
        # Display results
        print(f"\n📊 SCREENING RESULTS:")
        print("=" * 40)
        
        if results and not results.empty:
            print(f"🎯 Found {len(results)} signals:")
            print()
            
            # Group by signal type
            bullish = results[results['signal'] == 'BULLISH']
            bearish = results[results['signal'] == 'BEARISH']
            
            if not bullish.empty:
                print(f"📈 BULLISH Signals ({len(bullish)}):")
                for _, row in bullish.head(5).iterrows():
                    print(f"   ✅ {row['symbol']}: ₹{row['current_price']:.2f} | Entry: ₹{row['entry']:.2f} | SL: ₹{row['stop_loss']:.2f} | Target: ₹{row['target']:.2f} | Confidence: {row['confidence']}%")
                if len(bullish) > 5:
                    print(f"   ... and {len(bullish) - 5} more bullish signals")
                print()
            
            if not bearish.empty:
                print(f"📉 BEARISH Signals ({len(bearish)}):")
                for _, row in bearish.head(5).iterrows():
                    print(f"   ❌ {row['symbol']}: ₹{row['current_price']:.2f} | Entry: ₹{row['entry']:.2f} | SL: ₹{row['stop_loss']:.2f} | Target: ₹{row['target']:.2f} | Confidence: {row['confidence']}%")
                if len(bearish) > 5:
                    print(f"   ... and {len(bearish) - 5} more bearish signals")
                print()
            
            # Performance metrics
            avg_confidence = results['confidence'].mean()
            avg_risk_reward = results['risk_reward_ratio'].mean()
            
            print(f"📊 Performance Metrics:")
            print(f"   🎯 Average Confidence: {avg_confidence:.1f}%")
            print(f"   ⚖️ Average Risk/Reward: {avg_risk_reward:.2f}")
            print(f"   📈 Success Rate: {(len(results)/len(test_symbols)*100):.1f}%")
            
        else:
            print("❌ No signals generated")
            print("💡 This might be due to:")
            print("   - Market conditions not meeting criteria")
            print("   - Data quality issues")
            print("   - Screening parameters too strict")
        
        # Test with larger sample
        print(f"\n🔍 Testing with Larger Sample (50 symbols)...")
        
        # Get more symbols for testing
        conn = sqlite3.connect("data/comprehensive_equity.db")
        large_sample = conn.execute('''
            SELECT symbol, company_name, is_fno 
            FROM securities 
            ORDER BY RANDOM() 
            LIMIT 50
        ''').fetchall()
        conn.close()
        
        start_time = time.time()
        large_results = screener.screen_stocks(large_sample)
        end_time = time.time()
        large_duration = end_time - start_time
        
        print(f"✅ Large sample screening completed in {large_duration:.2f} seconds")
        
        if large_results and not large_results.empty:
            print(f"🎯 Large sample found {len(large_results)} signals")
            print(f"📊 Processing rate: {len(large_sample)/large_duration:.1f} symbols/second")
        else:
            print("❌ No signals in large sample")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during screening: {e}")
        return False

def test_screening_performance():
    """Test screening performance with different sample sizes."""
    print(f"\n⚡ PERFORMANCE TESTING")
    print("=" * 40)
    
            from src.screening.duckdb_eod_screener import duckdb_eod_screener
        screener = duckdb_eod_screener
    
    # Test different sample sizes
    sample_sizes = [10, 25, 50, 100]
    
    for size in sample_sizes:
        print(f"\n🔍 Testing with {size} symbols...")
        
        # Get sample
        conn = sqlite3.connect("data/comprehensive_equity.db")
        sample = conn.execute(f'''
            SELECT symbol, company_name, is_fno 
            FROM securities 
            ORDER BY RANDOM() 
            LIMIT {size}
        ''').fetchall()
        conn.close()
        
        # Time the screening
        start_time = time.time()
        try:
            results = screener.screen_stocks(sample)
            end_time = time.time()
            duration = end_time - start_time
            
            signals = len(results) if results is not None and not results.empty else 0
            
            print(f"   ⏱️ Time: {duration:.2f}s")
            print(f"   🎯 Signals: {signals}")
            print(f"   📊 Rate: {size/duration:.1f} symbols/s")
            print(f"   📈 Success: {(signals/size*100):.1f}%")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    # Test basic screening
    success = test_screening_with_comprehensive_data()
    
    if success:
        # Test performance
        test_screening_performance()
    
    print(f"\n🎉 SCREENING SYSTEM TEST COMPLETED!")
    print("=" * 50)
