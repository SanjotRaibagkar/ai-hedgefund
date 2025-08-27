#!/usr/bin/env python3
"""
Quick FNO Backtesting
Fast backtesting of the FNO RAG system with limited data.
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fno_rag import FNOEngine, HorizonType
from loguru import logger


def quick_backtest():
    """Run a quick backtest of the FNO RAG system."""
    
    print("🧪 Quick FNO Backtesting")
    print("=" * 50)
    
    try:
        # Initialize FNO engine
        print("1. Initializing FNO RAG System...")
        start_time = time.time()
        fno_engine = FNOEngine()
        init_time = time.time() - start_time
        print(f"   ✅ Initialized in {init_time:.2f} seconds")
        
        # Get available symbols
        print("\n2. Getting available symbols...")
        symbols = fno_engine.get_available_symbols()
        print(f"   ✅ Found {len(symbols)} symbols")
        
        # Select a few major symbols for testing
        test_symbols = ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS', 'INFY']
        available_test_symbols = [s for s in test_symbols if s in symbols]
        
        if not available_test_symbols:
            print("   ⚠️ No test symbols found, using first 5 available symbols")
            available_test_symbols = symbols[:5]
        
        print(f"   📊 Testing symbols: {available_test_symbols}")
        
        # Test predictions for different horizons
        print("\n3. Testing predictions...")
        horizons = [HorizonType.DAILY, HorizonType.WEEKLY]
        results = []
        
        for symbol in available_test_symbols:
            for horizon in horizons:
                try:
                    print(f"   🔍 Testing {symbol} ({horizon.value})...")
                    result = fno_engine.predict_probability(symbol, horizon)
                    
                    results.append({
                        'symbol': symbol,
                        'horizon': horizon.value,
                        'up_probability': result.up_probability,
                        'down_probability': result.down_probability,
                        'neutral_probability': result.neutral_probability,
                        'confidence_score': result.confidence_score
                    })
                    
                    print(f"      ✅ Up: {result.up_probability:.3f}, Down: {result.down_probability:.3f}, Confidence: {result.confidence_score:.3f}")
                    
                except Exception as e:
                    print(f"      ❌ Failed: {e}")
                    results.append({
                        'symbol': symbol,
                        'horizon': horizon.value,
                        'up_probability': 0,
                        'down_probability': 0,
                        'neutral_probability': 0,
                        'confidence_score': 0,
                        'error': str(e)
                    })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"quick_backtest_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        
        print(f"\n4. Results saved to: {results_file}")
        
        # Show summary
        print("\n5. Backtest Summary:")
        print(f"   📊 Total predictions: {len(results)}")
        print(f"   📈 Symbols tested: {len(available_test_symbols)}")
        print(f"   🎯 Horizons tested: {len(horizons)}")
        
        # Show top predictions
        print("\n6. Top Predictions (by confidence):")
        top_results = results_df.sort_values('confidence_score', ascending=False).head(10)
        for _, row in top_results.iterrows():
            print(f"   🏆 {row['symbol']} ({row['horizon']}): Confidence={row['confidence_score']:.3f}, Up={row['up_probability']:.3f}")
        
        # Test natural language query
        print("\n7. Testing natural language query...")
        try:
            query = "Give me FNO stocks which can move 3% tomorrow"
            response = fno_engine.chat_query(query)
            print(f"   🤖 Query: {query}")
            if 'error' in response and response['error']:
                print(f"   ❌ Error: {response['message']}")
            else:
                print(f"   ✅ Response: {response.get('message', 'No message')[:200]}...")
        except Exception as e:
            print(f"   ❌ Chat query failed: {e}")
        
        total_time = time.time() - start_time
        print(f"\n🎉 Quick backtest completed in {total_time:.2f} seconds!")
        return True
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Starting Quick FNO Backtesting...")
    success = quick_backtest()
    
    if success:
        print(f"\n🎉 Backtest completed successfully!")
    else:
        print(f"\n❌ Backtest failed!")
