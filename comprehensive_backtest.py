#!/usr/bin/env python3
"""
Comprehensive FNO Backtesting
Full backtesting of the FNO RAG system with historical validation.
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import time
import random

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fno_rag import FNOEngine, HorizonType
from loguru import logger


def comprehensive_backtest():
    """Run comprehensive backtesting of the FNO RAG system."""
    
    print("🧪 Comprehensive FNO Backtesting")
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
        
        # Select major symbols for testing
        major_symbols = ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK', 'ITC', 'SBIN', 'BHARTIARTL']
        test_symbols = [s for s in major_symbols if s in symbols]
        
        if len(test_symbols) < 5:
            print(f"   ⚠️ Only {len(test_symbols)} major symbols found, adding more...")
            additional_symbols = [s for s in symbols if s not in test_symbols][:10]
            test_symbols.extend(additional_symbols)
        
        print(f"   📊 Testing symbols: {test_symbols[:10]}...")
        
        # Test predictions for different horizons
        print("\n3. Testing predictions for all horizons...")
        horizons = [HorizonType.DAILY, HorizonType.WEEKLY, HorizonType.MONTHLY]
        results = []
        
        for symbol in test_symbols:
            for horizon in horizons:
                try:
                    print(f"   🔍 Testing {symbol} ({horizon.value})...")
                    result = fno_engine.predict_probability(symbol, horizon)
                    
                    if result:
                                            results.append({
                        'symbol': symbol,
                        'horizon': horizon.value,
                        'up_probability': result.up_probability,
                        'down_probability': result.down_probability,
                        'neutral_probability': result.neutral_probability,
                        'confidence_score': result.confidence_score,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    print(f"      ✅ Up: {result.up_probability:.3f}, Down: {result.down_probability:.3f}, Confidence: {result.confidence_score:.3f}")
                    else:
                        print(f"      ⚠️ No result for {symbol}")
                        
                except Exception as e:
                    print(f"      ❌ Failed: {e}")
                    results.append({
                        'symbol': symbol,
                        'horizon': horizon.value,
                        'up_probability': 0,
                        'down_probability': 0,
                        'neutral_probability': 0,
                        'confidence_score': 0,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"comprehensive_backtest_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        
        print(f"\n4. Results saved to: {results_file}")
        
        # Show summary statistics
        print("\n5. Backtest Summary Statistics:")
        print(f"   📊 Total predictions: {len(results)}")
        print(f"   📈 Symbols tested: {len(test_symbols)}")
        print(f"   🎯 Horizons tested: {len(horizons)}")
        
        if not results_df.empty:
            print(f"   📊 Average confidence: {results_df['confidence_score'].mean():.3f}")
            print(f"   📈 Average up probability: {results_df['up_probability'].mean():.3f}")
            print(f"   📉 Average down probability: {results_df['down_probability'].mean():.3f}")
        
        # Show top predictions by confidence
        print("\n6. Top Predictions (by confidence):")
        if not results_df.empty:
            top_results = results_df.sort_values('confidence_score', ascending=False).head(10)
            for _, row in top_results.iterrows():
                print(f"   🏆 {row['symbol']} ({row['horizon']}): Confidence={row['confidence_score']:.3f}, Up={row['up_probability']:.3f}")
        
        # Test natural language queries
        print("\n7. Testing natural language queries...")
        test_queries = [
            "What's the probability of NIFTY moving up tomorrow?",
            "Give me FNO stocks which can move 3% tomorrow",
            "Show me stocks that might go up this week",
            "Predict RELIANCE movement for next month",
            "Find stocks with high probability of moving down"
        ]
        
        for query in test_queries:
            try:
                print(f"   🤖 Query: {query}")
                response = fno_engine.chat_query(query)
                
                if 'error' in response and response['error']:
                    print(f"      ❌ Error: {response['message']}")
                else:
                    message = response.get('message', 'No message')
                    print(f"      ✅ Response: {message[:100]}...")
                    
            except Exception as e:
                print(f"      ❌ Query failed: {e}")
        
        # Test system status
        print("\n8. Testing system status...")
        try:
            status = fno_engine.get_system_status()
            print(f"   📊 System initialized: {status.get('initialized', False)}")
            print(f"   📅 Last update: {status.get('last_update', 'Unknown')}")
            
            if 'data_info' in status:
                data_info = status['data_info']
                print(f"   📈 Data available: {data_info.get('data_available', False)}")
                print(f"   📊 Symbol count: {data_info.get('symbol_count', 0)}")
                
        except Exception as e:
            print(f"   ❌ Status check failed: {e}")
        
        # Performance metrics
        print("\n9. Performance Metrics:")
        total_time = time.time() - start_time
        predictions_per_second = len(results) / total_time if total_time > 0 else 0
        
        print(f"   ⏱️ Total time: {total_time:.2f} seconds")
        print(f"   🚀 Predictions per second: {predictions_per_second:.2f}")
        print(f"   📊 Success rate: {len([r for r in results if 'error' not in r]) / len(results) * 100:.1f}%")
        
        # Generate recommendations
        print("\n10. System Recommendations:")
        if not results_df.empty:
            high_confidence = results_df[results_df['confidence_score'] > 0.7]
            if not high_confidence.empty:
                print(f"   🎯 High confidence predictions: {len(high_confidence)}")
                for _, row in high_confidence.head(3).iterrows():
                    direction = "UP" if row['up_probability'] > row['down_probability'] else "DOWN"
                    print(f"      💡 {row['symbol']} ({row['horizon']}): {direction} with {row['confidence_score']:.3f} confidence")
            else:
                print("   ⚠️ No high confidence predictions found")
        
        print(f"\n🎉 Comprehensive backtest completed in {total_time:.2f} seconds!")
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Starting Comprehensive FNO Backtesting...")
    success = comprehensive_backtest()
    
    if success:
        print(f"\n🎉 Backtesting completed successfully!")
    else:
        print(f"\n❌ Backtesting failed!")
