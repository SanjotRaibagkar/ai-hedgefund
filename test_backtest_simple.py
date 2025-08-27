#!/usr/bin/env python3
"""
Simple Backtest Test
Test the backtesting system with a few symbols to verify it works.
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fno_rag.backtesting import FNOBacktestEngine
from src.fno_rag import FNOEngine, HorizonType


def test_simple_backtest():
    """Test the backtesting system with a few symbols."""
    
    print("🧪 Simple Backtest Test")
    print("=" * 40)
    
    try:
        # Initialize engines
        print("1. Initializing FNO RAG System...")
        fno_engine = FNOEngine()
        
        print("2. Initializing Backtest Engine...")
        backtest_engine = FNOBacktestEngine(fno_engine)
        
        # Get available symbols
        print("3. Getting available FNO symbols...")
        available_symbols = fno_engine.get_available_symbols()
        print(f"📊 Total FNO symbols available: {len(available_symbols)}")
        
        # Test with just 3 major symbols
        test_symbols = available_symbols[:3]
        print(f"🎯 Testing with {len(test_symbols)} symbols: {test_symbols}")
        
        # Calculate date range (last 2 months)
        end_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        
        print(f"📅 Test period: {start_date} to {end_date}")
        
        # Test only daily horizon for now
        print("\n4. Testing daily horizon...")
        
        # Generate test dates (just 5 dates)
        test_dates = backtest_engine.generate_test_dates(start_date, end_date, 5)
        print(f"📅 Test dates: {test_dates}")
        
        # Run backtest for daily horizon only
        results = backtest_engine.backtest_all_symbols(test_symbols, test_dates, HorizonType.DAILY)
        
        print(f"\n5. Results:")
        print(f"   Total predictions: {len(results)}")
        
        if results:
            # Calculate basic stats
            correct_predictions = sum(1 for r in results if r.prediction_correct)
            accuracy = correct_predictions / len(results) if results else 0.0
            
            print(f"   Correct predictions: {correct_predictions}")
            print(f"   Accuracy: {accuracy:.2%}")
            
            # Show first few results
            print(f"\n6. Sample Results:")
            for i, result in enumerate(results[:5]):
                print(f"   {i+1}. {result.symbol} ({result.prediction_date}): "
                      f"Predicted {result.predicted_up_prob:.2f}/{result.predicted_down_prob:.2f}/{result.predicted_neutral_prob:.2f}, "
                      f"Actual: {result.actual_direction} ({result.actual_return:.3f}), "
                      f"Correct: {result.prediction_correct}")
        
        print(f"\n✅ Simple backtest completed!")
        return True
        
    except Exception as e:
        print(f"❌ Simple backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Starting Simple Backtest Test...")
    success = test_simple_backtest()
    
    if success:
        print(f"\n🎉 Simple backtest test passed!")
    else:
        print(f"\n❌ Simple backtest test failed!")

