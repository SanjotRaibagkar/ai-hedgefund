#!/usr/bin/env python3
"""
FNO RAG System - Comprehensive Backtesting Script
Run 6-month backtest for all FNO symbols with 1-day, 1-week, and 1-month horizons.
"""

import sys
import os
from datetime import datetime, timedelta
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fno_rag.backtesting import FNOBacktestEngine
from src.fno_rag import FNOEngine


def run_comprehensive_backtest():
    """Run comprehensive 6-month backtest for all FNO symbols."""
    
    print("🚀 FNO RAG System - Comprehensive 6-Month Backtest")
    print("=" * 60)
    
    try:
        # Initialize FNO engine and backtest engine
        print("1. Initializing FNO RAG System...")
        fno_engine = FNOEngine()
        
        print("2. Initializing Backtest Engine...")
        backtest_engine = FNOBacktestEngine(fno_engine)
        
        # Get available symbols
        print("3. Getting available FNO symbols...")
        available_symbols = fno_engine.get_available_symbols()
        print(f"📊 Total FNO symbols available: {len(available_symbols)}")
        
        # Select major symbols for testing (first 20 for initial test)
        test_symbols = available_symbols[:20]
        print(f"🎯 Testing with {len(test_symbols)} symbols: {test_symbols[:5]}{'...' if len(test_symbols) > 5 else ''}")
        
        # Calculate 6-month date range (use past dates)
        end_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')  # 30 days ago
        start_date = (datetime.now() - timedelta(days=210)).strftime('%Y-%m-%d')  # 210 days ago (7 months)
        
        print(f"📅 Test period: {start_date} to {end_date}")
        print(f"📅 6-month historical data range")
        
        # Run comprehensive backtest
        print("\n4. Starting comprehensive backtest...")
        print("   - Testing 1-day horizon")
        print("   - Testing 1-week horizon") 
        print("   - Testing 1-month horizon")
        print("   - Random 10 test dates over 6 months")
        print("   - All symbols for each horizon")
        
        overall_summary = backtest_engine.run_comprehensive_backtest(
            symbols=test_symbols,
            start_date=start_date,
            end_date=end_date,
            num_test_dates=10
        )
        
        # Display results
        print("\n5. Backtest Results Summary:")
        print("=" * 40)
        
        if overall_summary:
            print(f"📊 Test Period: {overall_summary.get('test_period', 'N/A')}")
            print(f"🎯 Symbols Tested: {overall_summary.get('symbols_tested', 0)}")
            print(f"📅 Test Dates: {len(overall_summary.get('test_dates', []))} dates")
            
            print("\n📈 Results by Horizon:")
            for horizon, stats in overall_summary.get('horizons', {}).items():
                print(f"   {horizon.upper()}:")
                print(f"     - Total Predictions: {stats.get('total_predictions', 0)}")
                print(f"     - Accuracy: {stats.get('accuracy', 0):.2%}")
                print(f"     - Avg Confidence: {stats.get('avg_confidence', 0):.3f}")
                print(f"     - Avg Prediction Accuracy: {stats.get('avg_prediction_accuracy', 0):.3f}")
        
        print(f"\n✅ Backtest completed successfully!")
        print(f"📁 Results saved to: data/backtest_results/")
        
        # Show file structure
        print(f"\n📂 Generated Files:")
        results_dir = "data/backtest_results"
        if os.path.exists(results_dir):
            files = os.listdir(results_dir)
            for file in sorted(files):
                if file.endswith('.csv') or file.endswith('.json'):
                    print(f"   - {file}")
        
        return overall_summary
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_extended_backtest():
    """Run extended backtest with more symbols and dates."""
    
    print("\n🔄 FNO RAG System - Extended Backtest")
    print("=" * 50)
    
    try:
        # Initialize engines
        fno_engine = FNOEngine()
        backtest_engine = FNOBacktestEngine(fno_engine)
        
        # Get all available symbols
        available_symbols = fno_engine.get_available_symbols()
        
        # Test with more symbols (up to 50)
        test_symbols = available_symbols[:50]
        print(f"🎯 Extended test with {len(test_symbols)} symbols")
        
        # Calculate date range (last 3 months for extended test)
        end_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')  # 30 days ago
        start_date = (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')  # 120 days ago (4 months)
        
        print(f"📅 Extended test period: {start_date} to {end_date}")
        
        # Run extended backtest
        extended_summary = backtest_engine.run_comprehensive_backtest(
            symbols=test_symbols,
            start_date=start_date,
            end_date=end_date,
            num_test_dates=15  # More test dates
        )
        
        print(f"✅ Extended backtest completed!")
        return extended_summary
        
    except Exception as e:
        print(f"❌ Extended backtest failed: {e}")
        return None


def analyze_results():
    """Analyze backtest results and provide insights."""
    
    print("\n📊 FNO RAG System - Results Analysis")
    print("=" * 40)
    
    try:
        results_dir = "data/backtest_results"
        
        if not os.path.exists(results_dir):
            print("❌ No backtest results found")
            return
        
        # Find overall summary file
        summary_files = [f for f in os.listdir(results_dir) if f.endswith('_overall.json')]
        
        if not summary_files:
            print("❌ No overall summary files found")
            return
        
        # Load latest summary
        latest_summary = sorted(summary_files)[-1]
        summary_path = os.path.join(results_dir, latest_summary)
        
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        print(f"📄 Analyzing: {latest_summary}")
        print(f"📅 Test Period: {summary.get('test_period', 'N/A')}")
        print(f"🎯 Symbols Tested: {summary.get('symbols_tested', 0)}")
        
        # Analyze by horizon
        print("\n📈 Performance Analysis:")
        horizons = summary.get('horizons', {})
        
        best_horizon = None
        best_accuracy = 0
        
        for horizon, stats in horizons.items():
            accuracy = stats.get('accuracy', 0)
            confidence = stats.get('avg_confidence', 0)
            pred_accuracy = stats.get('avg_prediction_accuracy', 0)
            
            print(f"\n   {horizon.upper()} Horizon:")
            print(f"     📊 Accuracy: {accuracy:.2%}")
            print(f"     🎯 Confidence: {confidence:.3f}")
            print(f"     📈 Prediction Accuracy: {pred_accuracy:.3f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_horizon = horizon
        
        if best_horizon:
            print(f"\n🏆 Best Performing Horizon: {best_horizon.upper()} ({best_accuracy:.2%} accuracy)")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if best_accuracy > 0.6:
            print(f"   ✅ System shows good predictive capability")
        elif best_accuracy > 0.5:
            print(f"   ⚠️  System shows moderate predictive capability")
        else:
            print(f"   ❌ System needs improvement")
        
        print(f"   📋 Review detailed CSV files for symbol-specific analysis")
        print(f"   🔧 Consider retraining ML models if accuracy is low")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")


if __name__ == "__main__":
    print("🚀 Starting FNO RAG System Backtesting...")
    
    # Run comprehensive backtest
    summary = run_comprehensive_backtest()
    
    if summary:
        # Run extended backtest if comprehensive was successful
        extended_summary = run_extended_backtest()
        
        # Analyze results
        analyze_results()
        
        print(f"\n🎉 All backtesting completed successfully!")
        print(f"📁 Check data/backtest_results/ for detailed results")
        print(f"📊 Review CSV files for symbol-specific performance")
        print(f"📈 Check JSON summaries for overall statistics")
    else:
        print(f"\n❌ Backtesting failed. Please check logs and try again.")
