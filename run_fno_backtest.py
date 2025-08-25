#!/usr/bin/env python3
"""
Run FNO ML Strategy Backtest
Simple script to run the comprehensive FNO backtest.
"""

import sys
import os
sys.path.append('./src')

from datetime import datetime
from src.ml.fno_backtesting import FNOMLBacktesting

def main():
    """Run the FNO ML strategy backtest."""
    print("🚀 FNO ML Strategy Comprehensive Backtest")
    print("=" * 80)
    
    try:
        # Initialize backtesting framework
        backtester = FNOMLBacktesting()
        
        # Run comprehensive backtest
        results = backtester.run_comprehensive_backtest(
            start_date="2025-07-01",
            end_date=datetime.now().strftime('%Y-%m-%d')
        )
        
        if 'error' in results:
            print(f"❌ Backtest failed: {results['error']}")
            return
        
        # Print summary
        print(f"\n✅ Backtest completed successfully!")
        print(f"📊 Tickers tested: {results['tickers_tested']}")
        print(f"📅 Period: {results['backtest_period']['start']} to {results['backtest_period']['end']}")
        print(f"📁 Results saved to: {results['csv_file_path']}")
        
        # Print comprehensive report
        print(f"\n📋 COMPREHENSIVE REPORT:")
        print(results['comprehensive_report'])
        
        # Print top performers
        print(f"\n🏆 TOP PERFORMERS BY DIRECTIONAL ACCURACY:")
        ticker_performance = []
        
        for ticker, result in results['backtest_results'].items():
            if 'error' not in result and 'performance_metrics' in result:
                perf_metrics = result['performance_metrics']
                avg_directional_accuracy = 0
                count = 0
                
                for horizon in ['1d', '5d', '21d']:
                    if horizon in perf_metrics:
                        avg_directional_accuracy += perf_metrics[horizon]['directional_accuracy']
                        count += 1
                
                if count > 0:
                    avg_directional_accuracy /= count
                    ticker_performance.append((ticker, avg_directional_accuracy))
        
        # Sort by performance
        ticker_performance.sort(key=lambda x: x[1], reverse=True)
        
        for i, (ticker, accuracy) in enumerate(ticker_performance[:10], 1):
            print(f"   {i:2d}. {ticker}: {accuracy:.2%}")
        
        print(f"\n✅ FNO ML Strategy Backtest completed successfully!")
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
