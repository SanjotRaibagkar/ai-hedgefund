#!/usr/bin/env python3
"""
Simple FNO ML Strategy Test
Tests the FNO-enhanced ML strategy with available data.
"""

import sys
import pandas as pd
import duckdb
from datetime import datetime, timedelta
from src.ml.fno_ml_strategy import FNOMLStrategy

def find_stock_with_data():
    """Find a stock that has both price and FNO data."""
    print("🔍 FINDING STOCK WITH AVAILABLE DATA")
    print("=" * 50)
    
    try:
        conn = duckdb.connect('data/comprehensive_equity.duckdb')
        
        # Get stocks with price data
        price_stocks = conn.execute('''
            SELECT symbol, COUNT(*) as price_count 
            FROM price_data 
            GROUP BY symbol 
            HAVING COUNT(*) > 100 
            ORDER BY COUNT(*) DESC 
            LIMIT 20
        ''').fetchall()
        
        # Get stocks with FNO data
        fno_stocks = conn.execute('''
            SELECT TckrSymb, COUNT(*) as fno_count 
            FROM fno_bhav_copy 
            GROUP BY TckrSymb 
            HAVING COUNT(*) > 1000 
            ORDER BY COUNT(*) DESC 
            LIMIT 20
        ''').fetchall()
        
        conn.close()
        
        print("📊 Stocks with Price Data:")
        for symbol, count in price_stocks:
            print(f"   • {symbol}: {count} records")
        
        print("\n📈 Stocks with FNO Data:")
        for symbol, count in fno_stocks:
            print(f"   • {symbol}: {count} records")
        
        # Find common stocks
        price_symbols = {row[0] for row in price_stocks}
        fno_symbols = {row[0] for row in fno_stocks}
        common_symbols = price_symbols.intersection(fno_symbols)
        
        print(f"\n🎯 Common Stocks: {list(common_symbols)}")
        
        if common_symbols:
            # Return the stock with most data
            best_stock = max(common_symbols, key=lambda x: 
                           next(row[1] for row in price_stocks if row[0] == x))
            return best_stock
        else:
            # Return first stock with FNO data that might have price data
            for fno_symbol, fno_count in fno_stocks:
                if fno_symbol in price_symbols:
                    return fno_symbol
            return None
            
    except Exception as e:
        print(f"❌ Error finding stock: {e}")
        return None

def test_fno_ml_strategy_simple():
    """Test the FNO-based ML strategy with available data."""
    print("\n🚀 TESTING FNO-BASED ML STRATEGY")
    print("=" * 80)
    
    # Find a stock with data
    test_ticker = find_stock_with_data()
    if not test_ticker:
        print("❌ No suitable stock found for testing")
        return
    
    print(f"\n🎯 Testing with Stock: {test_ticker}")
    
    try:
        # Initialize the strategy with reduced minimum data requirement
        print("\n📊 1. Initializing FNO-Enhanced ML Strategy...")
        strategy_config = {
            'min_data_points': 30,  # Reduced for testing
            'fno_weight': 0.4,
            'price_weight': 0.6,
            'confidence_threshold': 0.7,
            'min_oi_threshold': 100,  # Reduced for testing
            'min_volume_threshold': 50  # Reduced for testing
        }
        strategy = FNOMLStrategy(strategy_config=strategy_config)
        print("✅ Strategy initialized successfully")
        
        # Get strategy summary
        print("\n📋 2. Strategy Configuration:")
        summary = strategy.get_strategy_summary()
        print(f"   • Strategy Type: {summary['strategy_type']}")
        print(f"   • Model Type: {summary['ml_config']['model_type']}")
        print(f"   • Prediction Horizons: {summary['prediction_horizons']}")
        print(f"   • FNO Weight: {summary['strategy_config']['fno_weight']}")
        print(f"   • Price Weight: {summary['strategy_config']['price_weight']}")
        print(f"   • Min Data Points: {summary['strategy_config']['min_data_points']}")
        
        # Set date range for testing (last 6 months to get more data)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        
        print(f"\n📅 3. Testing with Date Range: {start_date} to {end_date}")
        
        # Train the models
        print(f"\n🤖 4. Training FNO-Enhanced ML Models for {test_ticker}...")
        training_results = strategy.train_models(
            ticker=test_ticker,
            start_date=start_date,
            end_date=end_date
        )
        
        if 'error' in training_results:
            print(f"❌ Training failed: {training_results['error']}")
            return
        
        print("✅ Model training completed")
        
        # Display training results for each horizon
        for horizon_key, result in training_results.items():
            if 'model_performance' in result:
                perf = result['model_performance']
                print(f"   • {horizon_key} Horizon:")
                print(f"     - Test R² Score: {perf['test_r2']:.4f}")
                print(f"     - Test RMSE: {perf['test_rmse']:.4f}")
                print(f"     - Training Samples: {perf['training_samples']}")
                print(f"     - Test Samples: {perf['test_samples']}")
        
        # Make predictions
        print(f"\n🔮 5. Making Multi-Horizon Predictions for {test_ticker}...")
        prediction_results = strategy.predict_returns(
            ticker=test_ticker,
            start_date=start_date,
            end_date=end_date
        )
        
        if 'error' in prediction_results:
            print(f"❌ Prediction failed: {prediction_results['error']}")
            return
        
        print("✅ Predictions completed")
        
        # Display predictions for each horizon
        for horizon_key, pred_data in prediction_results.items():
            if 'latest_prediction' in pred_data:
                print(f"   • {horizon_key} Prediction: {pred_data['latest_prediction']:.4f} ({pred_data['latest_prediction']*100:.2f}%)")
                print(f"     - Confidence: {pred_data.get('prediction_confidence', 0):.4f}")
        
        # Get complete analysis
        print(f"\n📈 6. Performing Complete FNO-Enhanced Analysis for {test_ticker}...")
        analysis_results = strategy.analyze_stock(
            ticker=test_ticker,
            start_date=start_date,
            end_date=end_date
        )
        
        if 'error' in analysis_results:
            print(f"❌ Analysis failed: {analysis_results['error']}")
            return
        
        print("✅ Analysis completed")
        
        # Display signals for each horizon
        signals = analysis_results['signals']
        print(f"\n📊 7. Trading Signals by Horizon:")
        for horizon_key, signal in signals.items():
            print(f"   • {horizon_key}: {signal['action']} ({signal['strength']})")
            print(f"     - Predicted Return: {signal['predicted_return']*100:.2f}%")
            print(f"     - Confidence: {signal['confidence']:.4f}")
            print(f"     - Reasoning: {signal['reasoning']}")
        
        # Display overall signal
        overall_signal = analysis_results['overall_signal']
        print(f"\n🎯 8. Overall Trading Recommendation:")
        print(f"   • Action: {overall_signal['action']}")
        print(f"   • Strength: {overall_signal['strength']}")
        print(f"   • Confidence: {overall_signal['confidence']:.4f}")
        print(f"   • Reasoning: {overall_signal['reasoning']}")
        
        # Display feature importance for the best performing model
        print(f"\n🔍 9. Feature Importance (Best Model):")
        if training_results:
            best_horizon = max(training_results.keys(), 
                              key=lambda x: training_results[x]['model_performance']['test_r2'])
            
            if 'feature_importance' in training_results[best_horizon]:
                feature_importance = training_results[best_horizon]['feature_importance']
                # Show top 10 features
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                print(f"   • Top features for {best_horizon} horizon:")
                for feature, importance in sorted_features:
                    print(f"     - {feature}: {importance:.4f}")
        
        print(f"\n✅ FNO-Based ML Strategy Test Completed Successfully!")
        print(f"📅 Analysis Timestamp: {analysis_results['analysis_timestamp']}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fno_ml_strategy_simple()
