#!/usr/bin/env python3
"""
Test FNO-Based ML Strategy
Demonstrates the FNO-enhanced ML strategy with multi-horizon predictions.
"""

import sys
import pandas as pd
from datetime import datetime, timedelta
from src.ml.fno_ml_strategy import FNOMLStrategy

def test_fno_ml_strategy():
    """Test the FNO-based ML strategy with a sample stock."""
    print("🚀 TESTING FNO-BASED ML STRATEGY")
    print("=" * 80)
    
    try:
        # Initialize the strategy
        print("\n📊 1. Initializing FNO-Enhanced ML Strategy...")
        strategy = FNOMLStrategy()
        print("✅ Strategy initialized successfully")
        
        # Get strategy summary
        print("\n📋 2. Strategy Configuration:")
        summary = strategy.get_strategy_summary()
        print(f"   • Strategy Type: {summary['strategy_type']}")
        print(f"   • Model Type: {summary['ml_config']['model_type']}")
        print(f"   • Prediction Horizons: {summary['prediction_horizons']}")
        print(f"   • FNO Weight: {summary['strategy_config']['fno_weight']}")
        print(f"   • Price Weight: {summary['strategy_config']['price_weight']}")
        print(f"   • Min OI Threshold: {summary['strategy_config']['min_oi_threshold']}")
        
        # Set date range for testing (last 6 months)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        
        print(f"\n📅 3. Testing with Date Range: {start_date} to {end_date}")
        
        # Test with a stock that has FNO data
        test_ticker = "NIFTY"  # NIFTY has good FNO data
        print(f"\n🎯 4. Testing with Stock: {test_ticker}")
        
        # Train the models
        print(f"\n🤖 5. Training FNO-Enhanced ML Models for {test_ticker}...")
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
        print(f"\n🔮 6. Making Multi-Horizon Predictions for {test_ticker}...")
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
        print(f"\n📈 7. Performing Complete FNO-Enhanced Analysis for {test_ticker}...")
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
        print(f"\n📊 8. Trading Signals by Horizon:")
        for horizon_key, signal in signals.items():
            print(f"   • {horizon_key}: {signal['action']} ({signal['strength']})")
            print(f"     - Predicted Return: {signal['predicted_return']*100:.2f}%")
            print(f"     - Confidence: {signal['confidence']:.4f}")
            print(f"     - Reasoning: {signal['reasoning']}")
        
        # Display overall signal
        overall_signal = analysis_results['overall_signal']
        print(f"\n🎯 9. Overall Trading Recommendation:")
        print(f"   • Action: {overall_signal['action']}")
        print(f"   • Strength: {overall_signal['strength']}")
        print(f"   • Confidence: {overall_signal['confidence']:.4f}")
        print(f"   • Reasoning: {overall_signal['reasoning']}")
        
        # Display feature importance for the best performing model
        print(f"\n🔍 10. Feature Importance (Best Model):")
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

def test_multiple_stocks():
    """Test the strategy with multiple stocks."""
    print("\n" + "=" * 80)
    print("🧪 TESTING MULTIPLE STOCKS")
    print("=" * 80)
    
    test_stocks = ["BANKNIFTY", "RELIANCE", "TCS"]
    
    for stock in test_stocks:
        print(f"\n📊 Testing {stock}...")
        try:
            strategy = FNOMLStrategy()
            
            # Set date range
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            
            # Quick analysis
            analysis = strategy.analyze_stock(stock, start_date, end_date)
            
            if 'error' not in analysis:
                overall = analysis['overall_signal']
                print(f"   ✅ {stock}: {overall['action']} ({overall['strength']}) - Confidence: {overall['confidence']:.3f}")
                
                # Show horizon-specific predictions
                for horizon_key, signal in analysis['signals'].items():
                    print(f"     - {horizon_key}: {signal['predicted_return']*100:.2f}%")
            else:
                print(f"   ❌ {stock}: {analysis['error']}")
                
        except Exception as e:
            print(f"   ❌ {stock}: Error - {str(e)}")

if __name__ == "__main__":
    test_fno_ml_strategy()
    test_multiple_stocks()
