#!/usr/bin/env python3
"""
Test ML Strategies Module
Demonstrates the functionality of the ML-enhanced trading strategy system.
"""

import sys
import pandas as pd
from datetime import datetime, timedelta
from src.ml.ml_strategies import MLEnhancedEODStrategy

def test_ml_strategies():
    """Test the ML Strategies module with a sample stock."""
    print("🚀 TESTING ML STRATEGIES MODULE")
    print("=" * 80)
    
    try:
        # Initialize the strategy
        print("\n📊 1. Initializing ML-Enhanced EOD Strategy...")
        strategy = MLEnhancedEODStrategy()
        print("✅ Strategy initialized successfully")
        
        # Get strategy summary
        print("\n📋 2. Strategy Configuration:")
        summary = strategy.get_strategy_summary()
        print(f"   • Strategy Type: {summary['strategy_type']}")
        print(f"   • Model Type: {summary['model_type']}")
        print(f"   • Model Trained: {summary['model_trained']}")
        print(f"   • Momentum Weight: {summary['strategy_config']['momentum_weight']}")
        print(f"   • ML Weight: {summary['strategy_config']['ml_weight']}")
        print(f"   • Confidence Threshold: {summary['strategy_config']['confidence_threshold']}")
        
        # Set date range for testing (last 6 months)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        
        print(f"\n📅 3. Testing with Date Range: {start_date} to {end_date}")
        
        # Test with a popular Indian stock
        test_ticker = "RELIANCE"
        print(f"\n🎯 4. Testing with Stock: {test_ticker}")
        
        # Train the model
        print(f"\n🤖 5. Training ML Model for {test_ticker}...")
        training_results = strategy.train_model(
            ticker=test_ticker,
            start_date=start_date,
            end_date=end_date
        )
        
        if 'error' in training_results:
            print(f"❌ Training failed: {training_results['error']}")
            return
        
        print("✅ Model training completed")
        print(f"   • Model Type: {training_results['model_type']}")
        print(f"   • Test R² Score: {training_results['model_performance']['test_r2']:.4f}")
        print(f"   • Test RMSE: {training_results['model_performance']['test_rmse']:.4f}")
        print(f"   • Training Samples: {training_results['model_performance']['training_samples']}")
        print(f"   • Test Samples: {training_results['model_performance']['test_samples']}")
        
        # Make predictions
        print(f"\n🔮 6. Making Predictions for {test_ticker}...")
        prediction_results = strategy.predict_returns(
            ticker=test_ticker,
            start_date=start_date,
            end_date=end_date
        )
        
        if 'error' in prediction_results:
            print(f"❌ Prediction failed: {prediction_results['error']}")
            return
        
        print("✅ Predictions completed")
        print(f"   • Latest Prediction: {prediction_results['latest_prediction']:.4f}")
        print(f"   • Prediction Confidence: {prediction_results['prediction_confidence']:.4f}")
        
        if not prediction_results['predictions'].empty:
            print(f"   • Total Predictions: {len(prediction_results['predictions'])}")
            print(f"   • Prediction Range: {prediction_results['predictions']['predicted_return'].min():.4f} to {prediction_results['predictions']['predicted_return'].max():.4f}")
        
        # Get complete analysis
        print(f"\n📈 7. Performing Complete Analysis for {test_ticker}...")
        analysis_results = strategy.analyze_stock(
            ticker=test_ticker,
            start_date=start_date,
            end_date=end_date
        )
        
        if 'error' in analysis_results:
            print(f"❌ Analysis failed: {analysis_results['error']}")
            return
        
        print("✅ Analysis completed")
        
        # Display combined signals
        combined_signals = analysis_results['combined_signals']
        print(f"\n📊 8. Combined Signals:")
        print(f"   • Momentum Signal: {combined_signals['momentum_signal']:.4f}")
        print(f"   • ML Signal: {combined_signals['ml_signal']:.4f}")
        print(f"   • Combined Signal: {combined_signals['combined_signal']:.4f}")
        print(f"   • Combined Confidence: {combined_signals['combined_confidence']:.4f}")
        
        # Display final recommendation
        final_rec = analysis_results['final_recommendation']
        print(f"\n🎯 9. Final Trading Recommendation:")
        print(f"   • Action: {final_rec['action']}")
        print(f"   • Strength: {final_rec['strength']}")
        print(f"   • Confidence: {final_rec['confidence']:.4f}")
        print(f"   • Position Size: {final_rec['position_size']:.4f}")
        print(f"   • Reasoning: {final_rec['reasoning']}")
        
        # Display feature importance if available
        if 'feature_importance' in training_results and training_results['feature_importance']:
            print(f"\n🔍 10. Top Feature Importance:")
            feature_importance = training_results['feature_importance']
            # Sort by importance and show top 5
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for feature, importance in sorted_features:
                print(f"   • {feature}: {importance:.4f}")
        
        print(f"\n✅ ML Strategies Module Test Completed Successfully!")
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
    
    test_stocks = ["TCS", "INFY", "HDFCBANK"]
    
    for stock in test_stocks:
        print(f"\n📊 Testing {stock}...")
        try:
            strategy = MLEnhancedEODStrategy()
            
            # Set date range
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            
            # Quick analysis
            analysis = strategy.analyze_stock(stock, start_date, end_date)
            
            if 'error' not in analysis:
                rec = analysis['final_recommendation']
                print(f"   ✅ {stock}: {rec['action']} ({rec['strength']}) - Confidence: {rec['confidence']:.3f}")
            else:
                print(f"   ❌ {stock}: {analysis['error']}")
                
        except Exception as e:
            print(f"   ❌ {stock}: Error - {str(e)}")

if __name__ == "__main__":
    test_ml_strategies()
    test_multiple_stocks()
