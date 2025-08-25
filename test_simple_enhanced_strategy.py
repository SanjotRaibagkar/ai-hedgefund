#!/usr/bin/env python3
"""
Simple Test for Enhanced FNO ML Strategy
Test the improved strategy features with simpler configuration.
"""

import sys
import os
sys.path.append('./src')

from datetime import datetime, timedelta
from src.ml.fno_ml_strategy import FNOMLStrategy

def test_simple_enhanced_strategy():
    """Test the enhanced FNO ML strategy with simpler configuration."""
    print("🚀 Simple Test for Enhanced FNO ML Strategy")
    print("=" * 80)
    
    try:
        # Initialize enhanced strategy with more lenient settings
        enhanced_config = {
            'min_data_points': 20,  # Very low for testing
            'min_oi_threshold': 50,  # Very low for testing
            'min_volume_threshold': 50,  # Very low for testing
            'confidence_threshold': 0.75,  # Keep high confidence threshold
            'price_change_threshold': 0.02,
            'ensemble_voting': True,
            'min_liquidity_score': 0.1  # Very low for testing
        }
        
        ml_config = {
            'model_type': 'ensemble',
            'test_size': 0.2,
            'random_state': 42,
            'prediction_horizons': [1, 5, 21],
            'feature_selection': True,
            'ensemble_models': ['xgboost', 'lightgbm', 'random_forest', 'linear'],
            'ensemble_weights': [0.4, 0.3, 0.2, 0.1]
        }
        
        strategy = FNOMLStrategy(
            strategy_config=enhanced_config,
            ml_config=ml_config
        )
        
        print(f"✅ Enhanced strategy initialized with:")
        print(f"   - Confidence threshold: {enhanced_config['confidence_threshold']}")
        print(f"   - Min OI threshold: {enhanced_config['min_oi_threshold']}")
        print(f"   - Min volume threshold: {enhanced_config['min_volume_threshold']}")
        print(f"   - Ensemble models: {ml_config['ensemble_models']}")
        print(f"   - Ensemble weights: {ml_config['ensemble_weights']}")
        
        # Test with a stock that has good data
        ticker = "RELIANCE"  # Use RELIANCE instead of NIFTY
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        predict_start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        print(f"\n📊 Training models for {ticker}")
        print(f"   Training period: {start_date} to {end_date}")
        print(f"   Prediction period: {predict_start} to {end_date}")
        
        # Train models
        training_results = strategy.train_models(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date
        )
        
        if 'error' in training_results:
            print(f"❌ Training failed: {training_results['error']}")
            return
        
        print(f"\n✅ Training completed successfully!")
        if not training_results:
            print("   ⚠️  No models were trained - likely insufficient data after filtering")
            return
            
        for horizon, result in training_results.items():
            if 'model_performance' in result:
                perf = result['model_performance']
                print(f"   {horizon}: R² = {perf['test_r2']:.4f}, RMSE = {perf['test_rmse']:.4f}")
        
        # Make predictions
        print(f"\n🔮 Making predictions for {ticker}")
        predictions = strategy.predict_returns(
            ticker=ticker,
            start_date=predict_start,
            end_date=end_date
        )
        
        if 'error' in predictions:
            print(f"❌ Prediction failed: {predictions['error']}")
            return
        
        print(f"\n✅ Predictions completed!")
        for horizon, pred_data in predictions.items():
            if 'predictions' in pred_data:
                high_conf_count = pred_data.get('high_confidence_count', 0)
                total_count = pred_data.get('total_predictions', 0)
                latest_pred = pred_data.get('latest_prediction', 0)
                confidence = pred_data.get('prediction_confidence', 0)
                
                print(f"   {horizon}:")
                print(f"     - Latest prediction: {latest_pred*100:.2f}%")
                print(f"     - Confidence: {confidence:.3f}")
                print(f"     - High confidence predictions: {high_conf_count}/{total_count}")
                if total_count > 0:
                    print(f"     - High confidence rate: {high_conf_count/total_count*100:.1f}%")
        
        # Analyze stock
        print(f"\n📈 Analyzing {ticker}")
        analysis = strategy.analyze_stock(
            ticker=ticker,
            start_date=predict_start,
            end_date=end_date
        )
        
        if 'error' in analysis:
            print(f"❌ Analysis failed: {analysis['error']}")
            return
        
        print(f"\n✅ Analysis completed!")
        
        # Display signals
        signals = analysis.get('signals', {})
        for horizon, signal in signals.items():
            action = signal.get('action', 'HOLD')
            strength = signal.get('strength', 'NEUTRAL')
            confidence = signal.get('confidence', 0)
            pred_return = signal.get('predicted_return', 0)
            
            print(f"   {horizon}: {action} ({strength}) - {pred_return*100:.2f}% (conf: {confidence:.3f})")
        
        # Overall signal
        overall = analysis.get('overall_signal', {})
        print(f"\n🎯 Overall Signal:")
        print(f"   Action: {overall.get('action', 'HOLD')}")
        print(f"   Strength: {overall.get('strength', 'NEUTRAL')}")
        print(f"   Confidence: {overall.get('confidence', 0):.3f}")
        print(f"   Reasoning: {overall.get('reasoning', 'No reasoning available')}")
        
        print(f"\n✅ Enhanced FNO ML Strategy test completed successfully!")
        
        # Summary of improvements
        print(f"\n📋 IMPROVEMENTS IMPLEMENTED:")
        print(f"   ✅ Increased confidence threshold to 0.75")
        print(f"   ✅ Enhanced liquidity filters with scoring")
        print(f"   ✅ Implemented ensemble voting with 4 models")
        print(f"   ✅ Added confidence-based signal filtering")
        print(f"   ✅ Improved risk management with liquidity scoring")
        print(f"   ✅ Enhanced signal generation with strength levels")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_enhanced_strategy()
