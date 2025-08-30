#!/usr/bin/env python3
"""
Test Real-Time FNO Analysis
Test the real-time analysis system with realistic predictions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from real_time_analysis import RealTimeFNOPredictor
from loguru import logger

def test_real_time_analysis():
    """Test the real-time analysis system."""
    
    print("🧪 Testing Real-Time FNO Analysis")
    print("=" * 60)
    
    try:
        # Initialize predictor
        print("1. Initializing Real-Time Predictor...")
        predictor = RealTimeFNOPredictor()
        
        # Test with a few key symbols
        test_symbols = ['NIFTY', 'RELIANCE', 'BANKNIFTY']
        
        print(f"\n2. Testing Analysis for {len(test_symbols)} symbols:")
        
        all_predictions = []
        
        for symbol in test_symbols:
            print(f"\n📊 Analyzing {symbol}...")
            
            # Get latest data
            latest_data = predictor.get_latest_market_data(symbol)
            if not latest_data:
                print(f"   ❌ No data available for {symbol}")
                continue
            
            print(f"   ✅ Latest data found for {latest_data['TRADE_DATE']}")
            
            # Calculate features
            features = predictor.calculate_real_time_features(latest_data)
            if not features:
                print(f"   ❌ Could not calculate features for {symbol}")
                continue
            
            print(f"   📈 Current Price: ₹{features['close_price']:.2f}")
            print(f"   📊 Daily Return: {features['daily_return']:+.2f}%")
            print(f"   🧠 Sentiment: {features['sentiment']}")
            
            # Find similar cases
            similar_cases = predictor.get_similar_historical_cases(symbol, features)
            print(f"   🔍 Found {len(similar_cases)} similar historical cases")
            
            # Make prediction
            prediction = predictor.predict_next_day_movement(symbol, features, similar_cases)
            if prediction:
                all_predictions.append(prediction)
                
                # Display prediction
                print(f"   🎯 Prediction: {prediction['predicted_return']:+.2f}% ({prediction['direction']})")
                print(f"   🔥 Confidence: {prediction['confidence']:.1%}")
                print(f"   📋 Reasoning: {prediction['reasoning']}")
            else:
                print(f"   ❌ Could not generate prediction for {symbol}")
        
        # Display summary
        if all_predictions:
            print(f"\n3. Analysis Summary:")
            print("=" * 50)
            
            # Count directions
            up_count = sum(1 for p in all_predictions if p['direction'] == 'UP')
            down_count = sum(1 for p in all_predictions if p['direction'] == 'DOWN')
            flat_count = sum(1 for p in all_predictions if p['direction'] == 'FLAT')
            
            print(f"📈 Bullish: {up_count} symbols")
            print(f"📉 Bearish: {down_count} symbols")
            print(f"➡️ Neutral: {flat_count} symbols")
            
            # High confidence predictions
            high_confidence = [p for p in all_predictions if p['confidence'] > 0.7]
            if high_confidence:
                print(f"\n🔥 High Confidence Predictions (>70%):")
                for pred in high_confidence:
                    direction_emoji = "📈" if pred['direction'] == 'UP' else "📉" if pred['direction'] == 'DOWN' else "➡️"
                    print(f"   {direction_emoji} {pred['symbol']}: {pred['predicted_return']:+.2f}% ({pred['confidence']:.1%})")
            
            # Top movers
            top_movers = sorted(all_predictions, key=lambda x: abs(x['predicted_return']), reverse=True)[:3]
            print(f"\n🚀 Top Expected Movers:")
            for pred in top_movers:
                direction_emoji = "📈" if pred['direction'] == 'UP' else "📉" if pred['direction'] == 'DOWN' else "➡️"
                print(f"   {direction_emoji} {pred['symbol']}: {pred['predicted_return']:+.2f}%")
        
        print(f"\n✅ Real-Time Analysis Test Complete!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"❌ Test failed: {e}")
        return False

def test_single_symbol_analysis():
    """Test analysis for a single symbol."""
    
    print("\n🧪 Testing Single Symbol Analysis")
    print("=" * 40)
    
    try:
        predictor = RealTimeFNOPredictor()
        
        # Test with NIFTY
        symbol = 'NIFTY'
        print(f"📊 Analyzing {symbol}...")
        
        latest_data = predictor.get_latest_market_data(symbol)
        if latest_data:
            features = predictor.calculate_real_time_features(latest_data)
            if features:
                similar_cases = predictor.get_similar_historical_cases(symbol, features)
                prediction = predictor.predict_next_day_movement(symbol, features, similar_cases)
                if prediction:
                    predictor._display_prediction(prediction, features)
                    print("✅ Single symbol analysis successful!")
                    return True
                else:
                    print(f"❌ Could not generate prediction for {symbol}")
            else:
                print(f"❌ Could not calculate features for {symbol}")
        else:
            print(f"❌ No data available for {symbol}")
        
        return False
        
    except Exception as e:
        logger.error(f"Single symbol test failed: {e}")
        print(f"❌ Single symbol test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Real-Time FNO Analysis Testing")
    print("=" * 60)
    
    # Test 1: Multiple symbols
    success1 = test_real_time_analysis()
    
    # Test 2: Single symbol
    success2 = test_single_symbol_analysis()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Real-time analysis is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
