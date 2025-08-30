#!/usr/bin/env python3
"""
Enhanced Real-Time FNO Analysis with Realistic Predictions
Advanced market insights using the enhanced FNO RAG system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
import duckdb
from build_enhanced_vector_store import EnhancedFNOVectorStore
from src.fno_rag import FNOEngine

class EnhancedRealTimeFNOPredictor:
    """Enhanced real-time FNO analysis with realistic predictions."""
    
    def __init__(self):
        """Initialize the enhanced real-time predictor."""
        self.vector_store = EnhancedFNOVectorStore()
        self.fno_engine = FNOEngine()
        self.db_manager = self.vector_store.db_manager
        
        # Key symbols to monitor
        self.key_symbols = [
            'NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS', 'HDFC', 'INFY', 
            'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'AXISBANK'
        ]
        
        # Enhanced analysis thresholds
        self.pcr_threshold_high = 1.5
        self.pcr_threshold_low = 0.7
        self.oi_change_threshold = 10
        self.volume_threshold = 1000000  # High volume threshold
        
        # Market volatility thresholds
        self.low_volatility = 1.0   # < 1% daily range
        self.high_volatility = 3.0  # > 3% daily range
        
        logger.info("🚀 Enhanced Real-time FNO Predictor initialized")
    
    def get_latest_market_data(self, symbol: str) -> dict:
        """Get the latest market data for a symbol."""
        try:
            # Get the most recent trading day with more data
            latest_query = f"""
            SELECT 
                TckrSymb,
                TRADE_DATE,
                FinInstrmTp,
                OpnPric,
                HghPric,
                LwPric,
                ClsPric,
                TtlTradgVol,
                OpnIntrst,
                ChngInOpnIntrst,
                TtlTrfVal,
                PrvsClsgPric
            FROM fno_bhav_copy
            WHERE TckrSymb = '{symbol}'
            AND ClsPric > 100
            ORDER BY TRADE_DATE DESC
            LIMIT 1
            """
            
            result = self.db_manager.connection.execute(latest_query).fetchdf()
            
            if result.empty:
                return None
            
            return result.iloc[0].to_dict()
            
        except Exception as e:
            logger.error(f"Error getting latest data for {symbol}: {e}")
            return None
    
    def get_recent_price_history(self, symbol: str, days: int = 5) -> list:
        """Get recent price history for trend analysis."""
        try:
            history_query = f"""
            SELECT 
                TRADE_DATE,
                ClsPric,
                TtlTradgVol,
                OpnIntrst,
                ChngInOpnIntrst
            FROM fno_bhav_copy
            WHERE TckrSymb = '{symbol}'
            AND ClsPric > 100
            ORDER BY TRADE_DATE DESC
            LIMIT {days}
            """
            
            result = self.db_manager.connection.execute(history_query).fetchdf()
            return result.to_dict('records') if not result.empty else []
            
        except Exception as e:
            logger.error(f"Error getting price history for {symbol}: {e}")
            return []
    
    def calculate_enhanced_features(self, data: dict, history: list = None) -> dict:
        """Calculate enhanced real-time features for prediction."""
        if not data:
            return None
        
        try:
            # Basic price calculations
            close_price = data['ClsPric']
            open_price = data['OpnPric']
            high_price = data['HghPric']
            low_price = data['LwPric']
            prev_close = data['PrvsClsgPric']
            
            # Daily return
            daily_return = ((close_price - prev_close) / prev_close * 100) if prev_close > 0 else 0
            
            # Price range and volatility
            price_range = ((high_price - low_price) / close_price * 100) if close_price > 0 else 0
            
            # Volume analysis
            volume = data['TtlTradgVol']
            oi_change = data['ChngInOpnIntrst']
            
            # Trend analysis from history
            trend_strength = 0
            volume_trend = 0
            if history and len(history) > 1:
                # Calculate trend over recent days
                prices = [h['ClsPric'] for h in history]
                if len(prices) >= 2:
                    price_trend = (prices[0] - prices[-1]) / prices[-1] * 100
                    trend_strength = min(2, abs(price_trend) / 2)  # Normalize to 0-2
                
                # Volume trend
                volumes = [h['TtlTradgVol'] for h in history]
                if len(volumes) >= 2:
                    volume_trend = (volumes[0] - volumes[-1]) / volumes[-1] * 100 if volumes[-1] > 0 else 0
            
            # Enhanced sentiment calculation
            sentiment = self._calculate_enhanced_sentiment(
                daily_return, price_range, volume, oi_change, trend_strength, volume_trend
            )
            
            # Market momentum indicators
            momentum = self._calculate_momentum(daily_return, price_range, volume)
            
            return {
                'symbol': data['TckrSymb'],
                'date': data['TRADE_DATE'],
                'close_price': close_price,
                'daily_return': daily_return,
                'price_range': price_range,
                'volume': volume,
                'oi_change': oi_change,
                'sentiment': sentiment,
                'momentum': momentum,
                'trend_strength': trend_strength,
                'volume_trend': volume_trend,
                'high_price': high_price,
                'low_price': low_price
            }
            
        except Exception as e:
            logger.error(f"Error calculating enhanced features: {e}")
            return None
    
    def _calculate_enhanced_sentiment(self, daily_return: float, price_range: float, 
                                    volume: float, oi_change: float, 
                                    trend_strength: float, volume_trend: float) -> str:
        """Calculate enhanced market sentiment based on multiple factors."""
        sentiment_score = 0
        
        # Price movement sentiment (weighted by magnitude)
        if abs(daily_return) > 2.0:
            sentiment_score += 3 if daily_return > 0 else -3
        elif abs(daily_return) > 1.0:
            sentiment_score += 2 if daily_return > 0 else -2
        elif abs(daily_return) > 0.5:
            sentiment_score += 1 if daily_return > 0 else -1
        
        # Volatility sentiment
        if price_range > self.high_volatility:
            sentiment_score -= 1  # High volatility often bearish
        elif price_range < self.low_volatility:
            sentiment_score += 0.5  # Low volatility slightly bullish
        
        # Volume sentiment
        if volume > self.volume_threshold:
            if daily_return > 0:
                sentiment_score += 1  # High volume on up day
            else:
                sentiment_score -= 1  # High volume on down day
        
        # OI change sentiment
        if abs(oi_change) > self.oi_change_threshold:
            if oi_change > 0 and daily_return > 0:
                sentiment_score += 1  # OI building on up move
            elif oi_change < 0 and daily_return < 0:
                sentiment_score -= 1  # OI unwinding on down move
        
        # Trend strength sentiment
        if trend_strength > 1.0:
            sentiment_score += 1 if daily_return > 0 else -1
        
        # Volume trend sentiment
        if abs(volume_trend) > 20:
            sentiment_score += 0.5 if volume_trend > 0 else -0.5
        
        # Determine sentiment
        if sentiment_score >= 3:
            return "STRONG_BULLISH"
        elif sentiment_score >= 1.5:
            return "BULLISH"
        elif sentiment_score <= -3:
            return "STRONG_BEARISH"
        elif sentiment_score <= -1.5:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _calculate_momentum(self, daily_return: float, price_range: float, volume: float) -> str:
        """Calculate market momentum."""
        # Momentum based on return and volume
        if abs(daily_return) > 1.0 and volume > self.volume_threshold:
            return "STRONG" if daily_return > 0 else "WEAK"
        elif abs(daily_return) > 0.5:
            return "MODERATE" if daily_return > 0 else "SLIGHT"
        else:
            return "NEUTRAL"
    
    def get_similar_historical_cases(self, symbol: str, features: dict, top_k: int = 5) -> list:
        """Find similar historical cases for prediction."""
        try:
            # Create a more specific query based on current market conditions
            sentiment = features['sentiment'].lower()
            daily_return = features['daily_return']
            momentum = features['momentum'].lower()
            
            query = f"Find similar cases where {symbol} had {sentiment} sentiment with {daily_return:+.1f}% daily return and {momentum} momentum"
            
            # Use the enhanced vector store to find similar cases
            similar_cases = self.vector_store.query_rag_system(query, top_k=top_k)
            
            return similar_cases
            
        except Exception as e:
            logger.error(f"Error finding similar cases: {e}")
            return []
    
    def predict_next_day_movement(self, symbol: str, features: dict, similar_cases: list) -> dict:
        """Predict next day movement with enhanced logic."""
        try:
            # Enhanced base prediction from sentiment and momentum
            base_prediction = self._get_enhanced_base_prediction(features)
            
            # Adjust based on similar historical cases
            if similar_cases:
                historical_returns = []
                for case in similar_cases:
                    if 'next_day_return' in case:
                        historical_returns.append(case['next_day_return'])
                
                if historical_returns:
                    avg_historical_return = sum(historical_returns) / len(historical_returns)
                    
                    # Enhanced weighted prediction: 60% historical, 25% current sentiment, 15% momentum
                    weighted_return = (
                        0.6 * avg_historical_return + 
                        0.25 * base_prediction['return'] + 
                        0.15 * self._get_momentum_adjustment(features['momentum'])
                    )
                    
                    # Confidence based on consistency and feature strength
                    confidence = self._calculate_enhanced_confidence(historical_returns, features)
                    
                    return {
                        'symbol': symbol,
                        'predicted_return': weighted_return,
                        'confidence': confidence,
                        'direction': 'UP' if weighted_return > 0.1 else 'DOWN' if weighted_return < -0.1 else 'FLAT',
                        'base_prediction': base_prediction,
                        'historical_cases': len(similar_cases),
                        'avg_historical_return': avg_historical_return,
                        'reasoning': self._generate_enhanced_reasoning(features, similar_cases)
                    }
            
            # Fallback to enhanced base prediction
            return {
                'symbol': symbol,
                'predicted_return': base_prediction['return'],
                'confidence': 0.6,  # Higher base confidence
                'direction': base_prediction['direction'],
                'base_prediction': base_prediction,
                'historical_cases': 0,
                'avg_historical_return': 0,
                'reasoning': self._generate_enhanced_reasoning(features, [])
            }
            
        except Exception as e:
            logger.error(f"Error predicting movement for {symbol}: {e}")
            return None
    
    def _get_enhanced_base_prediction(self, features: dict) -> dict:
        """Get enhanced base prediction from sentiment and momentum."""
        sentiment_predictions = {
            'STRONG_BULLISH': {'return': 2.0, 'direction': 'UP'},
            'BULLISH': {'return': 1.2, 'direction': 'UP'},
            'NEUTRAL': {'return': 0.0, 'direction': 'FLAT'},
            'BEARISH': {'return': -1.2, 'direction': 'DOWN'},
            'STRONG_BEARISH': {'return': -2.0, 'direction': 'DOWN'}
        }
        
        base_pred = sentiment_predictions.get(features['sentiment'], {'return': 0.0, 'direction': 'FLAT'})
        
        # Adjust based on momentum
        momentum_adjustment = self._get_momentum_adjustment(features['momentum'])
        base_pred['return'] += momentum_adjustment * 0.3
        
        return base_pred
    
    def _get_momentum_adjustment(self, momentum: str) -> float:
        """Get momentum adjustment factor."""
        momentum_adjustments = {
            'STRONG': 1.5,
            'MODERATE': 0.8,
            'SLIGHT': 0.3,
            'WEAK': -1.5,
            'NEUTRAL': 0.0
        }
        return momentum_adjustments.get(momentum, 0.0)
    
    def _calculate_enhanced_confidence(self, historical_returns: list, features: dict) -> float:
        """Calculate enhanced prediction confidence."""
        if not historical_returns:
            return 0.6  # Higher base confidence
        
        # Consistency of historical returns
        variance = sum((r - sum(historical_returns)/len(historical_returns))**2 for r in historical_returns) / len(historical_returns)
        consistency = max(0, 1 - (variance / 50))  # Lower variance = higher consistency
        
        # Feature strength
        feature_strength = min(1.0, abs(features['daily_return']) / 3.0)  # Stronger moves = higher confidence
        
        # Trend strength
        trend_confidence = min(1.0, features['trend_strength'] / 2.0)
        
        # Volume confidence
        volume_confidence = min(1.0, features['volume'] / self.volume_threshold)
        
        # Combined confidence
        confidence = (0.4 * consistency) + (0.3 * feature_strength) + (0.2 * trend_confidence) + (0.1 * volume_confidence)
        
        return min(0.95, max(0.3, confidence))  # Clamp between 0.3 and 0.95
    
    def _generate_enhanced_reasoning(self, features: dict, similar_cases: list) -> str:
        """Generate enhanced reasoning for the prediction."""
        reasoning_parts = []
        
        # Current market conditions
        reasoning_parts.append(f"Sentiment: {features['sentiment']}")
        reasoning_parts.append(f"Daily return: {features['daily_return']:+.1f}%")
        reasoning_parts.append(f"Momentum: {features['momentum']}")
        
        if features['price_range'] > self.high_volatility:
            reasoning_parts.append(f"High volatility: {features['price_range']:.1f}%")
        
        if features['volume'] > self.volume_threshold:
            reasoning_parts.append(f"High volume: {features['volume']:,.0f}")
        
        if abs(features['oi_change']) > self.oi_change_threshold:
            reasoning_parts.append(f"OI change: {features['oi_change']:+.1f}%")
        
        if features['trend_strength'] > 1.0:
            reasoning_parts.append(f"Strong trend: {features['trend_strength']:.1f}")
        
        # Historical context
        if similar_cases:
            avg_return = sum(case.get('next_day_return', 0) for case in similar_cases) / len(similar_cases)
            reasoning_parts.append(f"Based on {len(similar_cases)} similar cases (avg: {avg_return:+.1f}%)")
        
        return " | ".join(reasoning_parts)
    
    def run_enhanced_real_time_analysis(self, symbols: list = None, interval_minutes: int = 5):
        """Run enhanced continuous real-time analysis."""
        if symbols is None:
            symbols = self.key_symbols
        
        logger.info(f"🚀 Starting enhanced real-time analysis for {len(symbols)} symbols")
        logger.info(f"📊 Analysis interval: {interval_minutes} minutes")
        
        while True:
            try:
                current_time = datetime.now()
                logger.info(f"\n🕐 Enhanced Real-time Analysis - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info("=" * 80)
                
                all_predictions = []
                
                for symbol in symbols:
                    # Get latest data and history
                    latest_data = self.get_latest_market_data(symbol)
                    if not latest_data:
                        logger.warning(f"⚠️ No data available for {symbol}")
                        continue
                    
                    history = self.get_recent_price_history(symbol, days=5)
                    
                    # Calculate enhanced features
                    features = self.calculate_enhanced_features(latest_data, history)
                    if not features:
                        continue
                    
                    # Find similar cases
                    similar_cases = self.get_similar_historical_cases(symbol, features)
                    
                    # Make prediction
                    prediction = self.predict_next_day_movement(symbol, features, similar_cases)
                    if prediction:
                        all_predictions.append(prediction)
                        
                        # Display enhanced prediction
                        self._display_enhanced_prediction(prediction, features)
                
                # Enhanced summary
                self._display_enhanced_summary(all_predictions)
                
                # Wait for next interval
                logger.info(f"⏳ Next analysis in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("🛑 Enhanced real-time analysis stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in enhanced real-time analysis: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _display_enhanced_prediction(self, prediction: dict, features: dict):
        """Display an enhanced single prediction."""
        symbol = prediction['symbol']
        direction_emoji = "📈" if prediction['direction'] == 'UP' else "📉" if prediction['direction'] == 'DOWN' else "➡️"
        confidence_emoji = "🔥" if prediction['confidence'] > 0.8 else "⚡" if prediction['confidence'] > 0.6 else "💡"
        
        print(f"\n{direction_emoji} {symbol} Enhanced Prediction:")
        print(f"   📊 Predicted Return: {prediction['predicted_return']:+.2f}%")
        print(f"   {confidence_emoji} Confidence: {prediction['confidence']:.1%}")
        print(f"   🎯 Direction: {prediction['direction']}")
        print(f"   📈 Current Price: ₹{features['close_price']:.2f}")
        print(f"   📊 Daily Return: {features['daily_return']:+.2f}%")
        print(f"   🧠 Sentiment: {features['sentiment']}")
        print(f"   ⚡ Momentum: {features['momentum']}")
        print(f"   📈 Trend Strength: {features['trend_strength']:.1f}")
        print(f"   📋 Reasoning: {prediction['reasoning']}")
    
    def _display_enhanced_summary(self, predictions: list):
        """Display enhanced analysis summary."""
        if not predictions:
            return
        
        print(f"\n📊 Enhanced Analysis Summary:")
        print("=" * 50)
        
        # Count directions
        up_count = sum(1 for p in predictions if p['direction'] == 'UP')
        down_count = sum(1 for p in predictions if p['direction'] == 'DOWN')
        flat_count = sum(1 for p in predictions if p['direction'] == 'FLAT')
        
        print(f"📈 Bullish: {up_count} symbols")
        print(f"📉 Bearish: {down_count} symbols")
        print(f"➡️ Neutral: {flat_count} symbols")
        
        # High confidence predictions
        high_confidence = [p for p in predictions if p['confidence'] > 0.7]
        if high_confidence:
            print(f"\n🔥 High Confidence Predictions (>70%):")
            for pred in high_confidence:
                direction_emoji = "📈" if pred['direction'] == 'UP' else "📉" if pred['direction'] == 'DOWN' else "➡️"
                print(f"   {direction_emoji} {pred['symbol']}: {pred['predicted_return']:+.2f}% ({pred['confidence']:.1%})")
        
        # Top movers
        top_movers = sorted(predictions, key=lambda x: abs(x['predicted_return']), reverse=True)[:3]
        print(f"\n🚀 Top Expected Movers:")
        for pred in top_movers:
            direction_emoji = "📈" if pred['direction'] == 'UP' else "📉" if pred['direction'] == 'DOWN' else "➡️"
            print(f"   {direction_emoji} {pred['symbol']}: {pred['predicted_return']:+.2f}%")

def main():
    """Main function for enhanced real-time analysis."""
    print("🚀 Enhanced Real-Time FNO Analysis with Realistic Predictions")
    print("=" * 70)
    
    # Initialize enhanced predictor
    predictor = EnhancedRealTimeFNOPredictor()
    
    # Get user input
    print("\n📊 Enhanced Analysis Options:")
    print("1. Run continuous enhanced real-time analysis")
    print("2. Single enhanced analysis for all symbols")
    print("3. Enhanced analysis for specific symbol")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        interval = int(input("Enter analysis interval in minutes (default 5): ") or "5")
        symbols_input = input("Enter symbols to analyze (comma-separated, or press Enter for default): ").strip()
        
        if symbols_input:
            symbols = [s.strip().upper() for s in symbols_input.split(',')]
        else:
            symbols = None
        
        predictor.run_enhanced_real_time_analysis(symbols, interval)
        
    elif choice == "2":
        # Single enhanced analysis
        all_predictions = []
        
        for symbol in predictor.key_symbols:
            latest_data = predictor.get_latest_market_data(symbol)
            if latest_data:
                history = predictor.get_recent_price_history(symbol, days=5)
                features = predictor.calculate_enhanced_features(latest_data, history)
                if features:
                    similar_cases = predictor.get_similar_historical_cases(symbol, features)
                    prediction = predictor.predict_next_day_movement(symbol, features, similar_cases)
                    if prediction:
                        all_predictions.append(prediction)
                        predictor._display_enhanced_prediction(prediction, features)
        
        predictor._display_enhanced_summary(all_predictions)
        
    elif choice == "3":
        symbol = input("Enter symbol to analyze: ").strip().upper()
        latest_data = predictor.get_latest_market_data(symbol)
        
        if latest_data:
            history = predictor.get_recent_price_history(symbol, days=5)
            features = predictor.calculate_enhanced_features(latest_data, history)
            if features:
                similar_cases = predictor.get_similar_historical_cases(symbol, features)
                prediction = predictor.predict_next_day_movement(symbol, features, similar_cases)
                if prediction:
                    predictor._display_enhanced_prediction(prediction, features)
                else:
                    print(f"❌ Could not generate prediction for {symbol}")
            else:
                print(f"❌ Could not calculate features for {symbol}")
        else:
            print(f"❌ No data available for {symbol}")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
