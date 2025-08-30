#!/usr/bin/env python3
"""
Real-Time FNO Analysis with Realistic Predictions
Live market insights using the enhanced FNO RAG system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import time
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
import duckdb
from build_enhanced_vector_store import EnhancedFNOVectorStore
from src.fno_rag import FNOEngine

class RealTimeFNOPredictor:
    """Real-time FNO analysis with realistic predictions."""
    
    def __init__(self):
        """Initialize the real-time predictor."""
        self.vector_store = EnhancedFNOVectorStore()
        self.fno_engine = FNOEngine()
        self.db_manager = self.vector_store.db_manager
        
        # Key symbols to monitor
        self.key_symbols = [
            'NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS', 'HDFC', 'INFY', 
            'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'AXISBANK'
        ]
        
        # Analysis thresholds
        self.pcr_threshold_high = 1.5  # High PCR indicates bearish sentiment
        self.pcr_threshold_low = 0.7   # Low PCR indicates bullish sentiment
        self.oi_change_threshold = 10   # Significant OI change percentage
        
        logger.info("🚀 Real-time FNO Predictor initialized")
    
    def get_latest_market_data(self, symbol: str) -> dict:
        """Get the latest market data for a symbol."""
        try:
            # Get the most recent trading day
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
    
    def calculate_real_time_features(self, data: dict) -> dict:
        """Calculate real-time features for prediction."""
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
            
            # Price range
            price_range = ((high_price - low_price) / close_price * 100) if close_price > 0 else 0
            
            # Volume analysis
            volume = data['TtlTradgVol']
            oi_change = data['ChngInOpnIntrst']
            
            # PCR calculation (simplified for real-time)
            pcr = 1.0  # Default PCR, would need options data for accurate calculation
            
            # Market sentiment indicators
            sentiment = self._calculate_sentiment(daily_return, pcr, oi_change, price_range)
            
            return {
                'symbol': data['TckrSymb'],
                'date': data['TRADE_DATE'],
                'close_price': close_price,
                'daily_return': daily_return,
                'price_range': price_range,
                'volume': volume,
                'oi_change': oi_change,
                'pcr': pcr,
                'sentiment': sentiment,
                'high_price': high_price,
                'low_price': low_price
            }
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return None
    
    def _calculate_sentiment(self, daily_return: float, pcr: float, oi_change: float, price_range: float) -> str:
        """Calculate market sentiment based on multiple factors."""
        sentiment_score = 0
        
        # Price movement sentiment
        if daily_return > 1.0:
            sentiment_score += 2
        elif daily_return > 0.5:
            sentiment_score += 1
        elif daily_return < -1.0:
            sentiment_score -= 2
        elif daily_return < -0.5:
            sentiment_score -= 1
        
        # PCR sentiment
        if pcr > self.pcr_threshold_high:
            sentiment_score -= 1  # Bearish
        elif pcr < self.pcr_threshold_low:
            sentiment_score += 1  # Bullish
        
        # OI change sentiment
        if oi_change > self.oi_change_threshold:
            sentiment_score += 1
        elif oi_change < -self.oi_change_threshold:
            sentiment_score -= 1
        
        # Price range sentiment (volatility)
        if price_range > 3.0:
            sentiment_score -= 1  # High volatility often bearish
        
        # Determine sentiment
        if sentiment_score >= 2:
            return "STRONG_BULLISH"
        elif sentiment_score >= 1:
            return "BULLISH"
        elif sentiment_score <= -2:
            return "STRONG_BEARISH"
        elif sentiment_score <= -1:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def get_similar_historical_cases(self, symbol: str, features: dict, top_k: int = 5) -> list:
        """Find similar historical cases for prediction."""
        try:
            # Create a query based on current market conditions
            query = f"Find similar cases where {symbol} had {features['sentiment'].lower()} sentiment with {features['daily_return']:+.1f}% daily return"
            
            # Use the enhanced vector store to find similar cases
            similar_cases = self.vector_store.query_rag_system(query, top_k=top_k)
            
            return similar_cases
            
        except Exception as e:
            logger.error(f"Error finding similar cases: {e}")
            return []
    
    def predict_next_day_movement(self, symbol: str, features: dict, similar_cases: list) -> dict:
        """Predict next day movement based on current features and historical cases."""
        try:
            # Base prediction from current sentiment
            base_prediction = self._get_base_prediction(features['sentiment'])
            
            # Adjust based on similar historical cases
            if similar_cases:
                historical_returns = []
                for case in similar_cases:
                    if 'next_day_return' in case:
                        historical_returns.append(case['next_day_return'])
                
                if historical_returns:
                    avg_historical_return = sum(historical_returns) / len(historical_returns)
                    
                    # Weighted prediction: 70% historical, 30% current sentiment
                    weighted_return = (0.7 * avg_historical_return) + (0.3 * base_prediction['return'])
                    
                    # Confidence based on consistency
                    confidence = self._calculate_confidence(historical_returns, features)
                    
                    return {
                        'symbol': symbol,
                        'predicted_return': weighted_return,
                        'confidence': confidence,
                        'direction': 'UP' if weighted_return > 0 else 'DOWN' if weighted_return < 0 else 'FLAT',
                        'base_prediction': base_prediction,
                        'historical_cases': len(similar_cases),
                        'avg_historical_return': avg_historical_return,
                        'reasoning': self._generate_reasoning(features, similar_cases)
                    }
            
            # Fallback to base prediction
            return {
                'symbol': symbol,
                'predicted_return': base_prediction['return'],
                'confidence': 0.5,
                'direction': base_prediction['direction'],
                'base_prediction': base_prediction,
                'historical_cases': 0,
                'avg_historical_return': 0,
                'reasoning': f"Based on {features['sentiment']} sentiment with {features['daily_return']:+.1f}% daily return"
            }
            
        except Exception as e:
            logger.error(f"Error predicting movement for {symbol}: {e}")
            return None
    
    def _get_base_prediction(self, sentiment: str) -> dict:
        """Get base prediction from sentiment."""
        sentiment_predictions = {
            'STRONG_BULLISH': {'return': 1.5, 'direction': 'UP'},
            'BULLISH': {'return': 0.8, 'direction': 'UP'},
            'NEUTRAL': {'return': 0.0, 'direction': 'FLAT'},
            'BEARISH': {'return': -0.8, 'direction': 'DOWN'},
            'STRONG_BEARISH': {'return': -1.5, 'direction': 'DOWN'}
        }
        
        return sentiment_predictions.get(sentiment, {'return': 0.0, 'direction': 'FLAT'})
    
    def _calculate_confidence(self, historical_returns: list, features: dict) -> float:
        """Calculate prediction confidence."""
        if not historical_returns:
            return 0.5
        
        # Consistency of historical returns
        variance = sum((r - sum(historical_returns)/len(historical_returns))**2 for r in historical_returns) / len(historical_returns)
        consistency = max(0, 1 - (variance / 100))  # Lower variance = higher consistency
        
        # Feature strength
        feature_strength = min(1.0, abs(features['daily_return']) / 2.0)  # Stronger moves = higher confidence
        
        # Combined confidence
        confidence = (0.6 * consistency) + (0.4 * feature_strength)
        
        return min(0.95, max(0.1, confidence))  # Clamp between 0.1 and 0.95
    
    def _generate_reasoning(self, features: dict, similar_cases: list) -> str:
        """Generate reasoning for the prediction."""
        reasoning_parts = []
        
        # Current market conditions
        reasoning_parts.append(f"Current sentiment: {features['sentiment']}")
        reasoning_parts.append(f"Daily return: {features['daily_return']:+.1f}%")
        
        if features['pcr'] != 1.0:
            reasoning_parts.append(f"PCR: {features['pcr']:.2f}")
        
        if abs(features['oi_change']) > self.oi_change_threshold:
            reasoning_parts.append(f"OI change: {features['oi_change']:+.1f}%")
        
        # Historical context
        if similar_cases:
            avg_return = sum(case.get('next_day_return', 0) for case in similar_cases) / len(similar_cases)
            reasoning_parts.append(f"Based on {len(similar_cases)} similar historical cases (avg: {avg_return:+.1f}%)")
        
        return " | ".join(reasoning_parts)
    
    def run_real_time_analysis(self, symbols: list = None, interval_minutes: int = 5):
        """Run continuous real-time analysis."""
        if symbols is None:
            symbols = self.key_symbols
        
        logger.info(f"🚀 Starting real-time analysis for {len(symbols)} symbols")
        logger.info(f"📊 Analysis interval: {interval_minutes} minutes")
        
        while True:
            try:
                current_time = datetime.now()
                logger.info(f"\n🕐 Real-time Analysis - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info("=" * 80)
                
                all_predictions = []
                
                for symbol in symbols:
                    # Get latest data
                    latest_data = self.get_latest_market_data(symbol)
                    if not latest_data:
                        logger.warning(f"⚠️ No data available for {symbol}")
                        continue
                    
                    # Calculate features
                    features = self.calculate_real_time_features(latest_data)
                    if not features:
                        continue
                    
                    # Find similar cases
                    similar_cases = self.get_similar_historical_cases(symbol, features)
                    
                    # Make prediction
                    prediction = self.predict_next_day_movement(symbol, features, similar_cases)
                    if prediction:
                        all_predictions.append(prediction)
                        
                        # Display prediction
                        self._display_prediction(prediction, features)
                
                # Summary
                self._display_summary(all_predictions)
                
                # Wait for next interval
                logger.info(f"⏳ Next analysis in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("🛑 Real-time analysis stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in real-time analysis: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _display_prediction(self, prediction: dict, features: dict):
        """Display a single prediction."""
        symbol = prediction['symbol']
        direction_emoji = "📈" if prediction['direction'] == 'UP' else "📉" if prediction['direction'] == 'DOWN' else "➡️"
        confidence_emoji = "🔥" if prediction['confidence'] > 0.8 else "⚡" if prediction['confidence'] > 0.6 else "💡"
        
        print(f"\n{direction_emoji} {symbol} Prediction:")
        print(f"   📊 Predicted Return: {prediction['predicted_return']:+.2f}%")
        print(f"   {confidence_emoji} Confidence: {prediction['confidence']:.1%}")
        print(f"   🎯 Direction: {prediction['direction']}")
        print(f"   📈 Current Price: ₹{features['close_price']:.2f}")
        print(f"   📊 Daily Return: {features['daily_return']:+.2f}%")
        print(f"   🧠 Sentiment: {features['sentiment']}")
        print(f"   📋 Reasoning: {prediction['reasoning']}")
    
    def _display_summary(self, predictions: list):
        """Display analysis summary."""
        if not predictions:
            return
        
        print(f"\n📊 Analysis Summary:")
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
    """Main function for real-time analysis."""
    print("🚀 Real-Time FNO Analysis with Realistic Predictions")
    print("=" * 60)
    
    # Initialize predictor
    predictor = RealTimeFNOPredictor()
    
    # Get user input
    print("\n📊 Analysis Options:")
    print("1. Run continuous real-time analysis")
    print("2. Single analysis for all symbols")
    print("3. Analysis for specific symbol")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        interval = int(input("Enter analysis interval in minutes (default 5): ") or "5")
        symbols_input = input("Enter symbols to analyze (comma-separated, or press Enter for default): ").strip()
        
        if symbols_input:
            symbols = [s.strip().upper() for s in symbols_input.split(',')]
        else:
            symbols = None
        
        predictor.run_real_time_analysis(symbols, interval)
        
    elif choice == "2":
        # Single analysis
        all_predictions = []
        
        for symbol in predictor.key_symbols:
            latest_data = predictor.get_latest_market_data(symbol)
            if latest_data:
                features = predictor.calculate_real_time_features(latest_data)
                if features:
                    similar_cases = predictor.get_similar_historical_cases(symbol, features)
                    prediction = predictor.predict_next_day_movement(symbol, features, similar_cases)
                    if prediction:
                        all_predictions.append(prediction)
                        predictor._display_prediction(prediction, features)
        
        predictor._display_summary(all_predictions)
        
    elif choice == "3":
        symbol = input("Enter symbol to analyze: ").strip().upper()
        latest_data = predictor.get_latest_market_data(symbol)
        
        if latest_data:
            features = predictor.calculate_real_time_features(latest_data)
            if features:
                similar_cases = predictor.get_similar_historical_cases(symbol, features)
                prediction = predictor.predict_next_day_movement(symbol, features, similar_cases)
                if prediction:
                    predictor._display_prediction(prediction, features)
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
