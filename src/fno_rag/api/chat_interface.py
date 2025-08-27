#!/usr/bin/env python3
"""
FNO Chat Interface
Provides natural language interface for FNO probability predictions.
"""

import os
import json
import requests
from typing import Dict, List, Optional, Any
import logging
from loguru import logger
from datetime import datetime

from ..models.data_models import (
    HorizonType, ProbabilityResult, PredictionRequest, 
    FNOSearchQuery, FNOSearchResult
)
from ..core.probability_predictor import FNOProbabilityPredictor


class FNOChatInterface:
    """Natural language interface for FNO analysis."""
    
    def __init__(self, groq_api_key: Optional[str] = None):
        """Initialize the chat interface."""
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        self.base_url = "https://api.groq.com/openai/v1"
        self.predictor = FNOProbabilityPredictor()
        self.logger = logger
        
        if not self.groq_api_key:
            self.logger.warning("GROQ_API_KEY not found. Chat interface will be limited.")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language query about FNO stocks."""
        try:
            self.logger.info(f"Processing query: {query}")
            
            # Parse query to extract intent and parameters
            parsed_query = self._parse_query(query)
            
            if parsed_query['intent'] == 'current_price':
                return self._handle_current_price(parsed_query)
            elif parsed_query['intent'] == 'probability_prediction':
                return self._handle_probability_prediction(parsed_query)
            elif parsed_query['intent'] == 'stock_search':
                return self._handle_stock_search(parsed_query)
            elif parsed_query['intent'] == 'system_status':
                return self._handle_system_status()
            else:
                return self._handle_general_query(query)
                
        except Exception as e:
            self.logger.error(f"Failed to process query: {e}")
            return {
                'error': True,
                'message': f"Sorry, I encountered an error: {str(e)}",
                'query': query
            }
    
    def _parse_query(self, query: str) -> Dict[str, Any]:
        """Parse natural language query to extract intent and parameters."""
        try:
            query_lower = query.lower()
            
            # Initialize parsed query
            parsed = {
                'intent': 'general',
                'symbols': [],
                'horizon': None,
                'direction': None,
                'min_probability': 0.1,
                'max_results': 10
            }
            
            # Check for current price/value intent first
            if any(word in query_lower for word in ['value', 'price', 'current', 'now', 'today', 'latest']):
                parsed['intent'] = 'current_price'
                
                # Extract symbols for current price
                symbols = self._extract_symbols(query)
                if symbols:
                    parsed['symbols'] = symbols
                else:
                    # If no specific symbols mentioned, try to extract from common patterns
                    if 'nifty' in query_lower:
                        parsed['symbols'] = ['NIFTY']
                    elif 'banknifty' in query_lower or 'bank nifty' in query_lower:
                        parsed['symbols'] = ['BANKNIFTY']
                    elif 'reliance' in query_lower:
                        parsed['symbols'] = ['RELIANCE']
                    elif 'tcs' in query_lower:
                        parsed['symbols'] = ['TCS']
                    elif 'infy' in query_lower or 'infosys' in query_lower:
                        parsed['symbols'] = ['INFY']
                
                return parsed
            
            # Check for probability prediction intent
            if any(word in query_lower for word in ['probability', 'chance', 'move', 'tomorrow', 'week', 'month', 'predict', 'forecast']):
                parsed['intent'] = 'probability_prediction'
                
                # Extract symbols
                symbols = self._extract_symbols(query)
                if symbols:
                    parsed['symbols'] = symbols
                else:
                    # If no specific symbols mentioned, try to extract from common patterns
                    if 'nifty' in query_lower:
                        parsed['symbols'] = ['NIFTY']
                    elif 'banknifty' in query_lower or 'bank nifty' in query_lower:
                        parsed['symbols'] = ['BANKNIFTY']
                    elif 'reliance' in query_lower:
                        parsed['symbols'] = ['RELIANCE']
                    elif 'tcs' in query_lower:
                        parsed['symbols'] = ['TCS']
                    elif 'infy' in query_lower or 'infosys' in query_lower:
                        parsed['symbols'] = ['INFY']
                
                # Extract horizon
                if 'tomorrow' in query_lower or 'daily' in query_lower or 'today' in query_lower:
                    parsed['horizon'] = HorizonType.DAILY
                elif 'week' in query_lower or 'weekly' in query_lower:
                    parsed['horizon'] = HorizonType.WEEKLY
                elif 'month' in query_lower or 'monthly' in query_lower:
                    parsed['horizon'] = HorizonType.MONTHLY
                else:
                    # Default to daily if no horizon specified
                    parsed['horizon'] = HorizonType.DAILY
                
                # Extract direction
                if 'up' in query_lower or 'rise' in query_lower or 'gain' in query_lower or 'positive' in query_lower:
                    parsed['direction'] = 'up'
                elif 'down' in query_lower or 'fall' in query_lower or 'drop' in query_lower or 'negative' in query_lower:
                    parsed['direction'] = 'down'
            
            # Check for stock search intent
            elif any(word in query_lower for word in ['find', 'search', 'which', 'stocks', 'fno', 'give me', 'show me', 'list']):
                parsed['intent'] = 'stock_search'
                
                # Extract horizon
                if 'tomorrow' in query_lower or 'daily' in query_lower or 'today' in query_lower:
                    parsed['horizon'] = HorizonType.DAILY
                elif 'week' in query_lower or 'weekly' in query_lower:
                    parsed['horizon'] = HorizonType.WEEKLY
                elif 'month' in query_lower or 'monthly' in query_lower:
                    parsed['horizon'] = HorizonType.MONTHLY
                else:
                    # Default to daily if no horizon specified
                    parsed['horizon'] = HorizonType.DAILY
                
                # Extract direction
                if 'up' in query_lower or 'rise' in query_lower or 'gain' in query_lower or 'positive' in query_lower:
                    parsed['direction'] = 'up'
                elif 'down' in query_lower or 'fall' in query_lower or 'drop' in query_lower or 'negative' in query_lower:
                    parsed['direction'] = 'down'
                
                # Extract percentage
                if '3%' in query or '3 percent' in query_lower:
                    parsed['min_probability'] = 0.3
                elif '5%' in query or '5 percent' in query_lower:
                    parsed['min_probability'] = 0.5
                elif '10%' in query or '10 percent' in query_lower:
                    parsed['min_probability'] = 0.1
                else:
                    # Default to 0.1 if no percentage specified
                    parsed['min_probability'] = 0.1
            
            # Check for system status intent
            elif any(word in query_lower for word in ['status', 'health', 'working', 'system']):
                parsed['intent'] = 'system_status'
            
            return parsed
            
        except Exception as e:
            self.logger.error(f"Failed to parse query: {e}")
            return {'intent': 'general'}
    
    def _extract_symbols(self, query: str) -> List[str]:
        """Extract stock symbols from query."""
        try:
            # Common stock symbols to look for
            common_symbols = [
                'RELIANCE', 'TCS', 'HDFC', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN',
                'BHARTIARTL', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'HCLTECH', 'SUNPHARMA',
                'ULTRACEMCO', 'TITAN', 'BAJFINANCE', 'WIPRO', 'NESTLEIND', 'POWERGRID',
                'TECHM', 'BAJAJFINSV', 'ADANIENT', 'JSWSTEEL', 'ONGC', 'COALINDIA',
                'TATAMOTORS', 'NTPC', 'INDUSINDBK', 'CIPLA', 'SHREECEM', 'DIVISLAB',
                'BRITANNIA', 'EICHERMOT', 'DRREDDY', 'HEROMOTOCO', 'KOTAKBANK', 'LT',
                'TATACONSUM', 'APOLLOHOSP', 'ADANIPORTS', 'M&M', 'BPCL', 'TATASTEEL',
                'VEDL', 'GRASIM', 'HINDALCO', 'UPL', 'SBILIFE', 'HDFCLIFE', 'ICICIGI',
                'BAJAJ-AUTO', 'TATAPOWER', 'SHRIRAMFIN', 'BERGEPAINT', 'DABUR', 'COLPAL'
            ]
            
            found_symbols = []
            query_upper = query.upper()
            
            for symbol in common_symbols:
                if symbol in query_upper:
                    found_symbols.append(symbol)
            
            return found_symbols
            
        except Exception as e:
            self.logger.error(f"Failed to extract symbols: {e}")
            return []
    
    def _handle_probability_prediction(self, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Handle probability prediction queries."""
        try:
            symbols = parsed_query['symbols']
            horizon = parsed_query['horizon'] or HorizonType.DAILY
            
            if not symbols:
                return {
                    'error': True,
                    'message': "Please specify which stock(s) you want to analyze. For example: 'What's the probability of RELIANCE moving up tomorrow?'"
                }
            
            results = []
            for symbol in symbols:
                try:
                    request = PredictionRequest(
                        symbol=symbol,
                        horizon=horizon,
                        include_explanations=True,
                        top_k_similar=5
                    )
                    
                    result = self.predictor.predict_probability(request)
                    results.append(result)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to predict for {symbol}: {e}")
                    continue
            
            if not results:
                return {
                    'error': True,
                    'message': f"Sorry, I couldn't generate predictions for the requested stocks."
                }
            
            # Format response
            response = self._format_probability_response(results, parsed_query)
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to handle probability prediction: {e}")
            return {
                'error': True,
                'message': f"Sorry, I encountered an error while generating predictions: {str(e)}"
            }
    
    def _handle_stock_search(self, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stock search queries."""
        try:
            horizon = parsed_query['horizon'] or HorizonType.DAILY
            direction = parsed_query['direction']
            min_probability = parsed_query['min_probability']
            
            # Create search query
            search_query = self._create_search_query(parsed_query)
            
            # Search for stocks
            results = self.predictor.search_stocks_by_probability(
                query=search_query,
                horizon=horizon,
                min_probability=min_probability,
                max_results=parsed_query['max_results']
            )
            
            if not results:
                return {
                    'error': True,
                    'message': f"Sorry, I couldn't find any stocks matching your criteria."
                }
            
            # Format response
            response = self._format_search_response(results, parsed_query)
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to handle stock search: {e}")
            return {
                'error': True,
                'message': f"Sorry, I encountered an error while searching for stocks: {str(e)}"
            }
    
    def _handle_system_status(self) -> Dict[str, Any]:
        """Handle system status queries."""
        try:
            status = self.predictor.get_system_status()
            
            response = {
                'type': 'system_status',
                'status': status,
                'message': self._format_status_message(status)
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to handle system status: {e}")
            return {
                'error': True,
                'message': f"Sorry, I couldn't retrieve system status: {str(e)}"
            }
    
    def _handle_current_price(self, parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Handle current price queries."""
        try:
            symbols = parsed_query.get('symbols', [])
            if not symbols:
                return {
                    'error': True,
                    'message': "Please specify which stock or index you want to know the current price for."
                }
            
            # Get current prices from the data processor
            current_prices = []
            for symbol in symbols:
                try:
                    # Try to get current price from the data processor
                    price_info = self._get_current_price(symbol)
                    current_prices.append(price_info)
                except Exception as e:
                    self.logger.error(f"Failed to get price for {symbol}: {e}")
                    current_prices.append({
                        'symbol': symbol,
                        'error': f"Could not retrieve current price for {symbol}"
                    })
            
            # Format response
            message_parts = []
            for price_info in current_prices:
                if 'error' in price_info:
                    message_parts.append(f"❌ {price_info['error']}")
                else:
                    message_parts.append(f"📊 {price_info['symbol']}: ₹{price_info['price']:.2f} ({price_info['change']:+.2f}%)")
            
            message = "\n".join(message_parts)
            
            return {
                'type': 'current_price',
                'message': message,
                'prices': current_prices,
                'query': parsed_query
            }
            
        except Exception as e:
            self.logger.error(f"Failed to handle current price query: {e}")
            return {
                'error': True,
                'message': f"Sorry, I couldn't get the current prices: {str(e)}"
            }
    
    def _get_current_price(self, symbol: str) -> Dict[str, Any]:
        """Get current price for a symbol from the database."""
        try:
            from datetime import datetime
            import duckdb
            
            # Connect to the database
            db_path = "data/comprehensive_equity.duckdb"
            
            # Query to get the latest price from fno_bav_copy table
            query = """
            SELECT 
                symbol,
                close_price as price,
                ((close_price - prev_close_price) / prev_close_price * 100) as change_percent,
                timestamp
            FROM fno_bav_copy 
            WHERE symbol = ? 
            ORDER BY timestamp DESC 
            LIMIT 1
            """
            
            with duckdb.connect(db_path) as conn:
                result = conn.execute(query, [symbol]).fetchone()
                
                if result:
                    symbol_name, price, change_percent, timestamp = result
                    return {
                        'symbol': symbol_name,
                        'price': float(price) if price else 0.0,
                        'change': float(change_percent) if change_percent else 0.0,
                        'timestamp': timestamp.isoformat() if timestamp else datetime.now().isoformat()
                    }
                else:
                    # If not found in fno_bav_copy, try to get from price_data table
                    query_price_data = """
                    SELECT 
                        symbol,
                        close as price,
                        ((close - prev_close) / prev_close * 100) as change_percent,
                        date
                    FROM price_data 
                    WHERE symbol = ? 
                    ORDER BY date DESC 
                    LIMIT 1
                    """
                    
                    result_price = conn.execute(query_price_data, [symbol]).fetchone()
                    
                    if result_price:
                        symbol_name, price, change_percent, date = result_price
                        return {
                            'symbol': symbol_name,
                            'price': float(price) if price else 0.0,
                            'change': float(change_percent) if change_percent else 0.0,
                            'timestamp': date.isoformat() if date else datetime.now().isoformat()
                        }
                    else:
                        # If still not found, return error
                        return {
                            'symbol': symbol,
                            'error': f"No data found for {symbol} in database"
                        }
                
        except Exception as e:
            self.logger.error(f"Failed to get current price for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': f"Database error: {str(e)}"
            }
    
    def _handle_general_query(self, query: str) -> Dict[str, Any]:
        """Handle general queries using GROQ API."""
        try:
            if not self.groq_api_key:
                return {
                    'error': True,
                    'message': "I can help you with FNO analysis, but I need a GROQ API key for general questions. Please ask me about specific stocks or probability predictions."
                }
            
            # Use GROQ API for general questions
            response = self._call_groq_api(query)
            return {
                'type': 'general_response',
                'message': response,
                'query': query
            }
            
        except Exception as e:
            self.logger.error(f"Failed to handle general query: {e}")
            return {
                'error': True,
                'message': f"Sorry, I couldn't process your query: {str(e)}"
            }
    
    def _format_probability_response(self, results: List[ProbabilityResult], 
                                   parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Format probability prediction response."""
        try:
            horizon = parsed_query['horizon'] or HorizonType.DAILY
            direction = parsed_query['direction']
            
            # Create response message
            message_parts = []
            
            for result in results:
                symbol = result.symbol
                up_prob = result.up_probability * 100
                down_prob = result.down_probability * 100
                neutral_prob = result.neutral_probability * 100
                confidence = result.confidence_score * 100
                
                if direction == 'up':
                    message_parts.append(f"{symbol}: Up={up_prob:.1f}%, Down={down_prob:.1f}%, Neutral={neutral_prob:.1f}%")
                elif direction == 'down':
                    message_parts.append(f"{symbol}: Up={up_prob:.1f}%, Down={down_prob:.1f}%, Neutral={neutral_prob:.1f}%")
                else:
                    message_parts.append(f"{symbol}: Up={up_prob:.1f}%, Down={down_prob:.1f}%, Neutral={neutral_prob:.1f}%")
                
                # Add confidence
                message_parts[-1] += f" (Confidence: {confidence:.1f}%)"
            
            message = "\n".join(message_parts)
            
            # Add horizon context
            horizon_text = {
                HorizonType.DAILY: "tomorrow",
                HorizonType.WEEKLY: "this week", 
                HorizonType.MONTHLY: "this month"
            }[horizon]
            
            message = f"Probability of {horizon_text} movement:\n{message}"
            
            return {
                'type': 'probability_prediction',
                'message': message,
                'results': [self._result_to_dict(result) for result in results],
                'horizon': horizon.value,
                'direction': direction
            }
            
        except Exception as e:
            self.logger.error(f"Failed to format probability response: {e}")
            return {
                'error': True,
                'message': "Sorry, I couldn't format the probability results."
            }
    
    def _format_search_response(self, results: List[ProbabilityResult], 
                              parsed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Format stock search response."""
        try:
            horizon = parsed_query['horizon'] or HorizonType.DAILY
            direction = parsed_query['direction']
            
            # Create response message
            message_parts = []
            
            for i, result in enumerate(results, 1):
                symbol = result.symbol
                up_prob = result.up_probability * 100
                down_prob = result.down_probability * 100
                neutral_prob = result.neutral_probability * 100
                
                if direction == 'up':
                    message_parts.append(f"{i}. {symbol}: Up={up_prob:.1f}%, Down={down_prob:.1f}%, Neutral={neutral_prob:.1f}%")
                elif direction == 'down':
                    message_parts.append(f"{i}. {symbol}: Up={up_prob:.1f}%, Down={down_prob:.1f}%, Neutral={neutral_prob:.1f}%")
                else:
                    message_parts.append(f"{i}. {symbol}: Up={up_prob:.1f}%, Down={down_prob:.1f}%, Neutral={neutral_prob:.1f}%")
            
            message = "\n".join(message_parts)
            
            # Add context
            horizon_text = {
                HorizonType.DAILY: "tomorrow",
                HorizonType.WEEKLY: "this week",
                HorizonType.MONTHLY: "this month"
            }[horizon]
            
            if direction:
                direction_text = "up" if direction == 'up' else "down"
                message = f"FNO stocks that can move {direction_text} {horizon_text}:\n{message}"
            else:
                message = f"FNO stocks with high probability moves {horizon_text}:\n{message}"
            
            return {
                'type': 'stock_search',
                'message': message,
                'results': [self._result_to_dict(result) for result in results],
                'horizon': horizon.value,
                'direction': direction
            }
            
        except Exception as e:
            self.logger.error(f"Failed to format search response: {e}")
            return {
                'error': True,
                'message': "Sorry, I couldn't format the search results."
            }
    
    def _format_status_message(self, status: Dict[str, Any]) -> str:
        """Format system status message."""
        try:
            message_parts = ["System Status:"]
            
            # ML Models
            ml_status = status.get('ml_models', {})
            for horizon, info in ml_status.items():
                loaded = "✅" if info.get('loaded') else "❌"
                message_parts.append(f"  {horizon} ML Model: {loaded}")
            
            # Vector Store
            vs_status = status.get('vector_store', {})
            vs_loaded = "✅" if vs_status.get('loaded') else "❌"
            vs_stats = vs_status.get('stats', {})
            message_parts.append(f"  Vector Store: {vs_loaded} ({vs_stats.get('total_conditions', 0)} conditions)")
            
            # Embedding Service
            emb_status = status.get('embedding_service', {})
            emb_working = "✅" if emb_status.get('working') else "❌"
            message_parts.append(f"  Embedding Service: {emb_working}")
            
            # Data Processor
            dp_status = status.get('data_processor', {})
            dp_working = "✅" if dp_status.get('working') else "❌"
            message_parts.append(f"  Data Processor: {dp_working}")
            
            return "\n".join(message_parts)
            
        except Exception as e:
            self.logger.error(f"Failed to format status message: {e}")
            return "System status unavailable"
    
    def _create_search_query(self, parsed_query: Dict[str, Any]) -> str:
        """Create search query from parsed parameters."""
        try:
            query_parts = []
            
            if parsed_query['direction']:
                query_parts.append(parsed_query['direction'])
            
            if parsed_query['horizon']:
                horizon_text = {
                    HorizonType.DAILY: "tomorrow",
                    HorizonType.WEEKLY: "this week",
                    HorizonType.MONTHLY: "this month"
                }[parsed_query['horizon']]
                query_parts.append(horizon_text)
            
            return " ".join(query_parts) if query_parts else "high probability"
            
        except Exception as e:
            self.logger.error(f"Failed to create search query: {e}")
            return "high probability"
    
    def _call_groq_api(self, query: str) -> str:
        """Call GROQ API for general questions."""
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "llama3-8b-8192",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant for FNO (Futures & Options) trading analysis. Provide concise and accurate information about trading, market analysis, and probability predictions."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"Sorry, I couldn't process your query. API error: {response.status_code}"
                
        except Exception as e:
            self.logger.error(f"Failed to call GROQ API: {e}")
            return f"Sorry, I couldn't process your query: {str(e)}"
    
    def _result_to_dict(self, result: ProbabilityResult) -> Dict[str, Any]:
        """Convert ProbabilityResult to dictionary."""
        try:
            return {
                'symbol': result.symbol,
                'horizon': result.horizon.value,
                'up_probability': result.up_probability,
                'down_probability': result.down_probability,
                'neutral_probability': result.neutral_probability,
                'confidence_score': result.confidence_score,
                'timestamp': result.timestamp.isoformat() if result.timestamp else None
            }
        except Exception as e:
            self.logger.error(f"Failed to convert result to dict: {e}")
            return {}

