#!/usr/bin/env python3
"""
Screening Manager
Coordinates all screening modules and provides unified interface
for stock screening, options analysis, and market predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
import json

from src.screening.unified_eod_screener import unified_eod_screener
from src.screening.intraday_screener import IntradayStockScreener
from src.screening.options_analyzer import OptionsAnalyzer
from src.screening.market_predictor import MarketPredictor


class ScreeningManager:
    """Comprehensive screening manager for all market analysis."""
    
    def __init__(self):
        """Initialize Screening Manager."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize all screeners
        self.eod_screener = unified_eod_screener
        self.intraday_screener = IntradayStockScreener()
        self.options_analyzer = OptionsAnalyzer()
        self.market_predictor = MarketPredictor()
        
        # Default Indian stock list
        self.default_indian_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
            'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'HCLTECH.NS', 'SUNPHARMA.NS',
            'WIPRO.NS', 'ULTRACEMCO.NS', 'TITAN.NS', 'BAJFINANCE.NS', 'NESTLEIND.NS'
        ]
        
    def run_comprehensive_screening(self, 
                                  stock_list: Optional[List[str]] = None,
                                  include_options: bool = True,
                                  include_predictions: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive screening for all modules.
        
        Args:
            stock_list: List of stocks to screen (None = all stocks from database)
            include_options: Whether to include options analysis
            include_predictions: Whether to include market predictions
            
        Returns:
            Comprehensive screening results
        """
        self.logger.info("Starting comprehensive screening")
        
        if stock_list is None:
            # Get all stocks from database for comprehensive screening
            try:
                from src.data.database.duckdb_manager import DatabaseManager
                db_manager = DatabaseManager()
                all_symbols = db_manager.get_available_symbols()
                # Convert to .NS format for consistency
                stock_list = [f"{symbol}.NS" for symbol in all_symbols]
                self.logger.info(f"Using all {len(stock_list)} stocks from database for comprehensive screening")
            except Exception as e:
                self.logger.error(f"Error getting all stocks from database: {e}")
                self.logger.info("Falling back to default stock list")
                stock_list = self.default_indian_stocks
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'stock_screening': {},
            'options_analysis': {},
            'market_predictions': {},
            'summary': {
                'total_stocks': len(stock_list),
                'eod_signals': 0,
                'intraday_signals': 0,
                'options_analysis_count': 0,
                'predictions_count': 0
            }
        }
        
        try:
            # 1. EOD Stock Screening
            self.logger.info("Running EOD stock screening...")
            # Convert .NS symbols to regular symbols for unified screener
            symbols = [s.replace('.NS', '') for s in stock_list]
            eod_results = self.get_eod_signals(stock_list)
            results['stock_screening']['eod'] = eod_results
            results['summary']['eod_signals'] = (
                eod_results['summary']['bullish_count'] + 
                eod_results['summary']['bearish_count']
            )
            
            # 2. Intraday Stock Screening
            self.logger.info("Running intraday stock screening...")
            intraday_results = self.intraday_screener.screen_stocks(stock_list)
            results['stock_screening']['intraday'] = intraday_results
            results['summary']['intraday_signals'] = (
                intraday_results['summary']['breakout_count'] + 
                intraday_results['summary']['reversal_count'] + 
                intraday_results['summary']['momentum_count']
            )
            
            # 3. Options Analysis
            if include_options:
                self.logger.info("Running options analysis...")
                options_results = {}
                for index in ['NIFTY', 'BANKNIFTY']:
                    try:
                        index_analysis = self.options_analyzer.analyze_index_options(index)
                        options_results[index] = index_analysis
                        results['summary']['options_analysis_count'] += 1
                    except Exception as e:
                        self.logger.error(f"Error analyzing {index} options: {e}")
                        options_results[index] = {}
                
                results['options_analysis'] = options_results
            
            # 4. Market Predictions
            if include_predictions:
                self.logger.info("Running market predictions...")
                prediction_results = {}
                timeframes = ['15min', '1hour', 'eod', 'multiday']
                
                for index in ['NIFTY', 'BANKNIFTY']:
                    prediction_results[index] = {}
                    for timeframe in timeframes:
                        try:
                            prediction = self.market_predictor.predict_market_movement(index, timeframe)
                            prediction_results[index][timeframe] = prediction
                            results['summary']['predictions_count'] += 1
                        except Exception as e:
                            self.logger.error(f"Error predicting {index} {timeframe}: {e}")
                            prediction_results[index][timeframe] = {}
                
                results['market_predictions'] = prediction_results
            
            self.logger.info("Comprehensive screening completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive screening: {e}")
            results['error'] = str(e)
        
        return results
    
    def get_eod_signals(self, stock_list: Optional[List[str]] = None, 
                       risk_reward_ratio: float = 2.0, analysis_mode: str = "comprehensive") -> Dict[str, Any]:
        """
        Get EOD trading signals using unified screener.
        
        Args:
            stock_list: List of stocks to screen
            risk_reward_ratio: Minimum risk-reward ratio
            
        Returns:
            EOD signals with entry, stop loss, and targets
        """
        if stock_list is None:
            stock_list = self.default_indian_stocks
        
        # Convert .NS symbols to regular symbols for unified screener
        symbols = [s.replace('.NS', '') for s in stock_list]
        
        # Run unified screener with specified analysis mode
        import asyncio
        try:
            # Use adjusted criteria for better signal generation
            results = asyncio.run(self.eod_screener.screen_universe(
                symbols=symbols,
                min_volume=50000,  # Lower volume threshold
                min_price=5.0,     # Lower price threshold
                analysis_mode=analysis_mode
            ))
            
            # Convert results to expected format
            return {
                'summary': {
                    'total_stocks': results['summary']['total_screened'],
                    'bullish_count': results['summary']['bullish_signals'],
                    'bearish_count': results['summary']['bearish_signals']
                },
                'bullish_signals': results['bullish_signals'],
                'bearish_signals': results['bearish_signals']
            }
        except Exception as e:
            self.logger.error(f"Error in EOD screening: {e}")
            return {
                'summary': {
                    'total_stocks': len(symbols),
                    'bullish_count': 0,
                    'bearish_count': 0
                },
                'bullish_signals': [],
                'bearish_signals': []
            }
    
    def get_intraday_signals(self, stock_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get intraday trading signals.
        
        Args:
            stock_list: List of stocks to screen
            
        Returns:
            Intraday signals with breakout, reversal, and momentum opportunities
        """
        if stock_list is None:
            stock_list = self.default_indian_stocks
        
        return self.intraday_screener.screen_stocks(stock_list)
    
    def get_options_analysis(self, index: str = 'NIFTY') -> Dict[str, Any]:
        """
        Get options analysis for specified index.
        
        Args:
            index: 'NIFTY' or 'BANKNIFTY'
            
        Returns:
            Options analysis with OI, volatility, and strike recommendations
        """
        return self.options_analyzer.analyze_index_options(index)
    
    def get_market_prediction(self, index: str = 'NIFTY', timeframe: str = '15min') -> Dict[str, Any]:
        """
        Get market movement prediction.
        
        Args:
            index: 'NIFTY' or 'BANKNIFTY'
            timeframe: '15min', '1hour', 'eod', 'multiday'
            
        Returns:
            Market movement prediction with direction and confidence
        """
        return self.market_predictor.predict_market_movement(index, timeframe)
    
    def get_top_signals(self, results: Dict[str, Any], signal_type: str = 'eod', 
                       limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get top signals from screening results.
        
        Args:
            results: Screening results
            signal_type: 'eod', 'intraday', 'options', 'predictions'
            limit: Number of top signals to return
            
        Returns:
            List of top signals
        """
        try:
            if signal_type == 'eod':
                bullish_signals = results['stock_screening']['eod']['bullish_signals'][:limit]
                bearish_signals = results['stock_screening']['eod']['bearish_signals'][:limit]
                return {
                    'bullish': bullish_signals,
                    'bearish': bearish_signals
                }
            
            elif signal_type == 'intraday':
                breakout_signals = results['stock_screening']['intraday']['breakout_signals'][:limit]
                reversal_signals = results['stock_screening']['intraday']['reversal_signals'][:limit]
                momentum_signals = results['stock_screening']['intraday']['momentum_signals'][:limit]
                return {
                    'breakout': breakout_signals,
                    'reversal': reversal_signals,
                    'momentum': momentum_signals
                }
            
            elif signal_type == 'options':
                return results['options_analysis']
            
            elif signal_type == 'predictions':
                return results['market_predictions']
            
            else:
                self.logger.error(f"Invalid signal type: {signal_type}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting top signals: {e}")
            return []
    
    def generate_trading_recommendations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading recommendations from screening results.
        
        Args:
            results: Comprehensive screening results
            
        Returns:
            Trading recommendations with actionable insights
        """
        try:
            recommendations = {
                'timestamp': datetime.now().isoformat(),
                'high_confidence_signals': [],
                'risk_management': {},
                'market_overview': {},
                'action_items': []
            }
            
            # High confidence EOD signals
            eod_results = results.get('stock_screening', {}).get('eod', {})
            high_confidence_eod = []
            
            for signal in eod_results.get('bullish_signals', []):
                if signal.get('confidence', 0) > 80:
                    high_confidence_eod.append({
                        'ticker': signal['ticker'],
                        'signal': 'BULLISH',
                        'confidence': signal['confidence'],
                        'entry': signal['entry_price'],
                        'stop_loss': signal['stop_loss'],
                        'targets': signal['targets'],
                        'risk_reward': signal['risk_reward_ratio']
                    })
            
            for signal in eod_results.get('bearish_signals', []):
                if signal.get('confidence', 0) > 80:
                    high_confidence_eod.append({
                        'ticker': signal['ticker'],
                        'signal': 'BEARISH',
                        'confidence': signal['confidence'],
                        'entry': signal['entry_price'],
                        'stop_loss': signal['stop_loss'],
                        'targets': signal['targets'],
                        'risk_reward': signal['risk_reward_ratio']
                    })
            
            recommendations['high_confidence_signals'] = high_confidence_eod[:5]
            
            # Market overview
            market_predictions = results.get('market_predictions', {})
            nifty_prediction = market_predictions.get('NIFTY', {}).get('eod', {})
            banknifty_prediction = market_predictions.get('BANKNIFTY', {}).get('eod', {})
            
            recommendations['market_overview'] = {
                'nifty_direction': nifty_prediction.get('prediction', {}).get('direction', 'NEUTRAL'),
                'nifty_confidence': nifty_prediction.get('prediction', {}).get('confidence', 50),
                'banknifty_direction': banknifty_prediction.get('prediction', {}).get('direction', 'NEUTRAL'),
                'banknifty_confidence': banknifty_prediction.get('prediction', {}).get('confidence', 50)
            }
            
            # Risk management
            total_signals = len(high_confidence_eod)
            recommendations['risk_management'] = {
                'max_positions': min(total_signals, 5),
                'position_size': 0.1,  # 10% per position
                'max_portfolio_risk': 0.02,  # 2% max portfolio risk
                'stop_loss_adherence': 'STRICT'
            }
            
            # Action items
            action_items = []
            
            if high_confidence_eod:
                action_items.append("Review high confidence EOD signals for entry opportunities")
            
            if nifty_prediction.get('prediction', {}).get('direction') != 'NEUTRAL':
                action_items.append(f"Monitor Nifty for {nifty_prediction['prediction']['direction']} movement")
            
            if banknifty_prediction.get('prediction', {}).get('direction') != 'NEUTRAL':
                action_items.append(f"Monitor BankNifty for {banknifty_prediction['prediction']['direction']} movement")
            
            recommendations['action_items'] = action_items
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating trading recommendations: {e}")
            return {}
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """
        Save screening results to file.
        
        Args:
            results: Screening results
            filename: Output filename (optional)
            
        Returns:
            Path to saved file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"screening_results_{timestamp}.json"
            
            filepath = f"results/{filename}"
            
            # Create results directory if it doesn't exist
            import os
            os.makedirs('results', exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Results saved to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            return ""
    
    def get_screening_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all screening capabilities."""
        return {
            'name': 'Screening Manager',
            'description': 'Comprehensive screening system for Indian markets',
            'modules': {
                'eod_screener': self.eod_screener.get_screener_summary(),
                'intraday_screener': self.intraday_screener.get_screener_summary(),
                'options_analyzer': self.options_analyzer.get_analyzer_summary(),
                'market_predictor': self.market_predictor.get_predictor_summary()
            },
            'capabilities': [
                'EOD Stock Screening with Entry/SL/Targets',
                'Intraday Stock Screening with Breakout/Reversal Detection',
                'Options Analysis for Nifty and BankNifty',
                'Market Movement Predictions (15min to Multi-day)',
                'Trading Recommendations Generation',
                'Risk Management Guidelines'
            ],
            'supported_markets': [
                'Indian Stocks (NSE)',
                'Nifty 50 Index',
                'BankNifty Index'
            ],
            'default_stocks': self.default_indian_stocks[:10]  # Show first 10
        }
    
    def run_quick_screening(self, stock_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run quick screening for immediate trading opportunities.
        
        Args:
            stock_list: List of stocks to screen
            
        Returns:
            Quick screening results with top opportunities
        """
        if stock_list is None:
            stock_list = self.default_indian_stocks[:10]  # Top 10 stocks
        
        self.logger.info("Running quick screening...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'quick_signals': [],
            'market_sentiment': 'NEUTRAL'
        }
        
        try:
            # Get EOD signals
            eod_results = self.eod_screener.screen_stocks(stock_list)
            
            # Get top signals
            top_bullish = eod_results.get('bullish_signals', [])[:3]
            top_bearish = eod_results.get('bearish_signals', [])[:3]
            
            # Combine signals
            all_signals = []
            for signal in top_bullish + top_bearish:
                all_signals.append({
                    'ticker': signal['ticker'],
                    'signal': signal['signal'],
                    'confidence': signal['confidence'],
                    'entry': signal['entry_price'],
                    'stop_loss': signal['stop_loss'],
                    'target': signal['targets']['T1'],
                    'risk_reward': signal['risk_reward_ratio']
                })
            
            # Sort by confidence
            all_signals.sort(key=lambda x: x['confidence'], reverse=True)
            results['quick_signals'] = all_signals
            
            # Determine market sentiment
            bullish_count = len([s for s in all_signals if s['signal'] == 'BULLISH'])
            bearish_count = len([s for s in all_signals if s['signal'] == 'BEARISH'])
            
            if bullish_count > bearish_count:
                results['market_sentiment'] = 'BULLISH'
            elif bearish_count > bullish_count:
                results['market_sentiment'] = 'BEARISH'
            else:
                results['market_sentiment'] = 'NEUTRAL'
            
            self.logger.info(f"Quick screening completed: {len(all_signals)} signals found")
            
        except Exception as e:
            self.logger.error(f"Error in quick screening: {e}")
            results['error'] = str(e)
        
        return results 