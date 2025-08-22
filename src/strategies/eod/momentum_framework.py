"""
Momentum Strategy Framework for EOD Trading.
Coordinates long and short momentum strategies with portfolio-level management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from loguru import logger

from .long_momentum import LongMomentumStrategy
from .short_momentum import ShortMomentumStrategy
from .momentum_indicators import MomentumIndicators
from .position_sizing import PositionSizing
from .risk_management import RiskManager


class MomentumStrategyFramework:
    """Framework for managing momentum-based EOD strategies."""
    
    def __init__(self,
                 framework_name: str = "EOD Momentum Framework",
                 max_positions: int = 10,
                 max_long_positions: int = 5,
                 max_short_positions: int = 5,
                 portfolio_risk_limit: float = 0.02,
                 correlation_threshold: float = 0.7,
                 **kwargs):
        """Initialize momentum strategy framework."""
        self.framework_name = framework_name
        self.max_positions = max_positions
        self.max_long_positions = max_long_positions
        self.max_short_positions = max_short_positions
        self.portfolio_risk_limit = portfolio_risk_limit
        self.correlation_threshold = correlation_threshold
        
        # Initialize strategies
        self.long_strategy = LongMomentumStrategy(**kwargs.get('long_strategy', {}))
        self.short_strategy = ShortMomentumStrategy(**kwargs.get('short_strategy', {}))
        
        # Initialize components
        self.momentum_indicators = MomentumIndicators()
        self.position_sizing = PositionSizing(**kwargs.get('position_sizing', {}))
        self.risk_manager = RiskManager(**kwargs.get('risk_management', {}))
        
        # Portfolio state
        self.positions = {}
        self.portfolio_value = 100000
        self.cash_balance = self.portfolio_value
        
        # Framework parameters
        self.parameters = {
            'min_signal_confidence': 0.6,
            'max_sector_exposure': 0.3,
            'rebalance_frequency': 'daily',
            'stop_loss_enabled': True,
            'take_profit_enabled': True,
            'trailing_stop_enabled': False,
            'position_sizing_method': 'adaptive',
            'risk_parity_enabled': True,
            **kwargs
        }
    
    def analyze_universe(self, universe_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze entire universe for momentum opportunities."""
        try:
            logger.info(f"Analyzing universe of {len(universe_data)} tickers")
            
            results = {
                'long_signals': [],
                'short_signals': [],
                'hold_signals': [],
                'error_signals': [],
                'timestamp': datetime.now()
            }
            
            # Analyze each ticker
            for ticker, df in universe_data.items():
                try:
                    long_analysis = self.long_strategy.analyze_stock(df, ticker)
                    short_analysis = self.short_strategy.analyze_stock(df, ticker)
                    
                    best_signal = self._determine_best_signal(long_analysis, short_analysis)
                    
                    if best_signal['signal'] == 'buy':
                        results['long_signals'].append(best_signal)
                    elif best_signal['signal'] == 'sell':
                        results['short_signals'].append(best_signal)
                    elif best_signal['signal'] == 'hold':
                        results['hold_signals'].append(best_signal)
                    else:
                        results['error_signals'].append(best_signal)
                        
                except Exception as e:
                    logger.error(f"Failed to analyze {ticker}: {e}")
                    results['error_signals'].append({
                        'ticker': ticker,
                        'signal': 'error',
                        'confidence': 0,
                        'reason': f'Analysis failed: {e}'
                    })
            
            # Sort signals by confidence
            results['long_signals'].sort(key=lambda x: x.get('confidence', 0), reverse=True)
            results['short_signals'].sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            logger.info(f"Analysis complete: {len(results['long_signals'])} long, {len(results['short_signals'])} short signals")
            return results
            
        except Exception as e:
            logger.error(f"Universe analysis failed: {e}")
            return {'error': f'Universe analysis failed: {e}', 'timestamp': datetime.now()}
    
    def generate_trade_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trade recommendations based on analysis results."""
        try:
            recommendations = {
                'new_positions': [],
                'exit_positions': [],
                'adjust_positions': [],
                'portfolio_actions': [],
                'timestamp': datetime.now()
            }
            
            # Check existing positions for exits
            recommendations['exit_positions'] = self._check_position_exits()
            
            # Generate new position recommendations
            recommendations['new_positions'] = self._generate_new_position_recommendations(analysis_results)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Trade recommendations generation failed: {e}")
            return {'error': f'Trade recommendations failed: {e}', 'timestamp': datetime.now()}
    
    def _determine_best_signal(self, long_analysis: Dict[str, Any], short_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the best signal between long and short analysis."""
        try:
            long_confidence = long_analysis.get('confidence', 0)
            short_confidence = short_analysis.get('confidence', 0)
            
            if long_confidence < self.parameters['min_signal_confidence'] and \
               short_confidence < self.parameters['min_signal_confidence']:
                return {
                    'ticker': long_analysis.get('ticker', short_analysis.get('ticker')),
                    'signal': 'hold',
                    'confidence': max(long_confidence, short_confidence),
                    'reason': 'Both signals below confidence threshold'
                }
            
            if long_confidence > short_confidence:
                return long_analysis
            else:
                return short_analysis
                
        except Exception as e:
            logger.error(f"Signal determination failed: {e}")
            return {
                'ticker': long_analysis.get('ticker', 'UNKNOWN'),
                'signal': 'error',
                'confidence': 0,
                'reason': f'Signal determination failed: {e}'
            }
    
    def _check_position_exits(self) -> List[Dict[str, Any]]:
        """Check existing positions for exit signals."""
        try:
            exit_recommendations = []
            
            for ticker, position in self.positions.items():
                current_data = pd.DataFrame({
                    'close_price': [position.get('current_price', position.get('entry_price', 100))],
                    'volume': [1000000]
                })
                
                if position.get('type') == 'long':
                    exit_decision = self.long_strategy.should_exit_position(position, current_data)
                else:
                    exit_decision = self.short_strategy.should_exit_position(position, current_data)
                
                if exit_decision.get('should_exit', False):
                    exit_recommendations.append({
                        'ticker': ticker,
                        'action': 'exit',
                        'reason': exit_decision.get('reason', 'unknown'),
                        'exit_price': exit_decision.get('exit_price'),
                        'pnl': exit_decision.get('pnl', 0),
                        'position': position
                    })
            
            return exit_recommendations
            
        except Exception as e:
            logger.error(f"Position exit check failed: {e}")
            return []
    
    def _generate_new_position_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate new position recommendations."""
        try:
            recommendations = []
            
            current_long_count = len([p for p in self.positions.values() if p.get('type') == 'long'])
            current_short_count = len([p for p in self.positions.values() if p.get('type') == 'short'])
            
            # Process long signals
            for signal in analysis_results.get('long_signals', []):
                if current_long_count >= self.max_long_positions:
                    break
                
                if signal.get('confidence', 0) >= self.parameters['min_signal_confidence']:
                    recommendation = self._create_position_recommendation(signal, 'long')
                    if recommendation:
                        recommendations.append(recommendation)
                        current_long_count += 1
            
            # Process short signals
            for signal in analysis_results.get('short_signals', []):
                if current_short_count >= self.max_short_positions:
                    break
                
                if signal.get('confidence', 0) >= self.parameters['min_signal_confidence']:
                    recommendation = self._create_position_recommendation(signal, 'short')
                    if recommendation:
                        recommendations.append(recommendation)
                        current_short_count += 1
            
            return recommendations
            
        except Exception as e:
            logger.error(f"New position recommendations failed: {e}")
            return []
    
    def _create_position_recommendation(self, signal: Dict[str, Any], position_type: str) -> Optional[Dict[str, Any]]:
        """Create a position recommendation from a signal."""
        try:
            ticker = signal.get('ticker')
            if not ticker or ticker in self.positions:
                return None
            
            recommendation = {
                'ticker': ticker,
                'action': 'enter',
                'position_type': position_type,
                'entry_price': signal.get('entry_price'),
                'stop_loss': signal.get('stop_loss'),
                'take_profit': signal.get('take_profit'),
                'position_size': signal.get('position_size'),
                'confidence': signal.get('confidence', 0),
                'reason': signal.get('reason', ''),
                'risk_reward_ratio': signal.get('risk_reward_ratio'),
                'signal_data': signal
            }
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Position recommendation creation failed: {e}")
            return None
    
    def get_framework_summary(self) -> Dict[str, Any]:
        """Get framework summary and parameters."""
        return {
            'framework_name': self.framework_name,
            'parameters': self.parameters,
            'position_limits': {
                'max_positions': self.max_positions,
                'max_long_positions': self.max_long_positions,
                'max_short_positions': self.max_short_positions
            },
            'risk_limits': {
                'portfolio_risk_limit': self.portfolio_risk_limit,
                'correlation_threshold': self.correlation_threshold
            },
            'long_strategy_summary': self.long_strategy.get_strategy_summary(),
            'short_strategy_summary': self.short_strategy.get_strategy_summary(),
            'portfolio_state': {
                'total_positions': len(self.positions),
                'portfolio_value': self.portfolio_value,
                'cash_balance': self.cash_balance
            }
        } 