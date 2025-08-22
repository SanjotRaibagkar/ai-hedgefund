"""
Base Strategy Class for AI Hedge Fund

This module provides the foundation for all trading strategies.
Strategies can be implemented for different timeframes (daily, intraday)
and data types (stocks, options, derivatives).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, name: str, description: str = ""):
        """Initialize base strategy."""
        self.name = name
        self.description = description
        self.is_active = True
        self.created_at = datetime.now()
        self.last_executed = None
        self.execution_count = 0
        
    @abstractmethod
    def generate_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on input data.
        
        Args:
            data: Dictionary containing market data, technical indicators, etc.
            
        Returns:
            Dictionary containing signals, confidence levels, and reasoning
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate that required data is available for strategy execution.
        
        Args:
            data: Dictionary containing market data
            
        Returns:
            True if data is valid, False otherwise
        """
        pass
    
    def preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess data before strategy execution.
        Override in subclasses if needed.
        
        Args:
            data: Raw market data
            
        Returns:
            Preprocessed data
        """
        return data
    
    def postprocess_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess signals after generation.
        Override in subclasses if needed.
        
        Args:
            signals: Raw signals from strategy
            
        Returns:
            Processed signals
        """
        return signals
    
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the strategy with given data.
        
        Args:
            data: Market data for strategy execution
            
        Returns:
            Strategy results with signals and metadata
        """
        try:
            # Validate data
            if not self.validate_data(data):
                logger.warning(f"Strategy {self.name}: Invalid data provided")
                return {
                    'strategy_name': self.name,
                    'status': 'failed',
                    'error': 'Invalid data',
                    'signals': {},
                    'metadata': {
                        'execution_time': datetime.now(),
                        'data_validation': False
                    }
                }
            
            # Preprocess data
            processed_data = self.preprocess_data(data)
            
            # Generate signals
            raw_signals = self.generate_signals(processed_data)
            
            # Postprocess signals
            final_signals = self.postprocess_signals(raw_signals)
            
            # Update execution metadata
            self.last_executed = datetime.now()
            self.execution_count += 1
            
            return {
                'strategy_name': self.name,
                'status': 'success',
                'signals': final_signals,
                'metadata': {
                    'execution_time': self.last_executed,
                    'execution_count': self.execution_count,
                    'data_validation': True
                }
            }
            
        except Exception as e:
            logger.error(f"Strategy {self.name} execution failed: {e}")
            return {
                'strategy_name': self.name,
                'status': 'error',
                'error': str(e),
                'signals': {},
                'metadata': {
                    'execution_time': datetime.now(),
                    'data_validation': False
                }
            }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and metadata."""
        return {
            'name': self.name,
            'description': self.description,
            'is_active': self.is_active,
            'created_at': self.created_at,
            'last_executed': self.last_executed,
            'execution_count': self.execution_count,
            'strategy_type': self.__class__.__name__
        }
    
    def activate(self):
        """Activate the strategy."""
        self.is_active = True
        logger.info(f"Strategy {self.name} activated")
    
    def deactivate(self):
        """Deactivate the strategy."""
        self.is_active = False
        logger.info(f"Strategy {self.name} deactivated")


class StrategyManager:
    """Manager for multiple strategies."""
    
    def __init__(self):
        """Initialize strategy manager."""
        self.strategies: Dict[str, BaseStrategy] = {}
    
    def add_strategy(self, strategy: BaseStrategy):
        """Add a strategy to the manager."""
        self.strategies[strategy.name] = strategy
        logger.info(f"Strategy {strategy.name} added to manager")
    
    def remove_strategy(self, strategy_name: str):
        """Remove a strategy from the manager."""
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            logger.info(f"Strategy {strategy_name} removed from manager")
    
    def get_strategy(self, strategy_name: str) -> Optional[BaseStrategy]:
        """Get a strategy by name."""
        return self.strategies.get(strategy_name)
    
    def execute_all_strategies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all active strategies with given data."""
        results = {}
        
        for name, strategy in self.strategies.items():
            if strategy.is_active:
                results[name] = strategy.execute(data)
            else:
                results[name] = {
                    'strategy_name': name,
                    'status': 'skipped',
                    'reason': 'Strategy is inactive'
                }
        
        return results
    
    def get_all_strategies_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information for all strategies."""
        return {
            name: strategy.get_strategy_info() 
            for name, strategy in self.strategies.items()
        }
    
    def activate_strategy(self, strategy_name: str):
        """Activate a specific strategy."""
        strategy = self.get_strategy(strategy_name)
        if strategy:
            strategy.activate()
    
    def deactivate_strategy(self, strategy_name: str):
        """Deactivate a specific strategy."""
        strategy = self.get_strategy(strategy_name)
        if strategy:
            strategy.deactivate() 