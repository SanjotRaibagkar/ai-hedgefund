"""
Strategy Manager for AI Hedge Fund

This module provides a centralized manager for all trading strategies,
including intraday, options, and traditional strategies.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_strategy import StrategyManager as BaseStrategyManager
from .intraday_strategies import (
    MomentumBreakoutStrategy,
    MarketDepthStrategy,
    VWAPStrategy,
    GapTradingStrategy,
    IntradayMeanReversionStrategy
)
from .options_strategies import (
    IVSkewStrategy,
    GammaExposureStrategy,
    OptionsFlowStrategy,
    IronCondorStrategy,
    StraddleStrategy
)

logger = logging.getLogger(__name__)


class AdvancedStrategyManager(BaseStrategyManager):
    """Advanced strategy manager with specialized strategy categories."""
    
    def __init__(self):
        """Initialize the advanced strategy manager."""
        super().__init__()
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """Initialize all available strategies."""
        # Intraday Strategies
        self.add_strategy(MomentumBreakoutStrategy())
        self.add_strategy(MarketDepthStrategy())
        self.add_strategy(VWAPStrategy())
        self.add_strategy(GapTradingStrategy())
        self.add_strategy(IntradayMeanReversionStrategy())
        
        # Options Strategies
        self.add_strategy(IVSkewStrategy())
        self.add_strategy(GammaExposureStrategy())
        self.add_strategy(OptionsFlowStrategy())
        self.add_strategy(IronCondorStrategy())
        self.add_strategy(StraddleStrategy())
        
        logger.info(f"Initialized {len(self.strategies)} strategies")
    
    def get_intraday_strategies(self) -> Dict[str, Any]:
        """Get all intraday strategies."""
        intraday_names = [
            "Momentum Breakout",
            "Market Depth Analysis", 
            "VWAP Strategy",
            "Gap Trading",
            "Intraday Mean Reversion"
        ]
        return {name: self.strategies[name] for name in intraday_names if name in self.strategies}
    
    def get_options_strategies(self) -> Dict[str, Any]:
        """Get all options strategies."""
        options_names = [
            "IV Skew Strategy",
            "Gamma Exposure Strategy",
            "Options Flow Strategy",
            "Iron Condor Strategy",
            "Long Straddle Strategy"
        ]
        return {name: self.strategies[name] for name in options_names if name in self.strategies}
    
    def execute_intraday_strategies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all intraday strategies."""
        intraday_strategies = self.get_intraday_strategies()
        results = {}
        
        for name, strategy in intraday_strategies.items():
            if strategy.is_active:
                results[name] = strategy.execute(data)
        
        return results
    
    def execute_options_strategies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all options strategies."""
        options_strategies = self.get_options_strategies()
        results = {}
        
        for name, strategy in options_strategies.items():
            if strategy.is_active:
                results[name] = strategy.execute(data)
        
        return results
    
    def execute_strategies_by_category(self, data: Dict[str, Any], category: str = "all") -> Dict[str, Any]:
        """Execute strategies by category."""
        if category == "intraday":
            return self.execute_intraday_strategies(data)
        elif category == "options":
            return self.execute_options_strategies(data)
        elif category == "all":
            return self.execute_all_strategies(data)
        else:
            logger.warning(f"Unknown strategy category: {category}")
            return {}
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get summary of all strategies."""
        summary = {
            'total_strategies': len(self.strategies),
            'active_strategies': len([s for s in self.strategies.values() if s.is_active]),
            'categories': {
                'intraday': len(self.get_intraday_strategies()),
                'options': len(self.get_options_strategies())
            },
            'strategies': self.get_all_strategies_info()
        }
        return summary


# Global strategy manager instance
_strategy_manager = None


def get_strategy_manager() -> AdvancedStrategyManager:
    """Get the global strategy manager instance."""
    global _strategy_manager
    if _strategy_manager is None:
        _strategy_manager = AdvancedStrategyManager()
    return _strategy_manager


def execute_strategies(data: Dict[str, Any], category: str = "all") -> Dict[str, Any]:
    """Execute strategies with given data."""
    manager = get_strategy_manager()
    return manager.execute_strategies_by_category(data, category)


def get_strategy_summary() -> Dict[str, Any]:
    """Get strategy summary."""
    manager = get_strategy_manager()
    return manager.get_strategy_summary()


def activate_strategy(strategy_name: str):
    """Activate a specific strategy."""
    manager = get_strategy_manager()
    manager.activate_strategy(strategy_name)


def deactivate_strategy(strategy_name: str):
    """Deactivate a specific strategy."""
    manager = get_strategy_manager()
    manager.deactivate_strategy(strategy_name) 