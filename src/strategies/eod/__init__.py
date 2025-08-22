"""
EOD Momentum Strategies for AI Hedge Fund.
End-of-Day momentum-based strategies for swing trading.
"""

from .momentum_framework import MomentumStrategyFramework
from .long_momentum import LongMomentumStrategy
from .short_momentum import ShortMomentumStrategy
from .momentum_indicators import MomentumIndicators
from .position_sizing import PositionSizing
from .risk_management import RiskManager
from .strategy_manager import EODStrategyManager

__all__ = [
    'MomentumStrategyFramework',
    'LongMomentumStrategy',
    'ShortMomentumStrategy',
    'MomentumIndicators',
    'PositionSizing',
    'RiskManager',
    'EODStrategyManager'
] 