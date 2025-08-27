#!/usr/bin/env python3
"""
Trading Decision Guide
How to use the FNO ML system for trading decisions.
"""

import sys
import os
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fno_rag import FNOEngine, HorizonType
from loguru import logger

def trading_decision_guide():
    """Demonstrate how to use the ML system for trading decisions."""
    
    print("📈 FNO Trading Decision Guide")
    print("=" * 50)
    
    try:
        # Initialize FNO engine
        print("1. Initializing FNO ML System...")
        fno_engine = FNOEngine()
        print("   ✅ System ready for trading decisions")
        
        # Get top symbols for analysis
        print("\n2. Getting top FNO symbols...")
        symbols = ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK']
        
        print("\n3. 📊 Trading Decision Analysis")
        print("-" * 50)
        
        for symbol in symbols:
            print(f"\n🔍 Analyzing {symbol}:")
            
            # Get predictions for all horizons
            daily_result = fno_engine.predict_probability(symbol, HorizonType.DAILY)
            weekly_result = fno_engine.predict_probability(symbol, HorizonType.WEEKLY)
            monthly_result = fno_engine.predict_probability(symbol, HorizonType.MONTHLY)
            
            if daily_result and weekly_result and monthly_result:
                # Daily analysis
                print(f"   📅 Daily (Tomorrow):")
                print(f"      Up: {daily_result.up_probability:.1%} | Down: {daily_result.down_probability:.1%} | Confidence: {daily_result.confidence_score:.1%}")
                
                # Weekly analysis
                print(f"   📅 Weekly:")
                print(f"      Up: {weekly_result.up_probability:.1%} | Down: {weekly_result.down_probability:.1%} | Confidence: {weekly_result.confidence_score:.1%}")
                
                # Monthly analysis
                print(f"   📅 Monthly:")
                print(f"      Up: {monthly_result.up_probability:.1%} | Down: {monthly_result.down_probability:.1%} | Confidence: {monthly_result.confidence_score:.1%}")
                
                # Trading recommendation
                recommendation = get_trading_recommendation(daily_result, weekly_result, monthly_result)
                print(f"   💡 Recommendation: {recommendation}")
        
        print("\n4. 🎯 Trading Strategy Guidelines")
        print("-" * 50)
        print_trading_guidelines()
        
        print("\n5. 📋 Risk Management Rules")
        print("-" * 50)
        print_risk_management_rules()
        
        return True
        
    except Exception as e:
        print(f"❌ Error in trading guide: {e}")
        logger.error(f"Trading guide failed: {e}")
        return False

def get_trading_recommendation(daily, weekly, monthly):
    """Get trading recommendation based on predictions."""
    
    # Calculate weighted probabilities
    daily_weight = 0.5
    weekly_weight = 0.3
    monthly_weight = 0.2
    
    weighted_up = (
        daily.up_probability * daily_weight +
        weekly.up_probability * weekly_weight +
        monthly.up_probability * monthly_weight
    )
    
    weighted_down = (
        daily.down_probability * daily_weight +
        weekly.down_probability * weekly_weight +
        monthly.down_probability * monthly_weight
    )
    
    # Calculate confidence
    avg_confidence = (daily.confidence_score + weekly.confidence_score + monthly.confidence_score) / 3
    
    # Generate recommendation
    if avg_confidence < 0.6:
        return "HOLD - Low confidence, wait for clearer signals"
    
    if weighted_up > 0.6:
        return f"BUY - Strong bullish signals ({weighted_up:.1%} probability)"
    elif weighted_down > 0.6:
        return f"SELL - Strong bearish signals ({weighted_down:.1%} probability)"
    else:
        return "NEUTRAL - Mixed signals, consider hedging"

def print_trading_guidelines():
    """Print trading strategy guidelines."""
    
    guidelines = [
        "🎯 HIGH CONFIDENCE (>80%): Strong signals, consider larger positions",
        "🎯 MEDIUM CONFIDENCE (60-80%): Moderate signals, standard position sizes",
        "🎯 LOW CONFIDENCE (<60%): Weak signals, reduce position sizes or avoid",
        "",
        "📈 BULLISH SIGNALS:",
        "   • Up probability > 60% across multiple timeframes",
        "   • High confidence scores (>70%)",
        "   • Consider buying calls or going long",
        "",
        "📉 BEARISH SIGNALS:",
        "   • Down probability > 60% across multiple timeframes", 
        "   • High confidence scores (>70%)",
        "   • Consider buying puts or going short",
        "",
        "⚖️ NEUTRAL SIGNALS:",
        "   • Mixed probabilities across timeframes",
        "   • Consider straddle/strangle strategies",
        "   • Or wait for clearer directional signals"
    ]
    
    for guideline in guidelines:
        print(f"   {guideline}")

def print_risk_management_rules():
    """Print risk management rules."""
    
    rules = [
        "🛡️ POSITION SIZING:",
        "   • Never risk more than 2% of capital per trade",
        "   • Adjust position size based on confidence level",
        "   • Higher confidence = larger position allowed",
        "",
        "🛡️ STOP LOSS:",
        "   • Always set stop losses for every position",
        "   • Daily trades: 1-2% stop loss",
        "   • Weekly trades: 3-5% stop loss", 
        "   • Monthly trades: 5-10% stop loss",
        "",
        "🛡️ PROFIT TAKING:",
        "   • Take partial profits at 50% of target",
        "   • Trail stops to protect profits",
        "   • Don't be greedy - secure gains",
        "",
        "🛡️ DIVERSIFICATION:",
        "   • Don't put all eggs in one basket",
        "   • Trade multiple symbols",
        "   • Mix different timeframes",
        "",
        "🛡️ EMOTIONAL CONTROL:",
        "   • Stick to your trading plan",
        "   • Don't chase losses",
        "   • Take breaks after losses",
        "   • Keep a trading journal"
    ]
    
    for rule in rules:
        print(f"   {rule}")

def analyze_portfolio_risk():
    """Analyze portfolio risk across multiple symbols."""
    
    print("\n6. 📊 Portfolio Risk Analysis")
    print("-" * 50)
    
    # Sample portfolio
    portfolio = {
        'NIFTY': {'weight': 0.3, 'type': 'index'},
        'BANKNIFTY': {'weight': 0.2, 'type': 'index'},
        'RELIANCE': {'weight': 0.15, 'type': 'stock'},
        'TCS': {'weight': 0.15, 'type': 'stock'},
        'INFY': {'weight': 0.1, 'type': 'stock'},
        'HDFC': {'weight': 0.1, 'type': 'stock'}
    }
    
    print("Portfolio Composition:")
    for symbol, details in portfolio.items():
        print(f"   {symbol}: {details['weight']:.1%} ({details['type']})")
    
    print("\nRisk Assessment:")
    print("   • Index-heavy portfolio (50% in NIFTY + BANKNIFTY)")
    print("   • Good diversification across sectors")
    print("   • Consider adding more defensive stocks")
    print("   • Monitor correlation between positions")

if __name__ == "__main__":
    trading_decision_guide()
    analyze_portfolio_risk()
