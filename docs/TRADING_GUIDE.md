# 📈 Trading Guide Documentation

## Overview

This comprehensive trading guide explains how to use the FNO RAG system for making informed trading decisions in the Indian Futures & Options (FNO) market.

## Understanding the System

### Core Components

1. **ML Predictions**: Machine learning models provide probability predictions
2. **RAG Context**: Historical pattern matching enhances predictions
3. **Risk Management**: Built-in position sizing and stop-loss recommendations
4. **Multi-Timeframe Analysis**: Daily, weekly, and monthly predictions

### Prediction Types

- **Up Probability**: Chance of significant upward movement
- **Down Probability**: Chance of significant downward movement
- **Neutral Probability**: Chance of sideways movement
- **Confidence Score**: Reliability of the prediction

## Getting Started

### Basic Setup

```python
from src.fno_rag import FNOEngine, HorizonType

# Initialize the system
fno_engine = FNOEngine()

# Get a basic prediction
result = fno_engine.predict_probability("NIFTY", HorizonType.DAILY)
print(f"Up: {result.up_probability:.1%}")
print(f"Down: {result.down_probability:.1%}")
print(f"Confidence: {result.confidence_score:.1%}")
```

### Understanding Predictions

```python
def interpret_prediction(result):
    """Interpret prediction results."""
    
    print(f"📊 Prediction Analysis for {result.symbol}")
    print(f"Timeframe: {result.horizon.value}")
    print(f"Timestamp: {result.timestamp}")
    
    # Probability breakdown
    print(f"\n🟢 Up Probability: {result.up_probability:.1%}")
    print(f"🔴 Down Probability: {result.down_probability:.1%}")
    print(f"⚪ Neutral Probability: {result.neutral_probability:.1%}")
    
    # Confidence assessment
    if result.confidence_score >= 0.8:
        confidence_level = "HIGH"
        emoji = "🟢"
    elif result.confidence_score >= 0.6:
        confidence_level = "MEDIUM"
        emoji = "🟡"
    else:
        confidence_level = "LOW"
        emoji = "🔴"
    
    print(f"\n🎯 Confidence: {emoji} {confidence_level} ({result.confidence_score:.1%})")
    
    # Trading recommendation
    recommendation = get_trading_recommendation(result)
    print(f"\n💡 Recommendation: {recommendation}")
```

## Trading Strategies

### High Confidence Trading (>80% confidence)

**When to Use**: Strong signals with high confidence scores

**Strategy**:
- Consider larger position sizes
- Follow the dominant probability direction
- Use tight stop losses (1-2% for daily trades)
- Take profits at 50% of target

**Example**:
```python
# High confidence bullish signal
if result.confidence_score > 0.8 and result.up_probability > 0.7:
    print("🟢 STRONG BUY SIGNAL")
    print("Position Size: 3-5% of capital")
    print("Stop Loss: 1.5% below entry")
    print("Target: 3-5% above entry")
```

### Medium Confidence Trading (60-80% confidence)

**When to Use**: Moderate signals with reasonable confidence

**Strategy**:
- Use standard position sizes
- Consider hedging strategies for mixed signals
- Use moderate stop losses (2-3%)
- Monitor closely for signal changes

**Example**:
```python
# Medium confidence signal
if 0.6 <= result.confidence_score <= 0.8:
    print("🟡 MODERATE SIGNAL")
    print("Position Size: 1-2% of capital")
    print("Stop Loss: 2.5% below entry")
    print("Consider hedging if mixed signals")
```

### Low Confidence Trading (<60% confidence)

**When to Use**: Weak signals or unclear market conditions

**Strategy**:
- Use smaller position sizes or avoid trading
- Wait for clearer signals
- Use wider stop losses (3-5%)
- Focus on risk management

**Example**:
```python
# Low confidence signal
if result.confidence_score < 0.6:
    print("🔴 WEAK SIGNAL")
    print("Recommendation: HOLD or small position")
    print("Position Size: 0.5% of capital (if trading)")
    print("Stop Loss: 4% below entry")
```

## Risk Management

### Position Sizing

```python
def calculate_position_size(capital: float, confidence: float, risk_per_trade: float = 0.02):
    """Calculate position size based on confidence and risk tolerance."""
    
    # Base position size (2% risk per trade)
    base_size = capital * risk_per_trade
    
    # Adjust for confidence
    if confidence >= 0.8:
        size_multiplier = 1.5  # Increase size for high confidence
    elif confidence >= 0.6:
        size_multiplier = 1.0  # Standard size
    else:
        size_multiplier = 0.5  # Reduce size for low confidence
    
    position_size = base_size * size_multiplier
    
    return position_size
```

### Stop Loss Guidelines

```python
STOP_LOSS_GUIDELINES = {
    'daily': {
        'tight': 0.01,      # 1% for high confidence
        'normal': 0.02,     # 2% for medium confidence
        'wide': 0.03        # 3% for low confidence
    },
    'weekly': {
        'tight': 0.025,     # 2.5% for high confidence
        'normal': 0.035,    # 3.5% for medium confidence
        'wide': 0.05        # 5% for low confidence
    },
    'monthly': {
        'tight': 0.05,      # 5% for high confidence
        'normal': 0.075,    # 7.5% for medium confidence
        'wide': 0.10        # 10% for low confidence
    }
}
```

### Profit Taking

```python
def calculate_profit_targets(entry_price: float, direction: str, horizon: str):
    """Calculate profit targets based on timeframe."""
    
    if horizon == 'daily':
        targets = [0.02, 0.03, 0.05]  # 2%, 3%, 5%
    elif horizon == 'weekly':
        targets = [0.05, 0.075, 0.10]  # 5%, 7.5%, 10%
    else:  # monthly
        targets = [0.10, 0.15, 0.20]  # 10%, 15%, 20%
    
    if direction == 'up':
        return [entry_price * (1 + target) for target in targets]
    else:
        return [entry_price * (1 - target) for target in targets]
```

## Multi-Timeframe Analysis

### Combining Timeframes

```python
def get_multi_timeframe_analysis(symbol: str, fno_engine: FNOEngine):
    """Get comprehensive multi-timeframe analysis."""
    
    # Get predictions for all timeframes
    daily = fno_engine.predict_probability(symbol, HorizonType.DAILY)
    weekly = fno_engine.predict_probability(symbol, HorizonType.WEEKLY)
    monthly = fno_engine.predict_probability(symbol, HorizonType.MONTHLY)
    
    # Calculate weighted probabilities
    weights = {'daily': 0.5, 'weekly': 0.3, 'monthly': 0.2}
    
    weighted_up = (
        daily.up_probability * weights['daily'] +
        weekly.up_probability * weights['weekly'] +
        monthly.up_probability * weights['monthly']
    )
    
    weighted_down = (
        daily.down_probability * weights['daily'] +
        weekly.down_probability * weights['weekly'] +
        monthly.down_probability * weights['monthly']
    )
    
    # Generate recommendation
    if weighted_up > 0.6:
        recommendation = f"BUY - Strong bullish signals ({weighted_up:.1%})"
    elif weighted_down > 0.6:
        recommendation = f"SELL - Strong bearish signals ({weighted_down:.1%})"
    else:
        recommendation = "HOLD - Mixed signals, consider hedging"
    
    return {
        'daily': daily,
        'weekly': weekly,
        'monthly': monthly,
        'weighted_up': weighted_up,
        'weighted_down': weighted_down,
        'recommendation': recommendation
    }
```

### Timeframe Alignment

```python
def check_timeframe_alignment(analysis):
    """Check if timeframes are aligned."""
    
    daily_direction = 'up' if analysis['daily'].up_probability > analysis['daily'].down_probability else 'down'
    weekly_direction = 'up' if analysis['weekly'].up_probability > analysis['weekly'].down_probability else 'down'
    monthly_direction = 'up' if analysis['monthly'].up_probability > analysis['monthly'].down_probability else 'down'
    
    # Check alignment
    if daily_direction == weekly_direction == monthly_direction:
        alignment = "PERFECT"
        strength = "Very Strong"
    elif daily_direction == weekly_direction or weekly_direction == monthly_direction:
        alignment = "PARTIAL"
        strength = "Moderate"
    else:
        alignment = "MIXED"
        strength = "Weak"
    
    return {
        'alignment': alignment,
        'strength': strength,
        'daily': daily_direction,
        'weekly': weekly_direction,
        'monthly': monthly_direction
    }
```

## Portfolio Management

### Diversification

```python
def analyze_portfolio_risk(symbols: List[str], fno_engine: FNOEngine):
    """Analyze portfolio risk across multiple symbols."""
    
    portfolio_analysis = {}
    
    for symbol in symbols:
        # Get multi-timeframe analysis
        analysis = get_multi_timeframe_analysis(symbol, fno_engine)
        
        # Calculate risk score
        risk_score = calculate_risk_score(analysis)
        
        portfolio_analysis[symbol] = {
            'analysis': analysis,
            'risk_score': risk_score,
            'recommendation': analysis['recommendation']
        }
    
    # Portfolio-level insights
    total_risk = sum(analysis['risk_score'] for analysis in portfolio_analysis.values())
    avg_risk = total_risk / len(symbols)
    
    return {
        'individual_analysis': portfolio_analysis,
        'total_risk': total_risk,
        'average_risk': avg_risk,
        'diversification_score': calculate_diversification_score(portfolio_analysis)
    }
```

### Correlation Analysis

```python
def check_correlation_risk(symbols: List[str]):
    """Check for correlation risk in portfolio."""
    
    # High correlation pairs to avoid
    high_correlation_pairs = [
        ('NIFTY', 'BANKNIFTY'),
        ('RELIANCE', 'ONGC'),
        ('TCS', 'INFY'),
        ('HDFC', 'ICICIBANK')
    ]
    
    correlation_warnings = []
    
    for pair in high_correlation_pairs:
        if pair[0] in symbols and pair[1] in symbols:
            correlation_warnings.append(f"High correlation between {pair[0]} and {pair[1]}")
    
    return correlation_warnings
```

## Trading Session Examples

### Morning Analysis

```python
def morning_analysis(fno_engine: FNOEngine):
    """Perform morning market analysis."""
    
    print("🌅 Morning Market Analysis")
    print("=" * 50)
    
    # Key symbols to analyze
    key_symbols = ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS', 'INFY']
    
    for symbol in key_symbols:
        print(f"\n📊 {symbol} Analysis:")
        
        # Get daily prediction
        result = fno_engine.predict_probability(symbol, HorizonType.DAILY)
        
        if result:
            print(f"   Up: {result.up_probability:.1%}")
            print(f"   Down: {result.down_probability:.1%}")
            print(f"   Confidence: {result.confidence_score:.1%}")
            
            # Quick recommendation
            if result.confidence_score > 0.7:
                if result.up_probability > 0.6:
                    print(f"   💡 BUY {symbol}")
                elif result.down_probability > 0.6:
                    print(f"   💡 SELL {symbol}")
                else:
                    print(f"   💡 HOLD {symbol}")
            else:
                print(f"   💡 WAIT for clearer signals")
```

### Intraday Trading

```python
def intraday_trading_signals(fno_engine: FNOEngine):
    """Get intraday trading signals."""
    
    print("📈 Intraday Trading Signals")
    print("=" * 50)
    
    # Focus on liquid symbols for intraday
    intraday_symbols = ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS']
    
    signals = []
    
    for symbol in intraday_symbols:
        result = fno_engine.predict_probability(symbol, HorizonType.DAILY)
        
        if result and result.confidence_score > 0.75:
            signal = {
                'symbol': symbol,
                'direction': 'BUY' if result.up_probability > result.down_probability else 'SELL',
                'probability': max(result.up_probability, result.down_probability),
                'confidence': result.confidence_score,
                'entry': get_current_price(symbol),
                'stop_loss': calculate_stop_loss(symbol, result),
                'target': calculate_target(symbol, result)
            }
            signals.append(signal)
    
    # Sort by confidence
    signals.sort(key=lambda x: x['confidence'], reverse=True)
    
    return signals
```

### End-of-Day Review

```python
def end_of_day_review(fno_engine: FNOEngine, trades: List[Dict]):
    """Perform end-of-day review."""
    
    print("🌆 End-of-Day Review")
    print("=" * 50)
    
    # Review today's trades
    total_trades = len(trades)
    winning_trades = sum(1 for trade in trades if trade['pnl'] > 0)
    losing_trades = total_trades - winning_trades
    
    print(f"📊 Trading Summary:")
    print(f"   Total Trades: {total_trades}")
    print(f"   Winning Trades: {winning_trades}")
    print(f"   Losing Trades: {losing_trades}")
    print(f"   Win Rate: {winning_trades/total_trades:.1%}")
    
    # Calculate total P&L
    total_pnl = sum(trade['pnl'] for trade in trades)
    print(f"   Total P&L: ₹{total_pnl:,.2f}")
    
    # Review prediction accuracy
    accuracy = review_prediction_accuracy(trades)
    print(f"   Prediction Accuracy: {accuracy:.1%}")
    
    # Plan for tomorrow
    print(f"\n📅 Tomorrow's Plan:")
    tomorrow_signals = get_tomorrow_signals(fno_engine)
    for signal in tomorrow_signals:
        print(f"   {signal['symbol']}: {signal['recommendation']}")
```

## Advanced Strategies

### Options Trading

```python
def options_strategy(symbol: str, prediction: ProbabilityResult):
    """Generate options trading strategy."""
    
    if prediction.confidence_score < 0.7:
        return "Low confidence - avoid options trading"
    
    current_price = get_current_price(symbol)
    
    if prediction.up_probability > 0.7:
        # Bullish strategy
        strategy = {
            'type': 'BULLISH',
            'recommendation': 'Buy Call Options',
            'strike': current_price * 1.02,  # 2% above current
            'expiry': get_next_expiry(),
            'stop_loss': current_price * 0.98,
            'target': current_price * 1.05
        }
    elif prediction.down_probability > 0.7:
        # Bearish strategy
        strategy = {
            'type': 'BEARISH',
            'recommendation': 'Buy Put Options',
            'strike': current_price * 0.98,  # 2% below current
            'expiry': get_next_expiry(),
            'stop_loss': current_price * 1.02,
            'target': current_price * 0.95
        }
    else:
        # Neutral strategy
        strategy = {
            'type': 'NEUTRAL',
            'recommendation': 'Iron Condor or Straddle',
            'strike': current_price,
            'expiry': get_next_expiry(),
            'stop_loss': current_price * 0.97,
            'target': current_price * 1.03
        }
    
    return strategy
```

### Hedging Strategies

```python
def hedging_strategy(portfolio: Dict[str, float], fno_engine: FNOEngine):
    """Generate hedging strategy for portfolio."""
    
    # Calculate portfolio beta
    portfolio_beta = calculate_portfolio_beta(portfolio)
    
    # Get market prediction
    market_prediction = fno_engine.predict_probability("NIFTY", HorizonType.DAILY)
    
    if market_prediction.down_probability > 0.6:
        # Market bearish - hedge portfolio
        hedge_amount = portfolio_beta * sum(portfolio.values()) * 0.3
        hedge_strategy = {
            'type': 'PORTFOLIO_HEDGE',
            'instrument': 'NIFTY_PUT',
            'amount': hedge_amount,
            'strike': get_current_price("NIFTY") * 0.98,
            'expiry': get_next_expiry()
        }
    else:
        hedge_strategy = {
            'type': 'NO_HEDGE',
            'reason': 'Market conditions favorable'
        }
    
    return hedge_strategy
```

## Risk Management Rules

### Golden Rules

1. **Never risk more than 2% of capital per trade**
2. **Always set stop losses for every position**
3. **Take partial profits at 50% of target**
4. **Don't chase losses - stick to your plan**
5. **Keep a trading journal**

### Position Sizing Formula

```python
def calculate_position_size(capital: float, risk_per_trade: float, stop_loss_pct: float):
    """Calculate position size based on risk management rules."""
    
    # Risk amount = Capital * Risk per trade
    risk_amount = capital * risk_per_trade
    
    # Position size = Risk amount / Stop loss percentage
    position_size = risk_amount / stop_loss_pct
    
    return position_size

# Example
capital = 100000  # ₹1 lakh
risk_per_trade = 0.02  # 2%
stop_loss = 0.025  # 2.5%

position_size = calculate_position_size(capital, risk_per_trade, stop_loss)
print(f"Position Size: ₹{position_size:,.2f}")
```

### Stop Loss Guidelines

```python
STOP_LOSS_RULES = {
    'daily_trades': {
        'high_confidence': 0.015,  # 1.5%
        'medium_confidence': 0.025,  # 2.5%
        'low_confidence': 0.035   # 3.5%
    },
    'weekly_trades': {
        'high_confidence': 0.03,   # 3%
        'medium_confidence': 0.045,  # 4.5%
        'low_confidence': 0.06    # 6%
    },
    'monthly_trades': {
        'high_confidence': 0.06,   # 6%
        'medium_confidence': 0.09,  # 9%
        'low_confidence': 0.12    # 12%
    }
}
```

## Performance Tracking

### Trading Journal

```python
class TradingJournal:
    def __init__(self):
        self.trades = []
    
    def add_trade(self, trade_data):
        """Add a new trade to the journal."""
        trade_data['date'] = datetime.now()
        trade_data['trade_id'] = len(self.trades) + 1
        self.trades.append(trade_data)
    
    def get_statistics(self):
        """Get trading statistics."""
        if not self.trades:
            return {}
        
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t['pnl'] > 0)
        losing_trades = total_trades - winning_trades
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        avg_win = sum(t['pnl'] for t in self.trades if t['pnl'] > 0) / winning_trades if winning_trades > 0 else 0
        avg_loss = sum(t['pnl'] for t in self.trades if t['pnl'] < 0) / losing_trades if losing_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        }
```

### Performance Metrics

```python
def calculate_performance_metrics(trades: List[Dict]):
    """Calculate comprehensive performance metrics."""
    
    # Basic metrics
    total_trades = len(trades)
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] < 0]
    
    # Win rate
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    
    # Average win/loss
    avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
    
    # Profit factor
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    
    # Maximum drawdown
    cumulative_pnl = []
    running_total = 0
    for trade in trades:
        running_total += trade['pnl']
        cumulative_pnl.append(running_total)
    
    max_drawdown = calculate_max_drawdown(cumulative_pnl)
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': calculate_sharpe_ratio(trades),
        'calmar_ratio': calculate_calmar_ratio(trades, max_drawdown)
    }
```

## Best Practices

### Do's

1. **Follow your trading plan consistently**
2. **Use proper position sizing**
3. **Set stop losses for every trade**
4. **Keep detailed trading records**
5. **Review performance regularly**
6. **Stay disciplined with risk management**

### Don'ts

1. **Don't risk more than 2% per trade**
2. **Don't chase losses**
3. **Don't trade without a plan**
4. **Don't ignore stop losses**
5. **Don't overtrade**
6. **Don't let emotions drive decisions**

## Conclusion

The FNO RAG system provides powerful tools for making informed trading decisions. However, success in trading requires:

1. **Proper risk management**
2. **Disciplined execution**
3. **Continuous learning**
4. **Emotional control**
5. **Regular performance review**

Remember: The system provides probabilities, not certainties. Always use proper risk management and never risk more than you can afford to lose.
