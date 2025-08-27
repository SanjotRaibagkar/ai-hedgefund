# 💬 Natural Language Interface Documentation

## Overview

The Natural Language Interface allows users to interact with the FNO RAG system using plain English queries, providing an intuitive way to get trading predictions and market insights.

## Architecture

### Core Components

1. **Query Parser**: Intent recognition and entity extraction
2. **Response Generator**: Structured response creation
3. **Chat Interface**: User interaction management
4. **GROQ Integration**: LLM-powered responses (optional)

### Data Flow

```
User Query → Query Parser → Intent Recognition → Response Generator → Structured Response
     ↓
Entity Extraction → Symbol/Timeframe Detection → Probability Prediction → Enhanced Response
```

## Query Processing

### Intent Recognition

The system recognizes several types of intents:

```python
INTENT_TYPES = {
    'probability_prediction': 'Get probability for symbol movement',
    'stock_search': 'Find stocks matching criteria',
    'market_analysis': 'General market analysis',
    'help': 'Get help and usage information',
    'system_status': 'Check system status'
}
```

### Entity Extraction

#### Symbols

```python
def extract_symbols(query: str) -> List[str]:
    """Extract stock symbols from query."""
    
    # Common FNO symbols
    fno_symbols = [
        'NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS', 'INFY', 'HDFC', 
        'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL',
        'AXISBANK', 'ASIANPAINT', 'MARUTI', 'HCLTECH'
    ]
    
    # Extract symbols from query
    found_symbols = []
    query_upper = query.upper()
    
    for symbol in fno_symbols:
        if symbol in query_upper:
            found_symbols.append(symbol)
    
    return found_symbols
```

#### Timeframes

```python
def extract_timeframe(query: str) -> HorizonType:
    """Extract timeframe from query."""
    
    query_lower = query.lower()
    
    # Daily timeframes
    if any(word in query_lower for word in ['tomorrow', 'daily', '1 day', 'today']):
        return HorizonType.DAILY
    
    # Weekly timeframes
    elif any(word in query_lower for word in ['week', 'weekly', '7 day', 'next week']):
        return HorizonType.WEEKLY
    
    # Monthly timeframes
    elif any(word in query_lower for word in ['month', 'monthly', '30 day', 'next month']):
        return HorizonType.MONTHLY
    
    # Default to daily
    return HorizonType.DAILY
```

#### Directions

```python
def extract_direction(query: str) -> str:
    """Extract movement direction from query."""
    
    query_lower = query.lower()
    
    # Upward movement
    if any(word in query_lower for word in ['up', 'rise', 'gain', 'bullish', 'increase']):
        return 'up'
    
    # Downward movement
    elif any(word in query_lower for word in ['down', 'fall', 'drop', 'bearish', 'decrease']):
        return 'down'
    
    # Neutral/any direction
    return 'any'
```

#### Percentage Moves

```python
def extract_percentage_move(query: str) -> float:
    """Extract percentage move from query."""
    
    import re
    
    # Find percentage patterns
    percent_patterns = [
        r'(\d+)%',           # "5%"
        r'(\d+)\s*percent',  # "5 percent"
        r'(\d+)\s*per\s*cent'  # "5 per cent"
    ]
    
    for pattern in percent_patterns:
        match = re.search(pattern, query.lower())
        if match:
            return float(match.group(1)) / 100.0
    
    return None
```

## Query Examples

### Probability Prediction Queries

```python
# Basic probability queries
queries = [
    "What's the probability of NIFTY moving up tomorrow?",
    "Predict RELIANCE movement for next month",
    "What's the chance of TCS going down this week?",
    "Show me INFY probability for tomorrow",
    "Tell me about BANKNIFTY's daily outlook",
    "What are the odds of HDFC rising next week?"
]

# Parsed results
{
    'intent': 'probability_prediction',
    'symbols': ['NIFTY'],
    'horizon': HorizonType.DAILY,
    'direction': 'up'
}
```

### Stock Search Queries

```python
# Stock search queries
queries = [
    "Give me FNO stocks which can move 5% next week",
    "Find stocks with high daily up probability",
    "Show me bullish stocks for tomorrow",
    "Which stocks are likely to fall this month?",
    "Find high-probability trading opportunities"
]

# Parsed results
{
    'intent': 'stock_search',
    'horizon': HorizonType.WEEKLY,
    'direction': 'any',
    'percentage_move': 0.05
}
```

### Market Analysis Queries

```python
# Market analysis queries
queries = [
    "What's the market outlook today?",
    "Give me a market summary",
    "What's happening in the FNO market?",
    "Market analysis for this week"
]

# Parsed results
{
    'intent': 'market_analysis',
    'horizon': HorizonType.DAILY
}
```

## Response Generation

### Structured Responses

```python
def generate_response(parsed_query: Dict, fno_engine: FNOEngine) -> str:
    """Generate structured response based on parsed query."""
    
    intent = parsed_query.get('intent')
    
    if intent == 'probability_prediction':
        return generate_probability_response(parsed_query, fno_engine)
    elif intent == 'stock_search':
        return generate_stock_search_response(parsed_query, fno_engine)
    elif intent == 'market_analysis':
        return generate_market_analysis_response(parsed_query, fno_engine)
    else:
        return generate_help_response()
```

### Probability Response

```python
def generate_probability_response(parsed_query: Dict, fno_engine: FNOEngine) -> str:
    """Generate probability prediction response."""
    
    symbol = parsed_query.get('symbols', ['NIFTY'])[0]
    horizon = parsed_query.get('horizon', HorizonType.DAILY)
    direction = parsed_query.get('direction', 'any')
    
    # Get prediction
    result = fno_engine.predict_probability(symbol, horizon)
    
    if not result:
        return f"Sorry, I couldn't get a prediction for {symbol} at this time."
    
    # Format response
    response = f"📊 **{symbol} {horizon.value.title()} Prediction**\n\n"
    
    if direction == 'up':
        response += f"🟢 **Up Probability**: {result.up_probability:.1%}\n"
        response += f"🔴 Down Probability: {result.down_probability:.1%}\n"
        response += f"⚪ Neutral Probability: {result.neutral_probability:.1%}\n"
    elif direction == 'down':
        response += f"🟢 Up Probability: {result.up_probability:.1%}\n"
        response += f"🔴 **Down Probability**: {result.down_probability:.1%}\n"
        response += f"⚪ Neutral Probability: {result.neutral_probability:.1%}\n"
    else:
        response += f"🟢 Up Probability: {result.up_probability:.1%}\n"
        response += f"🔴 Down Probability: {result.down_probability:.1%}\n"
        response += f"⚪ Neutral Probability: {result.neutral_probability:.1%}\n"
    
    response += f"\n🎯 **Confidence**: {result.confidence_score:.1%}\n"
    
    # Add trading recommendation
    recommendation = get_trading_recommendation(result, direction)
    response += f"\n💡 **Recommendation**: {recommendation}"
    
    return response
```

### Stock Search Response

```python
def generate_stock_search_response(parsed_query: Dict, fno_engine: FNOEngine) -> str:
    """Generate stock search response."""
    
    horizon = parsed_query.get('horizon', HorizonType.DAILY)
    direction = parsed_query.get('direction', 'any')
    percentage_move = parsed_query.get('percentage_move')
    
    # Get all FNO symbols
    symbols = fno_engine.data_processor.get_all_fno_symbols()
    
    # Get predictions for all symbols
    results = []
    for symbol in symbols[:20]:  # Limit to top 20 for performance
        result = fno_engine.predict_probability(symbol, horizon)
        if result:
            results.append((symbol, result))
    
    # Filter and sort results
    if direction == 'up':
        results = [(s, r) for s, r in results if r.up_probability > 0.6]
        results.sort(key=lambda x: x[1].up_probability, reverse=True)
    elif direction == 'down':
        results = [(s, r) for s, r in results if r.down_probability > 0.6]
        results.sort(key=lambda x: x[1].down_probability, reverse=True)
    else:
        # Sort by highest probability in any direction
        results.sort(key=lambda x: max(x[1].up_probability, x[1].down_probability), reverse=True)
    
    # Generate response
    response = f"🔍 **Stock Search Results** ({horizon.value.title()})\n\n"
    
    if not results:
        response += "No stocks found matching your criteria."
        return response
    
    for i, (symbol, result) in enumerate(results[:10]):
        if direction == 'up':
            prob = result.up_probability
            emoji = "🟢"
        elif direction == 'down':
            prob = result.down_probability
            emoji = "🔴"
        else:
            prob = max(result.up_probability, result.down_probability)
            emoji = "🟢" if result.up_probability > result.down_probability else "🔴"
        
        response += f"{i+1}. {emoji} **{symbol}**: {prob:.1%} probability\n"
    
    return response
```

## Chat Interface

### FNOEngine Chat Method

```python
def chat(self, query: str) -> str:
    """Process natural language query and return response."""
    
    try:
        # Parse query
        parsed = self.chat_interface.parse_query(query)
        
        # Generate response based on intent
        if parsed['intent'] == 'probability_prediction':
            return self._handle_probability_query(parsed)
        elif parsed['intent'] == 'stock_search':
            return self._handle_stock_search_query(parsed)
        elif parsed['intent'] == 'market_analysis':
            return self._handle_market_analysis_query(parsed)
        elif parsed['intent'] == 'help':
            return self._get_help_message()
        else:
            return "I'm not sure how to help with that. Try asking about stock predictions or market analysis."
    
    except Exception as e:
        self.logger.error(f"Chat error: {e}")
        return "Sorry, I encountered an error processing your request. Please try again."
```

### Query Parsing

```python
def parse_query(self, query: str) -> Dict[str, Any]:
    """Parse natural language query into structured format."""
    
    parsed = {
        'intent': None,
        'symbols': [],
        'horizon': None,
        'direction': None,
        'percentage_move': None
    }
    
    query_lower = query.lower()
    
    # Check for probability prediction intent
    if any(word in query_lower for word in ['probability', 'chance', 'move', 'tomorrow', 'week', 'month']):
        parsed['intent'] = 'probability_prediction'
        
        # Extract symbols
        symbols = self._extract_symbols(query)
        if symbols:
            parsed['symbols'] = symbols
        
        # Extract horizon
        if 'tomorrow' in query_lower or 'daily' in query_lower:
            parsed['horizon'] = HorizonType.DAILY
        elif 'week' in query_lower or 'weekly' in query_lower:
            parsed['horizon'] = HorizonType.WEEKLY
        elif 'month' in query_lower or 'monthly' in query_lower:
            parsed['horizon'] = HorizonType.MONTHLY
        
        # Extract direction
        if any(word in query_lower for word in ['up', 'rise', 'gain', 'bullish']):
            parsed['direction'] = 'up'
        elif any(word in query_lower for word in ['down', 'fall', 'drop', 'bearish']):
            parsed['direction'] = 'down'
    
    # Check for stock search intent
    elif any(word in query_lower for word in ['find', 'search', 'which', 'stocks', 'give me', 'show me']):
        parsed['intent'] = 'stock_search'
        
        # Extract horizon and direction (same logic as above)
        # ... (similar extraction logic)
    
    return parsed
```

## GROQ Integration (Optional)

### Enhanced Responses with LLM

```python
def generate_enhanced_response(query: str, base_response: str, context: Dict) -> str:
    """Generate enhanced response using GROQ LLM."""
    
    if not self.groq_client:
        return base_response
    
    # Create enhanced prompt
    prompt = f"""
    User Query: {query}
    
    Base Response: {base_response}
    
    Context: {context}
    
    Please enhance this response to be more conversational and helpful. 
    Add relevant insights and make it more engaging while keeping the core information accurate.
    """
    
    try:
        response = self.groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        self.logger.warning(f"GROQ enhancement failed: {e}")
        return base_response
```

## Usage Examples

### Basic Usage

```python
from src.fno_rag import FNOEngine

# Initialize the system
fno_engine = FNOEngine()

# Ask questions in natural language
queries = [
    "What's the probability of NIFTY moving up tomorrow?",
    "Give me FNO stocks which can move 5% next week",
    "Tell me about RELIANCE's monthly outlook",
    "Find stocks with high daily up probability"
]

for query in queries:
    response = fno_engine.chat(query)
    print(f"Q: {query}")
    print(f"A: {response}\n")
```

### Interactive Chat

```python
def interactive_chat():
    """Interactive chat interface."""
    
    fno_engine = FNOEngine()
    
    print("🤖 FNO RAG Chat Interface")
    print("Type 'quit' to exit\n")
    
    while True:
        query = input("You: ").strip()
        
        if query.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye! 👋")
            break
        
        if not query:
            continue
        
        response = fno_engine.chat(query)
        print(f"🤖: {response}\n")
```

## Response Templates

### Probability Response Template

```python
PROBABILITY_TEMPLATE = """
📊 **{symbol} {horizon} Prediction**

🟢 Up Probability: {up_prob:.1%}
🔴 Down Probability: {down_prob:.1%}
⚪ Neutral Probability: {neutral_prob:.1%}

🎯 Confidence: {confidence:.1%}

💡 Recommendation: {recommendation}

📈 Historical Context: {rag_context}
"""
```

### Stock Search Template

```python
STOCK_SEARCH_TEMPLATE = """
🔍 **Stock Search Results** ({horizon})

{stock_list}

💡 Found {count} stocks matching your criteria.
🎯 Top recommendation: {top_stock} ({top_prob:.1%})
"""
```

### Market Analysis Template

```python
MARKET_ANALYSIS_TEMPLATE = """
📈 **Market Analysis** ({horizon})

🏆 Top Performers:
{top_performers}

📉 Weak Performers:
{weak_performers}

🎯 Overall Market Sentiment: {sentiment}
📊 Market Confidence: {confidence:.1%}
"""
```

## Error Handling

### Common Error Responses

```python
ERROR_RESPONSES = {
    'no_symbol': "I couldn't identify a stock symbol in your query. Please specify a symbol like NIFTY, RELIANCE, etc.",
    'no_prediction': "Sorry, I couldn't get a prediction for that symbol at this time. Please try again later.",
    'invalid_timeframe': "Please specify a timeframe: tomorrow/daily, week/weekly, or month/monthly.",
    'system_error': "I encountered an error processing your request. Please try again.",
    'no_data': "No data available for the requested symbol or timeframe."
}
```

### Graceful Degradation

```python
def handle_query_error(query: str, error: Exception) -> str:
    """Handle query errors gracefully."""
    
    error_type = type(error).__name__
    
    if 'symbol' in str(error).lower():
        return ERROR_RESPONSES['no_symbol']
    elif 'prediction' in str(error).lower():
        return ERROR_RESPONSES['no_prediction']
    elif 'timeframe' in str(error).lower():
        return ERROR_RESPONSES['invalid_timeframe']
    else:
        return ERROR_RESPONSES['system_error']
```

## Best Practices

1. **Query Understanding**: Use multiple patterns for intent recognition
2. **Response Quality**: Provide structured, informative responses
3. **Error Handling**: Graceful degradation for errors
4. **Performance**: Cache frequent queries and responses
5. **User Experience**: Clear, conversational language
6. **Context Awareness**: Include relevant market context

## Future Enhancements

1. **Multi-language Support**: Support for multiple languages
2. **Voice Interface**: Speech-to-text and text-to-speech
3. **Context Memory**: Remember user preferences and history
4. **Advanced NLP**: Use transformer models for better understanding
5. **Personalization**: Adapt responses based on user profile
6. **Integration**: Connect with external data sources
