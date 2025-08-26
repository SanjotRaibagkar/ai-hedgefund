#!/usr/bin/env python3
"""
MokshTechandInvestment Screening Dashboard
Web application for stock screening and market analysis.
"""

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime
import pandas as pd
import sys
import os

# Add src to path - fix the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..', '..')
sys.path.insert(0, project_root)

try:
    from src.screening.screening_manager import ScreeningManager
    print("✅ Successfully imported ScreeningManager")
except ImportError as e:
    print(f"❌ Import error: {e}")
    # Fallback import
    sys.path.append(os.path.join(project_root, 'src'))
    from screening.screening_manager import ScreeningManager

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "MokshTechandInvestment - AI Screening Dashboard"

# Initialize screening manager
screening_manager = ScreeningManager()

# Global variable to store current EOD results for search functionality
current_eod_results = {}

# App layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🚀 MokshTechandInvestment", className="text-primary mb-0"),
            html.H4("AI-Powered Stock Screening & Market Analysis", className="text-muted"),
            html.Hr()
        ], width=12)
    ], className="mb-4"),
    
    # Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Control Panel"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Run EOD Screening", id="btn-eod", color="primary", className="me-2"),
                            dbc.Button("Run Intraday Screening", id="btn-intraday", color="success", className="me-2", disabled=True),
                            dbc.Button("Run Options Analysis", id="btn-options", color="warning", className="me-2"),
                            dbc.Button("Run Market Predictions", id="btn-predictions", color="info", className="me-2"),
                            dbc.Button("Run Comprehensive Screening", id="btn-comprehensive", color="dark")
                        ], width=12)
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Stock List (comma-separated):"),
                            html.Div([
                                html.Small("💡 Leave empty to screen ALL stocks from database", className="text-muted"),
                                dcc.Textarea(
                                    id="stock-list",
                                    value="RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS",
                                    placeholder="Enter stock symbols or leave empty for all stocks...",
                                    style={'width': '100%', 'height': 60}
                                )
                            ])
                        ], width=4),
                        dbc.Col([
                            html.Label("Note:"),
                            html.Div([
                                html.Small("🚫 Intraday Screening is temporarily disabled", className="text-warning"),
                                html.Br(),
                                html.Small("✅ EOD Screening is fully functional", className="text-success"),
                                html.Br(),
                                html.Small("📊 Real-time progress shown in terminal", className="text-info")
                            ])
                        ], width=4),
                        dbc.Col([
                            html.Label("Risk-Reward Ratio:"),
                            dcc.Slider(
                                id="risk-reward-slider",
                                min=1.0,
                                max=5.0,
                                step=0.1,
                                value=2.0,
                                marks={i: str(i) for i in range(1, 6)}
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Analysis Mode:"),
                            dcc.Dropdown(
                                id="analysis-mode",
                                options=[
                                    {'label': 'Basic', 'value': 'basic'},
                                    {'label': 'Enhanced', 'value': 'enhanced'},
                                    {'label': 'Comprehensive', 'value': 'comprehensive'}
                                ],
                                value='comprehensive',
                                style={'width': '100%'}
                            )
                        ], width=4)
                    ])
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Progress Indicators
    dbc.Row([
        dbc.Col([
            html.Div(id="eod-progress", style={'display': 'none'})
        ], width=12)
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            html.Div(id="intraday-progress", style={'display': 'none'})
        ], width=12)
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            html.Div(id="options-progress", style={'display': 'none'})
        ], width=12)
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            html.Div(id="predictions-progress", style={'display': 'none'})
        ], width=12)
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            html.Div(id="comprehensive-progress", style={'display': 'none'})
        ], width=12)
    ], className="mb-3"),
    
    # Results Display
    dbc.Row([
        # EOD Signals
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📈 EOD Trading Signals"),
                dbc.CardBody([
                    html.Div(id="eod-results", children="Click 'Run EOD Screening' to start...")
                ])
            ])
        ], width=6),
        
        # Intraday Signals
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("⚡ Intraday Trading Signals"),
                dbc.CardBody([
                    html.Div(id="intraday-results", children="Click 'Run Intraday Screening' to start...")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Options Analysis
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Options Analysis"),
                dbc.CardBody([
                    html.Div(id="options-results", children="Click 'Run Options Analysis' to start...")
                ])
            ])
        ], width=6),
        
        # Market Predictions
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🔮 Market Predictions"),
                dbc.CardBody([
                    html.Div(id="predictions-results", children="Click 'Run Market Predictions' to start...")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Comprehensive Results
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🎯 Comprehensive Analysis"),
                dbc.CardBody([
                    html.Div(id="comprehensive-results", children="Click 'Run Comprehensive Screening' to start...")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("© 2024 MokshTechandInvestment. All rights reserved.", className="text-center text-muted")
        ], width=12)
    ])
], fluid=True)

# Helper Functions
def create_signal_list(signals, signal_type):
    """Create a formatted list of signals with proper styling."""
    if not signals:
        return html.P("No signals found.", className="text-muted")
    
    signal_items = []
    for i, signal in enumerate(signals):
        # Determine color based on signal type
        if signal_type == 'bullish':
            color_class = "text-success"
            icon = "📈"
        else:
            color_class = "text-danger"
            icon = "📉"
        
        # Create signal card
        signal_card = dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.H6(f"{icon} {signal['symbol']}", className=f"{color_class} mb-2"),
                    html.P(f"Confidence: {signal['confidence']}%", className="mb-1"),
                    html.P(f"Entry: ₹{signal['entry_price']:.2f} | SL: ₹{signal['stop_loss']:.2f}", className="mb-1"),
                    html.P(f"Target 1: ₹{signal['targets']['T1']:.2f} | Risk-Reward: {signal['risk_reward_ratio']:.2f}", className="mb-0"),
                ])
            ])
        ], className="mb-2")
        
        signal_items.append(signal_card)
    
    return signal_items

def create_intraday_signal_list(signals, signal_type):
    """Create a formatted list of intraday signals with proper styling."""
    if not signals:
        return html.P("No signals found.", className="text-muted")
    
    signal_items = []
    for i, signal in enumerate(signals):
        # Determine color and icon based on signal type
        if signal_type == 'breakout':
            color_class = "text-primary"
            icon = "🚀"
        elif signal_type == 'reversal':
            color_class = "text-warning"
            icon = "🔄"
        else:  # momentum
            color_class = "text-info"
            icon = "💨"
        
        # Create signal card
        signal_card = dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.H6(f"{icon} {signal.get('ticker', signal.get('symbol', 'N/A'))}", className=f"{color_class} mb-2"),
                    html.P(f"Type: {signal.get('signal_type', 'N/A')} | Confidence: {signal.get('confidence', 0)}%", className="mb-1"),
                    html.P(f"Entry: ₹{signal.get('entry_price', 0):.2f} | SL: ₹{signal.get('stop_loss', 0):.2f}", className="mb-1"),
                    html.P(f"Risk-Reward: {signal.get('risk_reward_ratio', 0):.2f}", className="mb-0"),
                ])
            ])
        ], className="mb-2")
        
        signal_items.append(signal_card)
    
    return signal_items

# Callbacks
@app.callback(
    [Output("eod-results", "children"),
     Output("eod-progress", "children"),
     Output("eod-progress", "style")],
    Input("btn-eod", "n_clicks"),
    Input("stock-list", "value"),
    Input("risk-reward-slider", "value"),
    Input("analysis-mode", "value"),
    prevent_initial_call=True
)
def run_eod_screening(n_clicks, stock_list, risk_reward, analysis_mode):
    if n_clicks is None:
        return "Click 'Run EOD Screening' to start...", None, {'display': 'none'}
    
    try:
        # Show progress indicator with animated elements
        progress = dbc.Alert([
            dbc.Spinner(size="lg", spinner_class_name="me-3"),
            html.Div([
                html.H5("🔄 EOD Screening in Progress...", className="mb-2"),
                html.P("Analyzing 2,129+ stocks for trading signals", className="mb-2"),
                html.Div([
                    html.Span("📊 Loading market data...", className="me-3"),
                    html.Span("⚡ Calculating indicators...", className="me-3"),
                    html.Span("🎯 Generating signals...", className="me-3"),
                    html.Span("📈 Filtering results...", className="me-3")
                ], className="text-muted mb-2"),
                html.Small("⏱️ Estimated time: 30-60 seconds for comprehensive analysis", className="text-info"),
                html.Br(),
                html.Small("💡 You can see real-time progress in the terminal/console", className="text-muted")
            ])
        ], color="info", className="mb-3")
        
        # Debug logging
        print(f"🔍 EOD Screening Debug:")
        print(f"   n_clicks: {n_clicks}")
        print(f"   stock_list: {repr(stock_list)}")
        print(f"   risk_reward: {risk_reward}")
        print(f"   analysis_mode: {analysis_mode}")
        
        # Parse stock list - handle None and empty cases
        if stock_list is None or stock_list.strip() == "":
            stocks = None  # This will trigger screening manager to use all symbols
            print(f"   ✅ Using all symbols from database (stocks=None)")
        else:
            stocks = [s.strip() for s in stock_list.split(",") if s.strip()]
            # If no valid stocks found, use all symbols
            if not stocks:
                stocks = None
                print(f"   ✅ No valid stocks found, using all symbols (stocks=None)")
            else:
                print(f"   ✅ Using specific stocks: {stocks}")
        
        print(f"   🚀 Starting screening...")
        
        # Show initial progress
        print(f"   📊 Progress: Starting analysis...")
        
        # Run screening with analysis mode
        results = screening_manager.get_eod_signals(stocks, risk_reward, analysis_mode)
        
        # Store results globally for search functionality
        global current_eod_results
        current_eod_results = results
        
        print(f"   ✅ Screening completed!")
        print(f"   📊 Results: {results['summary']['total_stocks']} stocks, {results['summary']['bullish_count']} bullish, {results['summary']['bearish_count']} bearish")
        
        # Create results display
        bullish_count = results['summary']['bullish_count']
        bearish_count = results['summary']['bearish_count']
        total_stocks = results['summary']['total_stocks']
        
        # Calculate neutral count
        neutral_count = total_stocks - bullish_count - bearish_count
        
        # Create organized results with sections
        content = [
            # Summary
            dbc.Alert([
                html.H5("📊 EOD Screening Results", className="mb-2"),
                html.P(f"Analysis Mode: {analysis_mode.title()} | Total Stocks: {total_stocks}", className="mb-1"),
                html.P(f"Bullish: {bullish_count} | Bearish: {bearish_count} | Neutral: {neutral_count}", className="mb-0")
            ], color="success", className="mb-3"),
            
            # Search and Filter
            dbc.Row([
                dbc.Col([
                    dcc.Input(
                        id="eod-search",
                        type="text",
                        placeholder="Search stocks by symbol/name...",
                        style={'width': '100%', 'padding': '8px', 'border': '1px solid #ddd', 'borderRadius': '4px'}
                    )
                ], width=6),
                dbc.Col([
                    dcc.Dropdown(
                        id="eod-filter",
                        options=[
                            {'label': 'All Signals', 'value': 'all'},
                            {'label': 'Bullish Only', 'value': 'bullish'},
                            {'label': 'Bearish Only', 'value': 'bearish'},
                            {'label': 'High Confidence (>80%)', 'value': 'high_confidence'}
                        ],
                        value='all',
                        style={'width': '100%'}
                    )
                ], width=6)
            ], className="mb-3"),
            
            # Results Sections
            dbc.Row([
                # Bullish Section
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(f"📈 Bullish Signals ({bullish_count})", className="text-success"),
                        dbc.CardBody([
                            html.Div(
                                id="bullish-signals",
                                children=create_signal_list(results.get('bullish_signals', []), 'bullish'),
                                style={'maxHeight': '400px', 'overflowY': 'auto'}
                            )
                        ])
                    ])
                ], width=4),
                
                # Bearish Section
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(f"📉 Bearish Signals ({bearish_count})", className="text-danger"),
                        dbc.CardBody([
                            html.Div(
                                id="bearish-signals",
                                children=create_signal_list(results.get('bearish_signals', []), 'bearish'),
                                style={'maxHeight': '400px', 'overflowY': 'auto'}
                            )
                        ])
                    ])
                ], width=4),
                
                # Neutral Section
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(f"➡️ Neutral Signals ({neutral_count})", className="text-muted"),
                        dbc.CardBody([
                            html.Div([
                                html.P("Stocks that don't meet bullish or bearish criteria.", className="text-muted"),
                                html.P("These may be good for watchlist or further analysis.", className="text-muted")
                            ])
                        ])
                    ])
                ], width=4)
            ])
        ]
        
        # Show message if no signals found
        if bullish_count == 0 and bearish_count == 0:
            content = [
                dbc.Alert([
                    html.H6("💡 No EOD Signals Found", className="text-info"),
                    html.P("This is normal with limited data. Historical data would generate more signals."),
                    html.P("✅ Analysis completed successfully!")
                ], color="info")
            ]
        
        return content, None, {'display': 'none'}
        
    except Exception as e:
        error_content = html.Div([
            html.H5("❌ Error occurred during EOD screening"),
            html.P(str(e), className="text-danger")
        ])
        return error_content, None, {'display': 'none'}

@app.callback(
    [Output("intraday-results", "children"),
     Output("intraday-progress", "children"),
     Output("intraday-progress", "style")],
    Input("btn-intraday", "n_clicks"),
    Input("stock-list", "value"),
    prevent_initial_call=True
)
def run_intraday_screening(n_clicks, stock_list):
    if n_clicks is None:
        return "Click 'Run Intraday Screening' to start...", None, {'display': 'none'}
    
    try:
        # Show progress indicator
        progress = dbc.Alert([
            dbc.Spinner(size="sm", spinner_class_name="me-2"),
            "🔄 Running Intraday Screening... Please wait while we analyze real-time data."
        ], color="info", className="mb-3")
        
        # Parse stock list - handle None and empty cases
        if stock_list is None or stock_list.strip() == "":
            stocks = None  # This will trigger screening manager to use all symbols
        else:
            stocks = [s.strip() for s in stock_list.split(",") if s.strip()]
            # If no valid stocks found, use all symbols
            if not stocks:
                stocks = None
        
        # Run screening
        results = screening_manager.get_intraday_signals(stocks)
        
        # Create results display
        breakout_count = results['summary']['breakout_count']
        reversal_count = results['summary']['reversal_count']
        momentum_count = results['summary']['momentum_count']
        total_stocks = results['summary']['total_stocks']
        
        # Create organized results with sections
        content = [
            # Summary
            dbc.Alert([
                html.H5("⚡ Intraday Screening Results", className="mb-2"),
                html.P(f"Total Stocks: {total_stocks}", className="mb-1"),
                html.P(f"Breakout: {breakout_count} | Reversal: {reversal_count} | Momentum: {momentum_count}", className="mb-0")
            ], color="success", className="mb-3"),
            
            # Results Sections
            dbc.Row([
                # Breakout Section
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(f"🚀 Breakout Signals ({breakout_count})", className="text-primary"),
                        dbc.CardBody([
                            html.Div(
                                id="breakout-signals",
                                children=create_intraday_signal_list(results.get('breakout_signals', []), 'breakout'),
                                style={'maxHeight': '400px', 'overflowY': 'auto'}
                            )
                        ])
                    ])
                ], width=4),
                
                # Reversal Section
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(f"🔄 Reversal Signals ({reversal_count})", className="text-warning"),
                        dbc.CardBody([
                            html.Div(
                                id="reversal-signals",
                                children=create_intraday_signal_list(results.get('reversal_signals', []), 'reversal'),
                                style={'maxHeight': '400px', 'overflowY': 'auto'}
                            )
                        ])
                    ])
                ], width=4),
                
                # Momentum Section
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(f"💨 Momentum Signals ({momentum_count})", className="text-info"),
                        dbc.CardBody([
                            html.Div(
                                id="momentum-signals",
                                children=create_intraday_signal_list(results.get('momentum_signals', []), 'momentum'),
                                style={'maxHeight': '400px', 'overflowY': 'auto'}
                            )
                        ])
                    ])
                ], width=4)
            ])
        ]
        
        # Show message if no signals found
        if breakout_count == 0 and reversal_count == 0 and momentum_count == 0:
            content = [
                dbc.Alert([
                    html.H6("💡 No Intraday Signals Found", className="text-info"),
                    html.P("This is normal with limited data. Real-time data feeds would generate more signals."),
                    html.P("✅ Analysis completed successfully!")
                ], color="info")
            ]
        
        return content, None, {'display': 'none'}
        
    except Exception as e:
        error_content = html.Div([
            html.H5("❌ Error occurred during intraday screening"),
            html.P(str(e), className="text-danger")
        ])
        return error_content, None, {'display': 'none'}

@app.callback(
    [Output("options-results", "children"),
     Output("options-progress", "children"),
     Output("options-progress", "style")],
    Input("btn-options", "n_clicks"),
    prevent_initial_call=True
)
def run_options_analysis(n_clicks):
    if n_clicks is None:
        return "Click 'Run Options Analysis' to start...", None, {'display': 'none'}
    
    try:
        # Show progress indicator
        progress = dbc.Alert([
            dbc.Spinner(size="sm", spinner_class_name="me-2"),
            "🔄 Running Options Analysis... Please wait while we analyze options data."
        ], color="info", className="mb-3")
        # Import the unified options analyzer
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        
        from src.screening.fixed_enhanced_options_analyzer import get_latest_analysis
        
        content = [
            html.H5("🎯 Options Analysis Results"),
            html.Br()
        ]
        
        # Analyze both indices using unified analyzer
        indices = ['NIFTY', 'BANKNIFTY']
        
        for index in indices:
            try:
                # Get analysis using unified analyzer
                result = get_latest_analysis(index)
                
                if result:
                    # Display results
                    content.extend([
                        html.H6(f"📊 {index} Analysis"),
                        html.Div([
                            html.Strong(f"Spot Price: ₹{result['current_price']:,.2f}"),
                            html.Br(),
                            html.Strong(f"ATM Strike: ₹{result['atm_strike']:,.2f}"),
                            html.Br(),
                            html.Strong(f"PCR: {result['pcr']:.2f}"),
                            html.Br(),
                            html.Strong(f"Signal: {result['signal']} ({result['confidence']:.1f}% confidence)"),
                            html.Br(),
                            html.Strong(f"Trade: {result['suggested_trade']}"),
                            html.Br(),
                            html.Small(f"Call OI: {result['atm_call_oi']:,.0f} | Put OI: {result['atm_put_oi']:,.0f}"),
                            html.Br(),
                            html.Small(f"Call ΔOI: {result['atm_call_oi_change']:+,.0f} | Put ΔOI: {result['atm_put_oi_change']:+,.0f}"),
                            html.Br(),
                            html.Small(f"Support: ₹{result['support']:,.2f} | Resistance: ₹{result['resistance']:,.2f}"),
                            html.Br(),
                            html.Small(f"Timestamp: {result['timestamp']}"),
                            html.Hr()
                        ])
                    ])
                else:
                    content.extend([
                        html.H6(f"❌ {index} Analysis Failed"),
                        html.P(f"Could not fetch {index} options data"),
                        html.Hr()
                    ])
                    
            except Exception as e:
                content.extend([
                    html.H6(f"❌ {index} Analysis Error"),
                    html.P(str(e), className="text-danger"),
                    html.Hr()
                ])
        
        return content, None, {'display': 'none'}
        
    except Exception as e:
        error_content = html.Div([
            html.H5("❌ Error occurred during options analysis"),
            html.P(str(e), className="text-danger")
        ])
        return error_content, None, {'display': 'none'}

@app.callback(
    [Output("predictions-results", "children"),
     Output("predictions-progress", "children"),
     Output("predictions-progress", "style")],
    Input("btn-predictions", "n_clicks"),
    prevent_initial_call=True
)
def run_market_predictions(n_clicks):
    if n_clicks is None:
        return "Click 'Run Market Predictions' to start...", None, {'display': 'none'}
    
    try:
        # Show progress indicator
        progress = dbc.Alert([
            dbc.Spinner(size="sm", spinner_class_name="me-2"),
            "🔄 Running Market Predictions... Please wait while we analyze ML models."
        ], color="info", className="mb-3")
        # Import enhanced options ML integration
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        
        from src.ml.enhanced_options_ml_integration import EnhancedOptionsMLIntegration
        
        # Initialize enhanced options ML integration
        options_ml = EnhancedOptionsMLIntegration()
        
        # Get options signals
        options_signals = options_ml.get_options_signals(['NIFTY', 'BANKNIFTY'])
        
        # Enhanced ML integration now uses real models instead of sample data
        
        content = [
            html.H5("🔮 ML + Options Market Predictions"),
            html.Br()
        ]
        
        if options_signals:
            # Get enhanced market sentiment
            sentiment_analysis = options_ml.get_enhanced_market_sentiment_score(options_signals)
            
            content.extend([
                html.H6("📊 Enhanced Market Sentiment Analysis:"),
                html.P(f"Overall Sentiment Score: {sentiment_analysis.get('sentiment_score', sentiment_analysis.get('overall_score', 0)):.3f}"),
                html.P(f"Sentiment: {'🟢 Bullish' if sentiment_analysis.get('sentiment', 'NEUTRAL') == 'BULLISH' else '🔴 Bearish' if sentiment_analysis.get('sentiment', 'NEUTRAL') == 'BEARISH' else '🟡 Neutral'}"),
                html.P(f"Confidence: {sentiment_analysis.get('confidence', 50):.1f}%"),
            ])
            
            # Show conflicts if any
            if sentiment_analysis.get('conflicts'):
                content.extend([
                    html.P("⚠️ Conflicts Detected:", className="text-warning"),
                    html.Ul([html.Li(conflict) for conflict in sentiment_analysis['conflicts']]),
                ])
            
            content.append(html.Hr())
            
            # Show enhanced options signals with ML predictions
            content.append(html.H6("🎯 Enhanced Options + ML Signals:"))
            for index, signal in options_signals.items():
                ml_pred = signal.get('ml_prediction', {})
                content.extend([
                    html.P(f"{index}: {signal['signal']} ({signal['confidence']:.1f}% confidence)"),
                    html.P(f"PCR: {signal['pcr']:.2f} | Signal Strength: {signal['signal_strength']:.3f}"),
                    html.P(f"ML Prediction: {ml_pred.get('direction', 'N/A')} ({ml_pred.get('confidence', 0):.1f}% confidence)"),
                    html.P(f"ML Model: {ml_pred.get('model_type', 'N/A')} | Prediction: {ml_pred.get('prediction', 0):.4f}"),
                    html.Br()
                ])
            
            # Get enhanced recommendations
            enhanced_recommendations = options_ml.get_enhanced_recommendations(options_signals)
            
            # Show enhanced recommendations
            content.append(html.H6("🤖 Enhanced ML + Options Recommendations:"))
            for rec in enhanced_recommendations.get('recommendations', []):
                content.extend([
                    html.P(f"{rec['index']}: {rec['recommendation']} ({rec['confidence']:.1f}% confidence)"),
                    html.P(f"Reason: {rec['reason']}"),
                    html.P(f"Options Signal: {rec['options_signal']} | ML Direction: {rec['ml_direction']}"),
                    html.Br()
                ])
        else:
            content.extend([
                html.Div([
                    html.H6("💡 No Options Data Available", className="text-warning"),
                    html.P("Options data is required for ML + Options integration."),
                    html.P("✅ ML predictions attempted successfully!"),
                    html.Hr()
                ], className="alert alert-warning")
            ])
        
        return content, None, {'display': 'none'}
        
    except Exception as e:
        error_content = html.Div([
            html.H5("❌ Error occurred during market predictions"),
            html.P(str(e), className="text-danger")
        ])
        return error_content, None, {'display': 'none'}

@app.callback(
    [Output("comprehensive-results", "children"),
     Output("comprehensive-progress", "children"),
     Output("comprehensive-progress", "style")],
    Input("btn-comprehensive", "n_clicks"),
    prevent_initial_call=True
)
def run_comprehensive_screening(n_clicks):
    if n_clicks is None:
        return "Click 'Run Comprehensive Screening' to start...", None, {'display': 'none'}
    
    try:
        # Show progress indicator
        progress = dbc.Alert([
            dbc.Spinner(size="sm", spinner_class_name="me-2"),
            "🔄 Running Comprehensive Screening... Please wait while we analyze all market data."
        ], color="info", className="mb-3")
        # Run comprehensive screening on ALL stocks from database
        results = screening_manager.run_comprehensive_screening(
            stock_list=None,  # None means use all stocks from database
            include_options=True,
            include_predictions=True
        )
        
        # Generate recommendations
        recommendations = screening_manager.generate_trading_recommendations(results)
        
        content = [
            html.H5("🚀 Comprehensive Analysis Results"),
            html.Br(),
            html.H6("📊 Summary:"),
            html.P(f"Total Stocks Analyzed: {results['summary']['total_stocks']}"),
            html.P(f"📈 EOD Signals: {results['summary']['eod_signals']}"),
            html.P(f"⚡ Intraday Signals: {results['summary']['intraday_signals']}"),
            html.P(f"🎯 Options Analysis: {results['summary']['options_analysis_count']}"),
            html.P(f"🔮 Predictions: {results['summary']['predictions_count']}"),
            html.Hr()
        ]
        
        # Show EOD signals if available
        eod_results = results.get('stock_screening', {}).get('eod', {})
        if eod_results.get('bullish_signals'):
            content.extend([
                html.H6("📈 Top EOD Bullish Signals:"),
                html.Ul([
                    html.Li(f"{signal.get('symbol', signal.get('ticker', 'Unknown'))} - {signal['confidence']}% confidence - Entry: ₹{signal['entry_price']:.2f}")
                    for signal in eod_results['bullish_signals'][:5]
                ]),
                html.Hr()
            ])
        
        if eod_results.get('bearish_signals'):
            content.extend([
                html.H6("📉 Top EOD Bearish Signals:"),
                html.Ul([
                    html.Li(f"{signal.get('symbol', signal.get('ticker', 'Unknown'))} - {signal['confidence']}% confidence - Entry: ₹{signal['entry_price']:.2f}")
                    for signal in eod_results['bearish_signals'][:5]
                ]),
                html.Hr()
            ])
        
        # Show message if no EOD signals found
        if not eod_results.get('bullish_signals') and not eod_results.get('bearish_signals'):
            content.extend([
                html.Div([
                    html.H6("💡 No EOD Signals Found", className="text-info"),
                    html.P("The comprehensive screening analyzed all stocks but found no signals with current criteria."),
                    html.P("This is normal - signals are generated only when specific technical conditions are met."),
                    html.Hr()
                ], className="alert alert-info")
            ])
        
        # High confidence signals
        if recommendations.get('high_confidence_signals'):
            content.extend([
                html.H6("🎯 High Confidence Signals:"),
                html.Ul([
                    html.Li(f"{signal['ticker']} - {signal['signal']} - {signal['confidence']}% confidence")
                    for signal in recommendations['high_confidence_signals'][:5]
                ]),
                html.Hr()
            ])
        
        # Market overview
        market_overview = recommendations.get('market_overview', {})
        content.extend([
            html.H6("Market Overview:"),
            html.P(f"Nifty: {market_overview.get('nifty_direction', 'NEUTRAL')} - {market_overview.get('nifty_confidence', 0)}% confidence"),
            html.P(f"BankNifty: {market_overview.get('banknifty_direction', 'NEUTRAL')} - {market_overview.get('banknifty_confidence', 0)}% confidence"),
            html.Hr()
        ])
        
        # Action items
        action_items = recommendations.get('action_items', [])
        if action_items:
            content.extend([
                html.H6("Action Items:"),
                html.Ul([
                    html.Li(item) for item in action_items[:3]
                ])
            ])
        
        return content, None, {'display': 'none'}
        
    except Exception as e:
        error_content = html.Div([
            html.H5("❌ Error occurred during comprehensive screening"),
            html.P(f"Error: {str(e)}", className="text-danger"),
            html.P("This might be due to:", className="text-muted"),
            html.Ul([
                html.Li("Database connection issues"),
                html.Li("Insufficient data for analysis"),
                html.Li("Network connectivity problems"),
                html.Li("System resource limitations")
            ], className="text-muted"),
            html.Hr(),
            html.P("💡 Try running individual screenings (EOD, Intraday) instead.", className="text-info")
        ])
        return error_content, None, {'display': 'none'}



# Search callback for EOD results
@app.callback(
    [Output("bullish-signals", "children"),
     Output("bearish-signals", "children")],
    [Input("eod-search", "value"),
     Input("eod-filter", "value")],
    prevent_initial_call=True
)
def filter_eod_results(search_term, filter_type):
    """Filter EOD results based on search term and filter type."""
    try:
        global current_eod_results
        
        # If no results available, show message
        if not current_eod_results:
            return [
                html.P("💡 Run EOD Screening first to enable search functionality", className="text-info"),
                html.P("💡 Run EOD Screening first to enable search functionality", className="text-info")
            ]
        
        # Get original signals
        original_bullish = current_eod_results.get('bullish_signals', [])
        original_bearish = current_eod_results.get('bearish_signals', [])
        
        # Filter by search term if provided
        if search_term and search_term.strip():
            search_term = search_term.strip().upper()
            
            # Filter bullish signals
            filtered_bullish = [
                signal for signal in original_bullish 
                if search_term in signal.get('symbol', '').upper()
            ]
            
            # Filter bearish signals
            filtered_bearish = [
                signal for signal in original_bearish 
                if search_term in signal.get('symbol', '').upper()
            ]
        else:
            filtered_bullish = original_bullish
            filtered_bearish = original_bearish
        
        # Apply additional filter type
        if filter_type == 'bullish':
            filtered_bearish = []
        elif filter_type == 'bearish':
            filtered_bullish = []
        elif filter_type == 'high_confidence':
            filtered_bullish = [s for s in filtered_bullish if s.get('confidence', 0) > 80]
            filtered_bearish = [s for s in filtered_bearish if s.get('confidence', 0) > 80]
        
        # Create filtered results
        bullish_results = create_signal_list(filtered_bullish, 'bullish')
        bearish_results = create_signal_list(filtered_bearish, 'bearish')
        
        # Add search info if search term was used
        if search_term and search_term.strip():
            search_info = html.Div([
                html.P(f"🔍 Search results for '{search_term}':", className="text-info"),
                html.P(f"📈 Bullish: {len(filtered_bullish)} | 📉 Bearish: {len(filtered_bearish)}", className="text-muted")
            ], className="mb-2")
            
            if isinstance(bullish_results, list):
                bullish_results.insert(0, search_info)
            else:
                bullish_results = [search_info, bullish_results]
            
            if isinstance(bearish_results, list):
                bearish_results.insert(0, search_info)
            else:
                bearish_results = [search_info, bearish_results]
        
        return bullish_results, bearish_results
            
    except Exception as e:
        return [
            html.P(f"Search error: {str(e)}", className="text-danger"),
            html.P(f"Search error: {str(e)}", className="text-danger")
        ]

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050) 