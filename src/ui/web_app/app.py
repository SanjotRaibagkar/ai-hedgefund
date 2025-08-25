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
                            dbc.Button("Run Intraday Screening", id="btn-intraday", color="success", className="me-2"),
                            dbc.Button("Run Options Analysis", id="btn-options", color="warning", className="me-2"),
                            dbc.Button("Run Market Predictions", id="btn-predictions", color="info", className="me-2"),
                            dbc.Button("Run Comprehensive Screening", id="btn-comprehensive", color="dark")
                        ], width=12)
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Stock List (comma-separated):"),
                            dcc.Textarea(
                                id="stock-list",
                                value="RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS",
                                style={'width': '100%', 'height': 60}
                            )
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

# Callbacks
@app.callback(
    Output("eod-results", "children"),
    Input("btn-eod", "n_clicks"),
    Input("stock-list", "value"),
    Input("risk-reward-slider", "value"),
    Input("analysis-mode", "value"),
    prevent_initial_call=True
)
def run_eod_screening(n_clicks, stock_list, risk_reward, analysis_mode):
    if n_clicks is None:
        return "Click 'Run EOD Screening' to start..."
    
    try:
        # Parse stock list
        stocks = [s.strip() for s in stock_list.split(",") if s.strip()]
        
        # Run screening with analysis mode
        results = screening_manager.get_eod_signals(stocks, risk_reward, analysis_mode)
        
        # Create results display
        bullish_count = results['summary']['bullish_count']
        bearish_count = results['summary']['bearish_count']
        total_stocks = results['summary']['total_stocks']
        
        content = [
            html.H5("📊 EOD Screening Results"),
            html.P(f"Analysis Mode: {analysis_mode.title()}"),
            html.P(f"Total Stocks Analyzed: {total_stocks}"),
            html.P(f"Bullish Signals: {bullish_count} | Bearish Signals: {bearish_count}"),
            html.Br()
        ]
        
        # Show message if no signals found
        if bullish_count == 0 and bearish_count == 0:
            content.extend([
                html.Div([
                    html.H6("💡 No EOD Signals Found", className="text-info"),
                    html.P("This is normal with limited data. Historical data would generate more signals."),
                    html.P("✅ Analysis completed successfully!"),
                    html.Hr()
                ], className="alert alert-info")
            ])
        else:
            # Show top signals
            if results['bullish_signals']:
                content.append(html.H6("Top Bullish Signals:"))
                for signal in results['bullish_signals'][:3]:
                    content.append(html.Div([
                        html.Strong(f"{signal['ticker']} - {signal['confidence']}% confidence"),
                        html.Br(),
                        html.Small(f"Entry: ₹{signal['entry_price']:.2f} | SL: ₹{signal['stop_loss']:.2f} | T1: ₹{signal['targets']['T1']:.2f}"),
                        html.Br(),
                        html.Small(f"Risk-Reward: {signal['risk_reward_ratio']:.2f}"),
                        html.Hr()
                    ]))
            
            if results['bearish_signals']:
                content.append(html.H6("Top Bearish Signals:"))
                for signal in results['bearish_signals'][:3]:
                    content.append(html.Div([
                        html.Strong(f"{signal['ticker']} - {signal['confidence']}% confidence"),
                        html.Br(),
                        html.Small(f"Entry: ₹{signal['entry_price']:.2f} | SL: ₹{signal['stop_loss']:.2f} | T1: ₹{signal['targets']['T1']:.2f}"),
                        html.Br(),
                        html.Small(f"Risk-Reward: {signal['risk_reward_ratio']:.2f}"),
                        html.Hr()
                    ]))
        
        return content
        
    except Exception as e:
        return html.Div([
            html.H5("❌ Error occurred during EOD screening"),
            html.P(str(e), className="text-danger")
        ])

@app.callback(
    Output("intraday-results", "children"),
    Input("btn-intraday", "n_clicks"),
    Input("stock-list", "value"),
    prevent_initial_call=True
)
def run_intraday_screening(n_clicks, stock_list):
    if n_clicks is None:
        return "Click 'Run Intraday Screening' to start..."
    
    try:
        # Parse stock list
        stocks = [s.strip() for s in stock_list.split(",") if s.strip()]
        
        # Run screening
        results = screening_manager.get_intraday_signals(stocks)
        
        # Create results display
        breakout_count = results['summary']['breakout_count']
        reversal_count = results['summary']['reversal_count']
        momentum_count = results['summary']['momentum_count']
        total_stocks = results['summary']['total_stocks']
        
        content = [
            html.H5("⚡ Intraday Screening Results"),
            html.P(f"Total Stocks Analyzed: {total_stocks}"),
            html.P(f"Breakout Signals: {breakout_count} | Reversal Signals: {reversal_count} | Momentum Signals: {momentum_count}"),
            html.Br()
        ]
        
        # Show message if no signals found
        if breakout_count == 0 and reversal_count == 0 and momentum_count == 0:
            content.extend([
                html.Div([
                    html.H6("💡 No Intraday Signals Found", className="text-info"),
                    html.P("This is normal with limited data. Real-time data feeds would generate more signals."),
                    html.P("✅ Analysis completed successfully!"),
                    html.Hr()
                ], className="alert alert-info")
            ])
        else:
            # Show top signals
            all_signals = (results['breakout_signals'] + 
                          results['reversal_signals'] + 
                          results['momentum_signals'])
            
            if all_signals:
                content.append(html.H6("Top Signals:"))
                for signal in all_signals[:5]:
                    content.append(html.Div([
                        html.Strong(f"{signal['ticker']} - {signal['signal_type']} - {signal['confidence']}%"),
                        html.Br(),
                        html.Small(f"Entry: ₹{signal['entry_price']:.2f} | SL: ₹{signal['stop_loss']:.2f}"),
                        html.Br(),
                        html.Small(f"Risk-Reward: {signal['risk_reward_ratio']:.2f}"),
                        html.Hr()
                    ]))
        
        return content
        
    except Exception as e:
        return html.Div([
            html.H5("❌ Error occurred during intraday screening"),
            html.P(str(e), className="text-danger")
        ])

@app.callback(
    Output("options-results", "children"),
    Input("btn-options", "n_clicks"),
    prevent_initial_call=True
)
def run_options_analysis(n_clicks):
    if n_clicks is None:
        return "Click 'Run Options Analysis' to start..."
    
    try:
        # Run analysis for both indices
        nifty_results = screening_manager.get_options_analysis('NIFTY')
        banknifty_results = screening_manager.get_options_analysis('BANKNIFTY')
        
        content = [
            html.H5("🎯 Options Analysis Results"),
            html.Br()
        ]
        
        # Check if any results are available
        if not nifty_results and not banknifty_results:
            content.extend([
                html.Div([
                    html.H6("💡 No Options Data Available", className="text-warning"),
                    html.P("NIFTY and BANKNIFTY options data requires special market access."),
                    html.P("✅ Analysis attempted successfully!"),
                    html.Hr()
                ], className="alert alert-warning")
            ])
        else:
            # Nifty analysis
            if nifty_results:
                nifty_analysis = nifty_results.get('analysis', {})
                oi_analysis = nifty_analysis.get('oi_analysis', {})
                sentiment = nifty_analysis.get('market_sentiment', {})
                
                content.extend([
                    html.H6("NIFTY Analysis:"),
                    html.P(f"Current Price: ₹{nifty_results.get('current_price', 0):.2f}"),
                    html.P(f"PCR: {oi_analysis.get('put_call_ratio', 0):.2f}"),
                    html.P(f"Sentiment: {sentiment.get('overall_sentiment', 'NEUTRAL')}"),
                    html.Hr()
                ])
            else:
                content.extend([
                    html.H6("NIFTY Analysis:"),
                    html.P("⚠️ No data available - requires special market access"),
                    html.Hr()
                ])
            
            # BankNifty analysis
            if banknifty_results:
                banknifty_analysis = banknifty_results.get('analysis', {})
                oi_analysis = banknifty_analysis.get('oi_analysis', {})
                sentiment = banknifty_analysis.get('market_sentiment', {})
                
                content.extend([
                    html.H6("BANKNIFTY Analysis:"),
                    html.P(f"Current Price: ₹{banknifty_results.get('current_price', 0):.2f}"),
                    html.P(f"PCR: {oi_analysis.get('put_call_ratio', 0):.2f}"),
                    html.P(f"Sentiment: {sentiment.get('overall_sentiment', 'NEUTRAL')}"),
                    html.Hr()
                ])
            else:
                content.extend([
                    html.H6("BANKNIFTY Analysis:"),
                    html.P("⚠️ No data available - requires special market access"),
                    html.Hr()
                ])
        
        return content
        
    except Exception as e:
        return html.Div([
            html.H5("❌ Error occurred during options analysis"),
            html.P(str(e), className="text-danger")
        ])

@app.callback(
    Output("predictions-results", "children"),
    Input("btn-predictions", "n_clicks"),
    prevent_initial_call=True
)
def run_market_predictions(n_clicks):
    if n_clicks is None:
        return "Click 'Run Market Predictions' to start..."
    
    try:
        content = [
            html.H5("Market Predictions"),
            html.Br()
        ]
        
        # Get predictions for different timeframes
        timeframes = ['15min', '1hour', 'eod', 'multiday']
        
        for timeframe in timeframes:
            nifty_pred = screening_manager.get_market_prediction('NIFTY', timeframe)
            banknifty_pred = screening_manager.get_market_prediction('BANKNIFTY', timeframe)
            
            if nifty_pred and nifty_pred.get('prediction'):
                pred_data = nifty_pred['prediction']
                content.extend([
                    html.H6(f"{timeframe.upper()} Predictions:"),
                    html.P(f"NIFTY: {pred_data.get('direction', 'NEUTRAL')} - {pred_data.get('confidence', 0)}% confidence"),
                ])
            
            if banknifty_pred and banknifty_pred.get('prediction'):
                pred_data = banknifty_pred['prediction']
                content.append(html.P(f"BANKNIFTY: {pred_data.get('direction', 'NEUTRAL')} - {pred_data.get('confidence', 0)}% confidence"))
            
            content.append(html.Hr())
        
        return content
        
    except Exception as e:
        return html.Div([
            html.H5("Error occurred during predictions"),
            html.P(str(e), className="text-danger")
        ])

@app.callback(
    Output("comprehensive-results", "children"),
    Input("btn-comprehensive", "n_clicks"),
    Input("stock-list", "value"),
    prevent_initial_call=True
)
def run_comprehensive_screening(n_clicks, stock_list):
    if n_clicks is None:
        return "Click 'Run Comprehensive Screening' to start..."
    
    try:
        # Parse stock list
        stocks = [s.strip() for s in stock_list.split(",") if s.strip()]
        
        # Run comprehensive screening
        results = screening_manager.run_comprehensive_screening(
            stock_list=stocks,
            include_options=True,
            include_predictions=True
        )
        
        # Generate recommendations
        recommendations = screening_manager.generate_trading_recommendations(results)
        
        content = [
            html.H5("Comprehensive Analysis Results"),
            html.Br(),
            html.H6("Summary:"),
            html.P(f"Total Stocks: {results['summary']['total_stocks']}"),
            html.P(f"EOD Signals: {results['summary']['eod_signals']}"),
            html.P(f"Intraday Signals: {results['summary']['intraday_signals']}"),
            html.P(f"Options Analysis: {results['summary']['options_analysis_count']}"),
            html.P(f"Predictions: {results['summary']['predictions_count']}"),
            html.Hr()
        ]
        
        # High confidence signals
        if recommendations.get('high_confidence_signals'):
            content.extend([
                html.H6("High Confidence Signals:"),
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
        
        return content
        
    except Exception as e:
        return html.Div([
            html.H5("Error occurred during comprehensive screening"),
            html.P(str(e), className="text-danger")
        ])

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050) 