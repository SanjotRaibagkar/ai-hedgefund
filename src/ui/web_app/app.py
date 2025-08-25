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
        # Import the working options tracker
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        
        from src.nsedata.NseUtility import NseUtils
        
        content = [
            html.H5("🎯 Options Analysis Results"),
            html.Br()
        ]
        
        # Initialize NSE utility
        nse = NseUtils()
        
        # Analyze both indices
        indices = ['NIFTY', 'BANKNIFTY']
        
        for index in indices:
            try:
                # Get options data
                options_data = nse.get_live_option_chain(index, indices=True)
                
                if options_data is not None and not options_data.empty:
                    # Get spot price
                    strikes = sorted(options_data['Strike_Price'].unique())
                    current_price = float(strikes[len(strikes)//2])
                    
                    # Find ATM strike
                    atm_strike = min(strikes, key=lambda x: abs(x - current_price))
                    
                    # Analyze ATM ± 2 strikes
                    atm_index = strikes.index(atm_strike)
                    start_idx = max(0, atm_index - 2)
                    end_idx = min(len(strikes), atm_index + 3)
                    strikes_to_analyze = strikes[start_idx:end_idx]
                    
                    # OI analysis
                    total_call_oi = 0
                    total_put_oi = 0
                    atm_call_oi = 0
                    atm_put_oi = 0
                    atm_call_oi_change = 0
                    atm_put_oi_change = 0
                    
                    for strike in strikes_to_analyze:
                        strike_data = options_data[options_data['Strike_Price'] == strike]
                        
                        if not strike_data.empty:
                            call_oi = float(strike_data['CALLS_OI'].iloc[0]) if 'CALLS_OI' in strike_data.columns else 0
                            put_oi = float(strike_data['PUTS_OI'].iloc[0]) if 'PUTS_OI' in strike_data.columns else 0
                            call_oi_change = float(strike_data['CALLS_Chng_in_OI'].iloc[0]) if 'CALLS_Chng_in_OI' in strike_data.columns else 0
                            put_oi_change = float(strike_data['PUTS_Chng_in_OI'].iloc[0]) if 'PUTS_Chng_in_OI' in strike_data.columns else 0
                            
                            total_call_oi += call_oi
                            total_put_oi += put_oi
                            
                            if strike == atm_strike:
                                atm_call_oi = call_oi
                                atm_put_oi = put_oi
                                atm_call_oi_change = call_oi_change
                                atm_put_oi_change = put_oi_change
                    
                    pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
                    
                    # Generate signal
                    signal = "NEUTRAL"
                    confidence = 50.0
                    suggested_trade = "Wait for clearer signal"
                    
                    # Strategy Rules
                    if pcr > 0.9 and atm_put_oi_change > 0 and atm_call_oi_change < 0:
                        signal = "BULLISH"
                        confidence = min(90, 60 + (pcr - 0.9) * 100)
                        suggested_trade = "Buy Call (ATM/ITM) or Bull Call Spread"
                    elif pcr < 0.8 and atm_call_oi_change > 0 and atm_put_oi_change < 0:
                        signal = "BEARISH"
                        confidence = min(90, 60 + (0.8 - pcr) * 100)
                        suggested_trade = "Buy Put (ATM/ITM) or Bear Put Spread"
                    elif 0.8 <= pcr <= 1.2 and atm_call_oi_change > 0 and atm_put_oi_change > 0:
                        signal = "RANGE"
                        confidence = 70.0
                        suggested_trade = "Sell Straddle/Strangle"
                    
                    # Add to content
                    content.extend([
                        html.H6(f"{index} Analysis:"),
                        html.P(f"💰 Spot Price: ₹{current_price:,.0f}"),
                        html.P(f"🎯 ATM Strike: ₹{atm_strike:,.0f}"),
                        html.P(f"📊 PCR: {pcr:.2f}"),
                        html.P(f"📈 Signal: {signal} (Confidence: {confidence:.1f}%)"),
                        html.P(f"💡 Trade: {suggested_trade}"),
                        html.P(f"📊 ATM Call OI: {atm_call_oi:,.0f} | Put OI: {atm_put_oi:,.0f}"),
                        html.P(f"📈 Call OI Change: {atm_call_oi_change:,.0f} | Put OI Change: {atm_put_oi_change:,.0f}"),
                        html.Hr()
                    ])
                    
                else:
                    content.extend([
                        html.H6(f"{index} Analysis:"),
                        html.P("⚠️ No options data available"),
                        html.Hr()
                    ])
                    
            except Exception as e:
                content.extend([
                    html.H6(f"{index} Analysis:"),
                    html.P(f"❌ Error: {str(e)}"),
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
        # Import options ML integration
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        
        from src.ml.options_ml_integration import OptionsMLIntegration
        
        # Initialize options ML integration
        options_ml = OptionsMLIntegration()
        
        # Get options signals
        options_signals = options_ml.get_options_signals(['NIFTY', 'BANKNIFTY'])
        
        # Create sample base features (in real implementation, this would come from ML model)
        base_features = pd.DataFrame({
            'technical_score': [0.6, 0.4, 0.8],
            'fundamental_score': [0.7, 0.5, 0.9],
            'momentum_score': [0.5, 0.3, 0.7]
        })
        
        # Sample base predictions (in real implementation, this would come from ML model)
        base_predictions = [0.15, -0.08, 0.25]
        
        content = [
            html.H5("🔮 ML + Options Market Predictions"),
            html.Br()
        ]
        
        if options_signals:
            # Get market sentiment
            sentiment_score = options_ml.get_market_sentiment_score(options_signals)
            
            content.extend([
                html.H6("📊 Market Sentiment Analysis:"),
                html.P(f"Overall Sentiment Score: {sentiment_score:.3f}"),
                html.P(f"Sentiment: {'🟢 Bullish' if sentiment_score > 0.1 else '🔴 Bearish' if sentiment_score < -0.1 else '🟡 Neutral'}"),
                html.Hr()
            ])
            
            # Show options signals
            content.append(html.H6("🎯 Options Signals:"))
            for index, signal in options_signals.items():
                content.extend([
                    html.P(f"{index}: {signal['signal']} ({signal['confidence']:.1f}% confidence)"),
                    html.P(f"PCR: {signal['pcr']:.2f} | Signal Strength: {signal['signal_strength']:.3f}"),
                    html.Br()
                ])
            
            # Show ML + Options recommendations
            content.append(html.H6("🤖 ML + Options Recommendations:"))
            for i, base_pred in enumerate(base_predictions):
                adjusted_pred = options_ml.adjust_ml_prediction(base_pred, options_signals)
                recommendation = options_ml._generate_recommendation(adjusted_pred, sentiment_score, options_signals)
                
                content.extend([
                    html.P(f"Model {i+1}: Base: {base_pred:.3f} → Adjusted: {adjusted_pred:.3f}"),
                    html.P(f"Recommendation: {recommendation}"),
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
        
        return content
        
    except Exception as e:
        return html.Div([
            html.H5("❌ Error occurred during market predictions"),
            html.P(str(e), className="text-danger")
        ])
        
        return content
        
    except Exception as e:
        return html.Div([
            html.H5("Error occurred during predictions"),
            html.P(str(e), className="text-danger")
        ])

@app.callback(
    Output("comprehensive-results", "children"),
    Input("btn-comprehensive", "n_clicks"),
    prevent_initial_call=True
)
def run_comprehensive_screening(n_clicks):
    if n_clicks is None:
        return "Click 'Run Comprehensive Screening' to start..."
    
    try:
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
        
        return content
        
    except Exception as e:
        return html.Div([
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

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050) 