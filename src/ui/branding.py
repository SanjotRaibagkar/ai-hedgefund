#!/usr/bin/env python3
"""
MokshTechandInvestment Branding
Company branding and logo information.
"""

# Company Information
COMPANY_NAME = "MokshTechandInvestment"
COMPANY_TAGLINE = "AI-Powered Investment Solutions"
COMPANY_DESCRIPTION = "Advanced stock screening and market analysis for intelligent trading decisions"
COMPANY_WEBSITE = "https://moksh-tech-investment.com"
COMPANY_EMAIL = "info@moksh-tech-investment.com"

# Logo ASCII Art
LOGO_ASCII = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║    🚀 MOKSHTECHANDINVESTMENT 🚀                              ║
    ║                                                              ║
    ║    AI-Powered Investment Solutions                           ║
    ║    Advanced Stock Screening & Market Analysis                ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
"""

# Color Scheme
COLORS = {
    'primary': '#1f77b4',      # Blue
    'secondary': '#ff7f0e',    # Orange
    'success': '#2ca02c',      # Green
    'warning': '#d62728',      # Red
    'info': '#9467bd',         # Purple
    'light': '#f8f9fa',        # Light Gray
    'dark': '#343a40'          # Dark Gray
}

# Product Information
PRODUCTS = {
    'stock_screener': {
        'name': 'AI Stock Screener',
        'description': 'Advanced stock screening with technical analysis',
        'features': [
            'EOD Trading Signals',
            'Intraday Opportunities',
            'Risk-Reward Analysis',
            'Entry/Exit Points'
        ]
    },
    'options_analyzer': {
        'name': 'Options Analysis',
        'description': 'Comprehensive options analysis for Nifty & BankNifty',
        'features': [
            'OI Pattern Analysis',
            'Volatility Analysis',
            'Strike Recommendations',
            'Market Sentiment'
        ]
    },
    'market_predictor': {
        'name': 'Market Predictor',
        'description': 'AI-powered market movement predictions',
        'features': [
            '15-min Predictions',
            '1-hour Forecasts',
            'EOD Expectations',
            'Multi-day Outlook'
        ]
    }
}

def get_branding_info():
    """Get complete branding information."""
    return {
        'company_name': COMPANY_NAME,
        'tagline': COMPANY_TAGLINE,
        'description': COMPANY_DESCRIPTION,
        'website': COMPANY_WEBSITE,
        'email': COMPANY_EMAIL,
        'logo': LOGO_ASCII,
        'colors': COLORS,
        'products': PRODUCTS
    }

def print_logo():
    """Print the company logo."""
    print(LOGO_ASCII)
    print(f"📧 {COMPANY_EMAIL}")
    print(f"🌐 {COMPANY_WEBSITE}")
    print("=" * 60) 