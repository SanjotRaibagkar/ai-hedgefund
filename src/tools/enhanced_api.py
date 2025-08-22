import datetime
import os
import pandas as pd
import requests
import time
import logging
from typing import List, Optional, Dict, Any

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    CompanyFactsResponse,
)
from src.data.providers.provider_factory import (
    get_provider_for_ticker,
    get_indian_news_provider,
    get_currency_service,
    get_market_calendar,
    get_mutual_fund_service,
    get_bond_service,
    get_commodity_service,
    get_forex_service,
    get_derivatives_service,
    get_corporate_actions_service,
    get_intraday_provider,
    get_market_data_provider,
    get_nse_utility_provider
)
from src.data.providers.base_provider import DataProviderError

# Global cache instance
_cache = get_cache()
logger = logging.getLogger(__name__)


def _is_indian_ticker(ticker: str) -> bool:
    """Check if the ticker is an Indian stock."""
    ticker = ticker.upper()
    return any(ticker.endswith(suffix) for suffix in ['.NS', '.BO', '.NSE', '.BSE'])


def _is_us_ticker(ticker: str) -> bool:
    """Check if the ticker is a US stock."""
    ticker = ticker.upper()
    # US tickers typically don't have suffixes or have specific ones
    return not _is_indian_ticker(ticker) and '.' not in ticker


def _get_preferred_provider(ticker: str, provider_type: str = "default") -> str:
    """
    Get preferred provider for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        provider_type: Type of provider ("default", "yahoo", "nse")
    
    Returns:
        Provider name to use
    """
    if provider_type == "yahoo":
        return "yahoo"
    elif provider_type == "nse":
        return "nse"
    else:
        # Default: Use NSEUtility for Indian stocks, Yahoo for US stocks
        return "nse" if _is_indian_ticker(ticker) else "yahoo"


def _make_api_request(url: str, headers: dict, method: str = "GET", json_data: dict = None, max_retries: int = 3) -> requests.Response:
    """
    Make an API request with rate limiting handling and moderate backoff.
    
    Args:
        url: The URL to request
        headers: Headers to include in the request
        method: HTTP method (GET or POST)
        json_data: JSON data for POST requests
        max_retries: Maximum number of retries (default: 3)
    
    Returns:
        requests.Response: The response object
    
    Raises:
        Exception: If the request fails with a non-429 error
    """
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        if method.upper() == "POST":
            response = requests.post(url, headers=headers, json=json_data)
        else:
            response = requests.get(url, headers=headers)
        
        if response.status_code == 429 and attempt < max_retries:
            # Linear backoff: 60s, 90s, 120s, 150s...
            delay = 60 + (30 * attempt)
            print(f"Rate limited (429). Attempt {attempt + 1}/{max_retries + 1}. Waiting {delay}s before retrying...")
            time.sleep(delay)
            continue
        
        # Return the response (whether success, other errors, or final 429)
        return response


def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> List[Price]:
    """Fetch price data from cache or appropriate data provider."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{start_date}_{end_date}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_prices(cache_key):
        return [Price(**price) for price in cached_data]

    # Determine data source based on ticker
    if _is_indian_ticker(ticker):
        # Use provider system for Indian stocks
        try:
            provider = get_provider_for_ticker(ticker)
            prices = provider.get_prices(ticker, start_date, end_date)
            
            if prices:
                # Cache the results
                _cache.set_prices(cache_key, [p.model_dump() for p in prices])
                return prices
            else:
                logger.warning(f"No price data found for Indian ticker: {ticker}")
                return []
                
        except DataProviderError as e:
            logger.error(f"Provider error for {ticker}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching prices for {ticker}: {str(e)}")
            return []
    
    else:
        # Use existing Financial Datasets API for US stocks
        try:
            headers = {}
            financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
            if financial_api_key:
                headers["X-API-KEY"] = financial_api_key

            url = f"https://api.financialdatasets.ai/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"
            response = _make_api_request(url, headers)
            if response.status_code != 200:
                raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

            # Parse response with Pydantic model
            price_response = PriceResponse(**response.json())
            prices = price_response.prices

            if not prices:
                return []

            # Cache the results using the comprehensive cache key
            _cache.set_prices(cache_key, [p.model_dump() for p in prices])
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching price data for {ticker}: {str(e)}")
            return []


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> List[FinancialMetrics]:
    """Fetch financial metrics from cache or appropriate data provider."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**metric) for metric in cached_data]

    # Determine data source based on ticker
    if _is_indian_ticker(ticker):
        # Use provider system for Indian stocks
        try:
            provider = get_provider_for_ticker(ticker)
            metrics = provider.get_financial_metrics(ticker, end_date, period, limit)
            
            if metrics:
                # Cache the results
                _cache.set_financial_metrics(cache_key, [m.model_dump() for m in metrics])
                return metrics
            else:
                logger.warning(f"No financial metrics found for Indian ticker: {ticker}")
                return []
                
        except DataProviderError as e:
            logger.error(f"Provider error for {ticker}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching metrics for {ticker}: {str(e)}")
            return []
    
    else:
        # Use existing Financial Datasets API for US stocks
        try:
            headers = {}
            financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
            if financial_api_key:
                headers["X-API-KEY"] = financial_api_key

            url = f"https://api.financialdatasets.ai/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}"
            response = _make_api_request(url, headers)
            if response.status_code != 200:
                raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

            # Parse response with Pydantic model
            metrics_response = FinancialMetricsResponse(**response.json())
            financial_metrics = metrics_response.financial_metrics

            if not financial_metrics:
                return []

            # Cache the results as dicts using the comprehensive cache key
            _cache.set_financial_metrics(cache_key, [m.model_dump() for m in financial_metrics])
            return financial_metrics
            
        except Exception as e:
            logger.error(f"Error fetching financial metrics for {ticker}: {str(e)}")
            return []


def get_line_items(
    ticker: str,
    end_date: str,
    line_items: List[str] = None,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> List[LineItem]:
    """Fetch line items from appropriate data provider."""
    # Default line items if none provided
    if line_items is None:
        line_items = ['revenue', 'net_income', 'total_assets', 'total_liabilities', 'total_equity']
    
    # Determine data source based on ticker
    if _is_indian_ticker(ticker):
        # Use provider system for Indian stocks
        try:
            provider = get_provider_for_ticker(ticker)
            line_items_data = provider.get_line_items(ticker, line_items, end_date, period, limit)
            return line_items_data
                
        except DataProviderError as e:
            logger.error(f"Provider error for {ticker}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching line items for {ticker}: {str(e)}")
            return []
    
    else:
        # Use existing Financial Datasets API for US stocks
        try:
            headers = {}
            financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
            if financial_api_key:
                headers["X-API-KEY"] = financial_api_key

            url = "https://api.financialdatasets.ai/financials/search/line-items"

            body = {
                "tickers": [ticker],
                "line_items": line_items,
                "end_date": end_date,
                "period": period,
                "limit": limit,
            }
            response = _make_api_request(url, headers, method="POST", json_data=body)
            if response.status_code != 200:
                raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")
            data = response.json()
            response_model = LineItemResponse(**data)
            search_results = response_model.search_results

            if not search_results:
                return []

            return search_results[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching line items for {ticker}: {str(e)}")
            return []


def get_market_cap(ticker: str, end_date: str, api_key: str = None) -> Optional[float]:
    """Fetch market cap from appropriate data provider."""
    # Determine data source based on ticker
    if _is_indian_ticker(ticker):
        # Use provider system for Indian stocks
        try:
            provider = get_provider_for_ticker(ticker)
            market_cap = provider.get_market_cap(ticker, end_date)
            return market_cap
                
        except DataProviderError as e:
            logger.error(f"Provider error for {ticker}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching market cap for {ticker}: {str(e)}")
            return None
    
    else:
        # Use existing Financial Datasets API for US stocks
        try:
            headers = {}
            financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
            if financial_api_key:
                headers["X-API-KEY"] = financial_api_key

            url = f"https://api.financialdatasets.ai/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit=1"
            response = _make_api_request(url, headers)
            if response.status_code != 200:
                raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

            data = response.json()
            metrics_response = FinancialMetricsResponse(**data)
            financial_metrics = metrics_response.financial_metrics

            if not financial_metrics:
                return None

            return financial_metrics[0].market_cap
            
        except Exception as e:
            logger.error(f"Error fetching market cap for {ticker}: {str(e)}")
            return None


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: Optional[str] = None,
    limit: int = 1000,
    api_key: str = None,
) -> List[CompanyNews]:
    """Fetch company news from appropriate data provider."""
    # Determine data source based on ticker
    if _is_indian_ticker(ticker):
        # Use provider system for Indian stocks
        try:
            provider = get_provider_for_ticker(ticker)
            news = provider.get_company_news(ticker, end_date, start_date, limit)
            return news
                
        except DataProviderError as e:
            logger.error(f"Provider error for {ticker}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching news for {ticker}: {str(e)}")
            return []
    
    else:
        # Use existing Financial Datasets API for US stocks
        try:
            headers = {}
            financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
            if financial_api_key:
                headers["X-API-KEY"] = financial_api_key

            all_news = []
            current_end_date = end_date

            while True:
                url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
                if start_date:
                    url += f"&start_date={start_date}"
                url += f"&limit={limit}"

                response = _make_api_request(url, headers)
                if response.status_code != 200:
                    raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

                data = response.json()
                response_model = CompanyNewsResponse(**data)
                company_news = response_model.news

                if not company_news:
                    break

                all_news.extend(company_news)

                # Only continue pagination if we have a start_date and got a full page
                if not start_date or len(company_news) < limit:
                    break

                # Update end_date to the oldest date from current batch for next iteration
                current_end_date = min(news.date for news in company_news).split("T")[0]

                # If we've reached or passed the start_date, we can stop
                if current_end_date <= start_date:
                    break

            if not all_news:
                return []

            return all_news
            
        except Exception as e:
            logger.error(f"Error fetching company news for {ticker}: {str(e)}")
            return []


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: Optional[str] = None,
    limit: int = 1000,
    api_key: str = None,
) -> List[InsiderTrade]:
    """Fetch insider trades from appropriate data provider."""
    # Determine data source based on ticker
    if _is_indian_ticker(ticker):
        # Use provider system for Indian stocks
        try:
            provider = get_provider_for_ticker(ticker)
            trades = provider.get_insider_trades(ticker, end_date, start_date, limit)
            return trades
                
        except DataProviderError as e:
            logger.error(f"Provider error for {ticker}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching insider trades for {ticker}: {str(e)}")
            return []
    
    else:
        # Use existing Financial Datasets API for US stocks
        try:
            headers = {}
            financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
            if financial_api_key:
                headers["X-API-KEY"] = financial_api_key

            all_trades = []
            current_end_date = end_date

            while True:
                url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
                if start_date:
                    url += f"&filing_date_gte={start_date}"
                url += f"&limit={limit}"

                response = _make_api_request(url, headers)
                if response.status_code != 200:
                    raise Exception(f"Error fetching data: {ticker} - {response.status_code} - {response.text}")

                data = response.json()
                response_model = InsiderTradeResponse(**data)
                insider_trades = response_model.insider_trades

                if not insider_trades:
                    break

                all_trades.extend(insider_trades)

                # Only continue pagination if we have a start_date and got a full page
                if not start_date or len(insider_trades) < limit:
                    break

                # Update end_date to the oldest filing date from current batch for next iteration
                current_end_date = min(trade.filing_date for trade in insider_trades).split("T")[0]

                # If we've reached or passed the start_date, we can stop
                if current_end_date <= start_date:
                    break

            if not all_trades:
                return []

            return all_trades
            
        except Exception as e:
            logger.error(f"Error fetching insider trades for {ticker}: {str(e)}")
            return []


# New Phase 2 Functions for Indian Market Specifics

def get_indian_market_status() -> Dict[str, Any]:
    """Get comprehensive Indian market status."""
    try:
        market_calendar = get_market_calendar()
        return market_calendar.get_market_status()
    except Exception as e:
        logger.error(f"Error fetching Indian market status: {str(e)}")
        return {}


def get_currency_rates() -> Dict[str, Any]:
    """Get current INR/USD currency exchange rates."""
    try:
        currency_service = get_currency_service()
        return currency_service.get_exchange_rate_info()
    except Exception as e:
        logger.error(f"Error fetching currency rates: {str(e)}")
        return {}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    """Convert currency amounts between INR and USD."""
    try:
        currency_service = get_currency_service()
        
        if from_currency == 'INR' and to_currency == 'USD':
            return currency_service.convert_inr_to_usd(amount)
        elif from_currency == 'USD' and to_currency == 'INR':
            return currency_service.convert_usd_to_inr(amount)
        else:
            logger.warning(f"Unsupported currency conversion: {from_currency} to {to_currency}")
            return amount
    except Exception as e:
        logger.error(f"Error converting currency: {str(e)}")
        return amount


def format_indian_currency(amount: float, currency: str = 'INR') -> str:
    """Format currency with Indian numbering system (lakhs, crores)."""
    try:
        currency_service = get_currency_service()
        return currency_service.format_currency(amount, currency)
    except Exception as e:
        logger.error(f"Error formatting currency: {str(e)}")
        return f"{amount:,.2f} {currency}"


def get_indian_news_aggregated(ticker: str, limit: int = 20) -> List[CompanyNews]:
    """Get aggregated news from Indian financial news sources."""
    try:
        if not _is_indian_ticker(ticker):
            logger.warning(f"get_indian_news_aggregated called with non-Indian ticker: {ticker}")
            return []
        
        news_provider = get_indian_news_provider()
        return news_provider.get_aggregated_news(ticker, limit=limit)
    except Exception as e:
        logger.error(f"Error fetching Indian news for {ticker}: {str(e)}")
        return []


def get_indian_market_sentiment(ticker: Optional[str] = None) -> Dict[str, Any]:
    """Get Indian market sentiment analysis."""
    try:
        news_provider = get_indian_news_provider()
        return news_provider.get_market_sentiment(ticker)
    except Exception as e:
        logger.error(f"Error analyzing Indian market sentiment: {str(e)}")
        return {'sentiment': 'neutral', 'confidence': 0.0}


def get_sector_performance() -> Dict[str, Any]:
    """Get Indian sector performance data."""
    try:
        # Try to get from NSE provider
        provider = get_provider_for_ticker('NIFTY.NS')
        if hasattr(provider, 'get_sector_performance'):
            return provider.get_sector_performance()
        else:
            logger.warning("Sector performance not available from current provider")
            return {}
    except Exception as e:
        logger.error(f"Error fetching sector performance: {str(e)}")
        return {}


def get_top_movers() -> Dict[str, List[Dict]]:
    """Get top gainers and losers from Indian market."""
    try:
        # Try to get from NSE provider
        provider = get_provider_for_ticker('NIFTY.NS')
        if hasattr(provider, 'get_top_gainers_losers'):
            return provider.get_top_gainers_losers()
        else:
            logger.warning("Top movers data not available from current provider")
            return {'gainers': [], 'losers': []}
    except Exception as e:
        logger.error(f"Error fetching top movers: {str(e)}")
        return {'gainers': [], 'losers': []}


def is_indian_market_open() -> bool:
    """Check if Indian market is currently open."""
    try:
        market_calendar = get_market_calendar()
        return market_calendar.is_market_open()
    except Exception as e:
        logger.error(f"Error checking if Indian market is open: {str(e)}")
        return False


def get_market_timings() -> Dict[str, str]:
    """Get Indian market timing information."""
    try:
        market_calendar = get_market_calendar()
        return market_calendar.get_market_timings()
    except Exception as e:
        logger.error(f"Error fetching market timings: {str(e)}")
        return {}


def normalize_market_cap_to_usd(market_cap: float, currency: str = 'INR') -> float:
    """Normalize market cap to USD for comparison."""
    try:
        if currency == 'USD':
            return market_cap
        
        currency_service = get_currency_service()
        return currency_service.normalize_market_cap(market_cap, currency, 'USD')
    except Exception as e:
        logger.error(f"Error normalizing market cap: {str(e)}")
        return market_cap


def prices_to_df(prices: List[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# ============================================================================
# PHASE 3: ADVANCED FEATURES - MUTUAL FUNDS
# ============================================================================

def get_top_mutual_funds(category: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
    """Get top performing mutual funds by category."""
    try:
        mutual_fund_service = get_mutual_fund_service()
        funds = mutual_fund_service.get_top_performing_funds(category, limit)
        return [fund.__dict__ for fund in funds]
    except Exception as e:
        logger.error(f"Error fetching top mutual funds: {str(e)}")
        return []


def get_mutual_fund_details(fund_code: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific mutual fund."""
    try:
        mutual_fund_service = get_mutual_fund_service()
        fund = mutual_fund_service.get_fund_details(fund_code)
        return fund.__dict__ if fund else None
    except Exception as e:
        logger.error(f"Error fetching mutual fund details: {str(e)}")
        return None


def get_mutual_fund_holdings(fund_code: str) -> List[Dict[str, Any]]:
    """Get portfolio holdings for a specific mutual fund."""
    try:
        mutual_fund_service = get_mutual_fund_service()
        holdings = mutual_fund_service.get_fund_holdings(fund_code)
        return [holding.__dict__ for holding in holdings]
    except Exception as e:
        logger.error(f"Error fetching mutual fund holdings: {str(e)}")
        return []


def search_mutual_funds(query: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
    """Search mutual funds by name or fund house."""
    try:
        mutual_fund_service = get_mutual_fund_service()
        funds = mutual_fund_service.search_funds(query, category)
        return [fund.__dict__ for fund in funds]
    except Exception as e:
        logger.error(f"Error searching mutual funds: {str(e)}")
        return []


# ============================================================================
# PHASE 3: ADVANCED FEATURES - BONDS
# ============================================================================

def get_government_bonds(limit: int = 20) -> List[Dict[str, Any]]:
    """Get list of government bonds."""
    try:
        bond_service = get_bond_service()
        bonds = bond_service.get_government_bonds(limit)
        return [bond.__dict__ for bond in bonds]
    except Exception as e:
        logger.error(f"Error fetching government bonds: {str(e)}")
        return []


def get_corporate_bonds(rating: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """Get list of corporate bonds."""
    try:
        bond_service = get_bond_service()
        bonds = bond_service.get_corporate_bonds(rating, limit)
        return [bond.__dict__ for bond in bonds]
    except Exception as e:
        logger.error(f"Error fetching corporate bonds: {str(e)}")
        return []


def get_yield_curve() -> Dict[str, Any]:
    """Get current government bond yield curve."""
    try:
        bond_service = get_bond_service()
        yield_curve = bond_service.get_yield_curve()
        return yield_curve.__dict__ if yield_curve else {}
    except Exception as e:
        logger.error(f"Error fetching yield curve: {str(e)}")
        return {}


def get_bond_yields(bond_code: str, days: int = 30) -> List[Dict[str, Any]]:
    """Get historical yield data for a bond."""
    try:
        bond_service = get_bond_service()
        yields = bond_service.get_bond_yields(bond_code, days)
        return [yield_data.__dict__ for yield_data in yields]
    except Exception as e:
        logger.error(f"Error fetching bond yields: {str(e)}")
        return []


# ============================================================================
# PHASE 3: ADVANCED FEATURES - COMMODITIES
# ============================================================================

def get_commodity_price(symbol: str) -> Optional[Dict[str, Any]]:
    """Get current price for a specific commodity."""
    try:
        commodity_service = get_commodity_service()
        commodity = commodity_service.get_commodity_price(symbol)
        return commodity.__dict__ if commodity else None
    except Exception as e:
        logger.error(f"Error fetching commodity price: {str(e)}")
        return None


def get_all_commodities() -> List[Dict[str, Any]]:
    """Get current prices for all major commodities."""
    try:
        commodity_service = get_commodity_service()
        commodities = commodity_service.get_all_commodities()
        return [commodity.__dict__ for commodity in commodities]
    except Exception as e:
        logger.error(f"Error fetching all commodities: {str(e)}")
        return []


def get_commodity_prices(symbol: str, days: int = 30) -> List[Dict[str, Any]]:
    """Get historical price data for a commodity."""
    try:
        commodity_service = get_commodity_service()
        prices = commodity_service.get_commodity_prices(symbol, days)
        return [price.__dict__ for price in prices]
    except Exception as e:
        logger.error(f"Error fetching commodity prices: {str(e)}")
        return []


def get_commodity_futures(symbol: str) -> List[Dict[str, Any]]:
    """Get futures contracts for a commodity."""
    try:
        commodity_service = get_commodity_service()
        contracts = commodity_service.get_futures_contracts(symbol)
        return [contract.__dict__ for contract in contracts]
    except Exception as e:
        logger.error(f"Error fetching commodity futures: {str(e)}")
        return []


# ============================================================================
# PHASE 3: ADVANCED FEATURES - FOREX
# ============================================================================

def get_forex_rate(pair: str) -> Optional[Dict[str, Any]]:
    """Get current forex rate for a currency pair."""
    try:
        forex_service = get_forex_service()
        rate = forex_service.get_forex_rate(pair)
        return rate.__dict__ if rate else None
    except Exception as e:
        logger.error(f"Error fetching forex rate: {str(e)}")
        return None


def get_all_forex_rates() -> List[Dict[str, Any]]:
    """Get current rates for all major forex pairs."""
    try:
        forex_service = get_forex_service()
        rates = forex_service.get_all_forex_rates()
        return [rate.__dict__ for rate in rates]
    except Exception as e:
        logger.error(f"Error fetching all forex rates: {str(e)}")
        return []


def get_forex_prices(pair: str, days: int = 30) -> List[Dict[str, Any]]:
    """Get historical forex price data."""
    try:
        forex_service = get_forex_service()
        prices = forex_service.get_forex_prices(pair, days)
        return [price.__dict__ for price in prices]
    except Exception as e:
        logger.error(f"Error fetching forex prices: {str(e)}")
        return []


def get_cross_rates() -> List[Dict[str, Any]]:
    """Get cross currency rates."""
    try:
        forex_service = get_forex_service()
        rates = forex_service.get_cross_rates()
        return [rate.__dict__ for rate in rates]
    except Exception as e:
        logger.error(f"Error fetching cross rates: {str(e)}")
        return []


def convert_forex(amount: float, from_currency: str, to_currency: str) -> Optional[float]:
    """Convert amount from one currency to another."""
    try:
        forex_service = get_forex_service()
        return forex_service.convert_currency(amount, from_currency, to_currency)
    except Exception as e:
        logger.error(f"Error converting currency: {str(e)}")
        return None


# ============================================================================
# PHASE 3: ADVANCED FEATURES - DERIVATIVES
# ============================================================================

def get_option_chain(underlying: str, expiry_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Get complete option chain for an underlying."""
    try:
        derivatives_service = get_derivatives_service()
        option_chain = derivatives_service.get_option_chain(underlying, expiry_date)
        if option_chain:
            return {
                'underlying': option_chain.underlying,
                'expiry_date': option_chain.expiry_date,
                'spot_price': option_chain.spot_price,
                'call_options': [opt.__dict__ for opt in option_chain.call_options],
                'put_options': [opt.__dict__ for opt in option_chain.put_options],
                'timestamp': option_chain.timestamp
            }
        return None
    except Exception as e:
        logger.error(f"Error fetching option chain: {str(e)}")
        return None


def get_futures_contracts(underlying: str) -> List[Dict[str, Any]]:
    """Get futures contracts for an underlying."""
    try:
        derivatives_service = get_derivatives_service()
        contracts = derivatives_service.get_futures_contracts(underlying)
        return [contract.__dict__ for contract in contracts]
    except Exception as e:
        logger.error(f"Error fetching futures contracts: {str(e)}")
        return []


def get_option_contract(symbol: str) -> Optional[Dict[str, Any]]:
    """Get specific option contract details."""
    try:
        derivatives_service = get_derivatives_service()
        contract = derivatives_service.get_option_contract(symbol)
        return contract.__dict__ if contract else None
    except Exception as e:
        logger.error(f"Error fetching option contract: {str(e)}")
        return None


def get_iv_smile(underlying: str, expiry_date: str) -> Dict[str, Any]:
    """Get implied volatility smile for an underlying."""
    try:
        derivatives_service = get_derivatives_service()
        return derivatives_service.get_implied_volatility_smile(underlying, expiry_date)
    except Exception as e:
        logger.error(f"Error fetching IV smile: {str(e)}")
        return {}


# ============================================================================
# PHASE 3: ADVANCED FEATURES - CORPORATE ACTIONS
# ============================================================================

def get_corporate_actions(symbol: Optional[str] = None, action_type: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """Get corporate actions for a symbol or all actions."""
    try:
        corporate_actions_service = get_corporate_actions_service()
        actions = corporate_actions_service.get_corporate_actions(symbol, action_type, limit)
        return [action.__dict__ for action in actions]
    except Exception as e:
        logger.error(f"Error fetching corporate actions: {str(e)}")
        return []


def get_dividends(symbol: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """Get dividend information."""
    try:
        corporate_actions_service = get_corporate_actions_service()
        dividends = corporate_actions_service.get_dividends(symbol, limit)
        return [dividend.__dict__ for dividend in dividends]
    except Exception as e:
        logger.error(f"Error fetching dividends: {str(e)}")
        return []


def get_stock_splits(symbol: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """Get stock split information."""
    try:
        corporate_actions_service = get_corporate_actions_service()
        splits = corporate_actions_service.get_stock_splits(symbol, limit)
        return [split.__dict__ for split in splits]
    except Exception as e:
        logger.error(f"Error fetching stock splits: {str(e)}")
        return []


def get_bonus_issues(symbol: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """Get bonus issue information."""
    try:
        corporate_actions_service = get_corporate_actions_service()
        bonus_issues = corporate_actions_service.get_bonus_issues(symbol, limit)
        return [bonus.__dict__ for bonus in bonus_issues]
    except Exception as e:
        logger.error(f"Error fetching bonus issues: {str(e)}")
        return []


def get_upcoming_corporate_actions(days: int = 30) -> List[Dict[str, Any]]:
    """Get upcoming corporate actions in the next N days."""
    try:
        corporate_actions_service = get_corporate_actions_service()
        actions = corporate_actions_service.get_upcoming_actions(days)
        return [action.__dict__ for action in actions]
    except Exception as e:
        logger.error(f"Error fetching upcoming corporate actions: {str(e)}")
        return []


# ============================================================================
# PHASE 4: INTRADAY DATA FUNCTIONS
# ============================================================================

def get_intraday_prices(symbol: str, interval: str = "1min") -> List[Price]:
    """Get intraday price data for a symbol."""
    try:
        intraday_provider = get_intraday_provider()
        return intraday_provider.get_intraday_prices(symbol, interval)
    except Exception as e:
        logger.error(f"Error fetching intraday prices for {symbol}: {str(e)}")
        return []


def get_market_depth(symbol: str) -> Dict[str, Any]:
    """Get market depth data for a symbol."""
    try:
        intraday_provider = get_intraday_provider()
        return intraday_provider.get_market_depth(symbol)
    except Exception as e:
        logger.error(f"Error fetching market depth for {symbol}: {str(e)}")
        return {}


def get_live_option_chain(symbol: str, expiry_date: str = None) -> pd.DataFrame:
    """Get live option chain for a symbol."""
    try:
        intraday_provider = get_intraday_provider()
        return intraday_provider.get_live_option_chain(symbol, expiry_date)
    except Exception as e:
        logger.error(f"Error fetching option chain for {symbol}: {str(e)}")
        return pd.DataFrame()


# ============================================================================
# MARKET DATA FUNCTIONS
# ============================================================================

def get_top_gainers_losers() -> Dict[str, List[str]]:
    """Get top gainers and losers."""
    try:
        market_data_provider = get_market_data_provider()
        return market_data_provider.get_top_gainers_losers()
    except Exception as e:
        logger.error(f"Error fetching gainers/losers: {str(e)}")
        return {'gainers': [], 'losers': []}


def get_most_active_stocks(by: str = "volume") -> pd.DataFrame:
    """Get most active stocks by volume or value."""
    try:
        market_data_provider = get_market_data_provider()
        return market_data_provider.get_most_active_stocks(by)
    except Exception as e:
        logger.error(f"Error fetching most active stocks: {str(e)}")
        return pd.DataFrame()


def get_fii_dii_activity() -> pd.DataFrame:
    """Get FII/DII activity data."""
    try:
        market_data_provider = get_market_data_provider()
        return market_data_provider.get_fii_dii_activity()
    except Exception as e:
        logger.error(f"Error fetching FII/DII activity: {str(e)}")
        return pd.DataFrame()


def get_corporate_actions_data(from_date: str = None, to_date: str = None) -> pd.DataFrame:
    """Get corporate actions data."""
    try:
        market_data_provider = get_market_data_provider()
        return market_data_provider.get_corporate_actions(from_date, to_date)
    except Exception as e:
        logger.error(f"Error fetching corporate actions: {str(e)}")
        return pd.DataFrame()


def get_index_pe_ratios() -> pd.DataFrame:
    """Get index PE ratios."""
    try:
        market_data_provider = get_market_data_provider()
        return market_data_provider.get_index_pe_ratio()
    except Exception as e:
        logger.error(f"Error fetching index PE ratios: {str(e)}")
        return pd.DataFrame()


def get_advance_decline_data() -> pd.DataFrame:
    """Get advance/decline data."""
    try:
        market_data_provider = get_market_data_provider()
        return market_data_provider.get_advance_decline()
    except Exception as e:
        logger.error(f"Error fetching advance/decline data: {str(e)}")
        return pd.DataFrame()


def is_trading_holiday(date_str: str = None) -> bool:
    """Check if a date is a trading holiday."""
    try:
        market_data_provider = get_market_data_provider()
        return market_data_provider.is_trading_holiday(date_str)
    except Exception as e:
        logger.error(f"Error checking trading holiday: {str(e)}")
        return False


# ============================================================================
# PROVIDER SELECTION FUNCTIONS
# ============================================================================

def get_prices_with_provider(ticker: str, start_date: str, end_date: str, provider: str = "default") -> List[Price]:
    """Get prices with specified provider preference."""
    try:
        if provider == "yahoo":
            # Force Yahoo Finance
            return get_prices_yahoo(ticker, start_date, end_date)
        elif provider == "nse":
            # Force NSEUtility
            nse_provider = get_nse_utility_provider()
            return nse_provider.get_prices(ticker, start_date, end_date)
        else:
            # Use default logic
            return get_prices(ticker, start_date, end_date)
    except Exception as e:
        logger.error(f"Error fetching prices with provider {provider}: {str(e)}")
        return []


def get_prices_yahoo(ticker: str, start_date: str, end_date: str) -> List[Price]:
    """Force Yahoo Finance for price data."""
    try:
        # Use original Yahoo Finance logic
        cache_key = f"{ticker}_{start_date}_{end_date}"
        
        if cached_data := _cache.get_prices(cache_key):
            return [Price(**price) for price in cached_data]
        
        # Original Yahoo Finance implementation
        # ... (implement Yahoo Finance specific logic)
        return []
    except Exception as e:
        logger.error(f"Error fetching Yahoo Finance prices: {str(e)}")
        return [] 