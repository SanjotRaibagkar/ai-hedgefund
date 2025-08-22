import requests
import json
import time
import pytz
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging
from bs4 import BeautifulSoup

from .base_provider import BaseDataProvider, DataProviderError
from src.data.models import Price, FinancialMetrics, LineItem, CompanyNews, InsiderTrade

logger = logging.getLogger(__name__)


class NSEIndiaProvider(BaseDataProvider):
    """NSE India data provider for real-time Indian market data and market-specific features."""
    
    def __init__(self):
        self.name = "NSE India"
        self.base_url = "https://www.nseindia.com"
        self.api_base = "https://www.nseindia.com/api"
        self.session = requests.Session()
        self.rate_limit_delay = 1.0  # 1 second delay between requests
        self.ist_timezone = pytz.timezone('Asia/Kolkata')
        
        # Initialize session with proper headers
        self._setup_session()
        
    def get_provider_name(self) -> str:
        return self.name
    
    def supports_ticker(self, ticker: str) -> bool:
        """Check if this provider supports the ticker format."""
        ticker = ticker.upper()
        # Support NSE tickers without suffix or with .NS suffix
        return ticker.endswith('.NS') or (not ticker.endswith('.BO') and '.' not in ticker)
    
    def _setup_session(self):
        """Setup session with proper headers to mimic browser requests."""
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://www.nseindia.com/',
        })
        
        # Get initial page to establish session
        try:
            self.session.get(self.base_url, timeout=10)
        except Exception as e:
            logger.warning(f"Failed to establish NSE session: {str(e)}")
    
    def _rate_limit(self):
        """Implement rate limiting to avoid overwhelming NSE servers."""
        time.sleep(self.rate_limit_delay)
    
    def _normalize_ticker(self, ticker: str) -> str:
        """Normalize ticker symbol for NSE."""
        ticker = ticker.upper()
        # Remove .NS suffix if present
        if ticker.endswith('.NS'):
            ticker = ticker[:-3]
        return ticker
    
    def _make_nse_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make a request to NSE API with proper error handling."""
        try:
            self._rate_limit()
            url = f"{self.api_base}/{endpoint}"
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"NSE API request failed for {endpoint}: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse NSE API response for {endpoint}: {str(e)}")
            return None
    
    def get_prices(self, ticker: str, start_date: str, end_date: str) -> List[Price]:
        """Get historical price data from NSE."""
        try:
            normalized_ticker = self._normalize_ticker(ticker)
            
            # NSE API for historical data
            endpoint = f"historical/cm/equity"
            params = {
                'symbol': normalized_ticker,
                'series': '["EQ"]',
                'from': start_date,
                'to': end_date
            }
            
            data = self._make_nse_request(endpoint, params)
            
            if not data or 'data' not in data:
                logger.warning(f"No price data found for {ticker} from NSE")
                return []
            
            prices = []
            for record in data['data']:
                try:
                    price = Price(
                        open=float(record.get('CH_OPENING_PRICE', 0)),
                        close=float(record.get('CH_CLOSING_PRICE', 0)),
                        high=float(record.get('CH_TRADE_HIGH_PRICE', 0)),
                        low=float(record.get('CH_TRADE_LOW_PRICE', 0)),
                        volume=int(record.get('CH_TOT_TRADED_QTY', 0)),
                        time=record.get('CH_TIMESTAMP', '')
                    )
                    prices.append(price)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid price record for {ticker}: {str(e)}")
                    continue
            
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching NSE price data for {ticker}: {str(e)}")
            raise DataProviderError(f"Failed to fetch NSE price data for {ticker}: {str(e)}")
    
    def get_financial_metrics(self, ticker: str, end_date: str, period: str = "ttm", limit: int = 10) -> List[FinancialMetrics]:
        """Get financial metrics from NSE (limited data available)."""
        try:
            normalized_ticker = self._normalize_ticker(ticker)
            
            # Get quote data which includes some financial metrics
            endpoint = f"quote-equity"
            params = {'symbol': normalized_ticker}
            
            data = self._make_nse_request(endpoint, params)
            
            if not data:
                logger.warning(f"No financial data found for {ticker} from NSE")
                return []
            
            # Extract available metrics
            info = data.get('info', {})
            price_info = data.get('priceInfo', {})
            
            # Calculate basic metrics
            market_cap = info.get('marketCap')
            pe_ratio = info.get('pe')
            pb_ratio = info.get('pb')
            dividend_yield = info.get('dividendYield')
            
            metrics = FinancialMetrics(
                ticker=f"{normalized_ticker}.NS",
                report_period=end_date,
                period=period,
                currency='INR',
                market_cap=market_cap,
                enterprise_value=None,
                price_to_earnings_ratio=pe_ratio,
                price_to_book_ratio=pb_ratio,
                price_to_sales_ratio=None,
                enterprise_value_to_ebitda_ratio=None,
                enterprise_value_to_revenue_ratio=None,
                free_cash_flow_yield=None,
                peg_ratio=None,
                gross_margin=None,
                operating_margin=None,
                net_margin=None,
                return_on_equity=None,
                return_on_assets=None,
                return_on_invested_capital=None,
                asset_turnover=None,
                inventory_turnover=None,
                receivables_turnover=None,
                days_sales_outstanding=None,
                operating_cycle=None,
                working_capital_turnover=None,
                current_ratio=None,
                quick_ratio=None,
                cash_ratio=None,
                operating_cash_flow_ratio=None,
                debt_to_equity=None,
                debt_to_assets=None,
                interest_coverage=None,
                revenue_growth=None,
                earnings_growth=None,
                book_value_growth=None,
                earnings_per_share_growth=None,
                free_cash_flow_growth=None,
                operating_income_growth=None,
                ebitda_growth=None,
                payout_ratio=dividend_yield,
                earnings_per_share=None,
                book_value_per_share=None,
                free_cash_flow_per_share=None
            )
            
            return [metrics]
            
        except Exception as e:
            logger.error(f"Error fetching NSE financial metrics for {ticker}: {str(e)}")
            raise DataProviderError(f"Failed to fetch NSE financial metrics for {ticker}: {str(e)}")
    
    def get_line_items(self, ticker: str, line_items: List[str], end_date: str, period: str = "ttm", limit: int = 10) -> List[LineItem]:
        """Get specific financial line items from NSE (limited data available)."""
        # NSE doesn't provide detailed financial statements through API
        logger.warning(f"Detailed financial line items not available from NSE for {ticker}")
        return []
    
    def get_market_cap(self, ticker: str, end_date: str) -> Optional[float]:
        """Get market capitalization from NSE."""
        try:
            normalized_ticker = self._normalize_ticker(ticker)
            
            endpoint = f"quote-equity"
            params = {'symbol': normalized_ticker}
            
            data = self._make_nse_request(endpoint, params)
            
            if not data or 'info' not in data:
                return None
            
            return data['info'].get('marketCap')
            
        except Exception as e:
            logger.error(f"Error fetching NSE market cap for {ticker}: {str(e)}")
            return None
    
    def get_company_news(self, ticker: str, end_date: str, start_date: Optional[str] = None, limit: int = 1000) -> List[CompanyNews]:
        """Get company news from NSE announcements."""
        try:
            normalized_ticker = self._normalize_ticker(ticker)
            
            # Get corporate announcements
            endpoint = f"corporate-announcements"
            params = {
                'symbol': normalized_ticker,
                'from_date': start_date or (datetime.now() - timedelta(days=30)).strftime('%d-%m-%Y'),
                'to_date': datetime.strptime(end_date, '%Y-%m-%d').strftime('%d-%m-%Y')
            }
            
            data = self._make_nse_request(endpoint, params)
            
            if not data:
                return []
            
            news_list = []
            announcements = data if isinstance(data, list) else data.get('data', [])
            
            for announcement in announcements[:limit]:
                try:
                    news_item = CompanyNews(
                        ticker=f"{normalized_ticker}.NS",
                        title=announcement.get('desc', ''),
                        summary=announcement.get('an_dt', ''),
                        url=f"{self.base_url}/corporates/content/announcements",
                        date=announcement.get('an_dt', ''),
                        source='NSE India',
                        sentiment=None
                    )
                    news_list.append(news_item)
                except Exception as e:
                    logger.warning(f"Skipping invalid news record for {ticker}: {str(e)}")
                    continue
            
            return news_list
            
        except Exception as e:
            logger.error(f"Error fetching NSE news for {ticker}: {str(e)}")
            return []
    
    def get_insider_trades(self, ticker: str, end_date: str, start_date: Optional[str] = None, limit: int = 1000) -> List[InsiderTrade]:
        """Get insider trades from NSE (if available)."""
        # NSE doesn't provide insider trading data through public API
        logger.warning(f"Insider trading data not available from NSE for {ticker}")
        return []
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get current Indian market status."""
        try:
            endpoint = "marketStatus"
            data = self._make_nse_request(endpoint)
            
            if not data:
                return {}
            
            return {
                'market_status': data.get('marketState', []),
                'timestamp': datetime.now(self.ist_timezone).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching market status: {str(e)}")
            return {}
    
    def get_sector_performance(self) -> Dict[str, Any]:
        """Get Indian sector performance data."""
        try:
            endpoint = "equity-stockIndices"
            params = {'index': 'SECTORAL INDICES'}
            
            data = self._make_nse_request(endpoint, params)
            
            if not data:
                return {}
            
            sector_data = {}
            for sector in data.get('data', []):
                sector_data[sector.get('index', '')] = {
                    'last_price': sector.get('last', 0),
                    'change': sector.get('change', 0),
                    'percent_change': sector.get('pChange', 0)
                }
            
            return sector_data
            
        except Exception as e:
            logger.error(f"Error fetching sector performance: {str(e)}")
            return {}
    
    def get_top_gainers_losers(self) -> Dict[str, List[Dict]]:
        """Get top gainers and losers from NSE."""
        try:
            gainers_endpoint = "equity-stockIndices"
            gainers_params = {'index': 'SECURITIES IN F&O'}
            
            data = self._make_nse_request(gainers_endpoint, gainers_params)
            
            if not data:
                return {'gainers': [], 'losers': []}
            
            stocks = data.get('data', [])
            
            # Sort by percentage change
            gainers = sorted([s for s in stocks if s.get('pChange', 0) > 0], 
                           key=lambda x: x.get('pChange', 0), reverse=True)[:10]
            
            losers = sorted([s for s in stocks if s.get('pChange', 0) < 0], 
                          key=lambda x: x.get('pChange', 0))[:10]
            
            return {
                'gainers': gainers,
                'losers': losers
            }
            
        except Exception as e:
            logger.error(f"Error fetching top gainers/losers: {str(e)}")
            return {'gainers': [], 'losers': []}
    
    def is_market_open(self) -> bool:
        """Check if Indian market is currently open."""
        try:
            now = datetime.now(self.ist_timezone)
            
            # Check if it's a weekday (Monday = 0, Sunday = 6)
            if now.weekday() >= 5:  # Saturday or Sunday
                return False
            
            # Market hours: 9:15 AM to 3:30 PM IST
            market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            return market_open <= now <= market_close
            
        except Exception as e:
            logger.error(f"Error checking market status: {str(e)}")
            return False