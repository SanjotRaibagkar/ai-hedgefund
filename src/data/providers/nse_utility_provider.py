"""
NSEUtility Provider for Indian Stock Market Data

This provider uses NSEUtility.py to fetch comprehensive Indian market data
including real-time prices, options data, intraday data, and market-specific metrics.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import pandas as pd

from src.data.providers.base_provider import BaseDataProvider, DataProviderError
from src.data.models import Price, FinancialMetrics, LineItem, CompanyNews, InsiderTrade
from src.nsedata.NseUtility import NseUtils

logger = logging.getLogger(__name__)


class NSEUtilityProvider(BaseDataProvider):
    """NSEUtility-based data provider for Indian markets."""
    
    def __init__(self):
        """Initialize NSEUtility provider."""
        try:
            self.nse = NseUtils()
            self._cache = {}
            self._cache_ttl = 300  # 5 minutes cache
            logger.info("NSEUtility provider initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize NSEUtility provider: {e}")
            raise DataProviderError(f"NSEUtility initialization failed: {e}")
    
    def _get_cached_data(self, key: str) -> Optional[Any]:
        """Get data from cache if not expired."""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self._cache_ttl):
                return data
            else:
                del self._cache[key]
        return None
    
    def _set_cached_data(self, key: str, data: Any):
        """Set data in cache with timestamp."""
        self._cache[key] = (data, datetime.now())
    
    def get_prices(self, symbol: str, start_date: str, end_date: str) -> List[Price]:
        """Get historical price data for a symbol."""
        try:
            # For real-time data, we'll get current price info
            price_info = self.nse.price_info(symbol)
            if not price_info:
                return []
            
            # Create a single price record with current data
            price = Price(
                time=datetime.now(),
                open=price_info['Open'],
                high=price_info['High'],
                low=price_info['Low'],
                close=price_info['LastTradedPrice'],
                volume=price_info.get('Volume', 0)
            )
            
            return [price]
            
        except Exception as e:
            logger.error(f"Error fetching prices for {symbol}: {e}")
            return []
    
    def get_prices_as_dataframe(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical price data as DataFrame for screening modules."""
        try:
            # Clean the symbol (remove .NS suffix for NSEUtility)
            clean_symbol = symbol.replace('.NS', '').replace('.BO', '').replace('.NSE', '').replace('.BSE', '')
            
            # Get current price info
            price_info = self.nse.price_info(clean_symbol)
            if not price_info:
                logger.warning(f"No price info available for {clean_symbol}")
                return pd.DataFrame()
            
            # Create DataFrame with proper column names that screening modules expect
            data = {
                'open_price': [price_info['Open']],
                'high_price': [price_info['High']],
                'low_price': [price_info['Low']],
                'close_price': [price_info['LastTradedPrice']],
                'volume': [price_info.get('Volume', 0)],
                'date': [datetime.now().date()]
            }
            
            df = pd.DataFrame(data)
            logger.info(f"Created DataFrame for {symbol} with {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching prices as DataFrame for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_financial_metrics(self, symbol: str, end_date: str, period: str = "ttm", limit: int = 1) -> List[FinancialMetrics]:
        """Get financial metrics for a symbol."""
        try:
            equity_info = self.nse.equity_info(symbol)
            if not equity_info or 'priceInfo' not in equity_info:
                return []
            
            price_info = equity_info['priceInfo']
            
            # Create financial metrics from available data
            metrics = FinancialMetrics(
                report_period=end_date,
                period=period,
                currency="INR",
                market_cap=price_info.get('marketCap', 0),
                enterprise_value=price_info.get('marketCap', 0),  # Approximate
                price_to_earnings_ratio=None,  # Not available in basic price info
                price_to_book_ratio=None,
                return_on_equity=None,
                debt_to_equity=None,
                earnings_growth=None,
                revenue_growth=None,
                book_value_growth=None
            )
            
            return [metrics]
            
        except Exception as e:
            logger.error(f"Error fetching financial metrics for {symbol}: {e}")
            return []
    
    def get_line_items(self, symbol: str, line_items: List[str], end_date: str, period: str = "ttm", limit: int = 1) -> List[LineItem]:
        """Get line items for a symbol."""
        try:
            equity_info = self.nse.equity_info(symbol)
            if not equity_info:
                return []
            
            # Create basic line items from available data
            line_item = LineItem(
                report_period=end_date,
                period=period,
                currency="INR",
                net_income=None,
                revenue=None,
                free_cash_flow=None,
                depreciation_and_amortization=None,
                capital_expenditure=None,
                working_capital=None
            )
            
            return [line_item]
            
        except Exception as e:
            logger.error(f"Error fetching line items for {symbol}: {e}")
            return []
    
    def get_market_cap(self, symbol: str, end_date: str) -> Optional[float]:
        """Get market cap for a symbol."""
        try:
            equity_info = self.nse.equity_info(symbol)
            if equity_info and 'priceInfo' in equity_info:
                return equity_info['priceInfo'].get('marketCap', 0)
            return None
        except Exception as e:
            logger.error(f"Error fetching market cap for {symbol}: {e}")
            return None
    
    def get_company_news(self, symbol: str, limit: int = 10) -> List[CompanyNews]:
        """Get company news for a symbol."""
        try:
            # Get corporate announcements
            corp_announcements = self.nse.get_corporate_announcement()
            if corp_announcements is None or corp_announcements.empty:
                return []
            
            # Filter announcements for the specific symbol
            symbol_announcements = corp_announcements[
                corp_announcements['symbol'].str.contains(symbol, case=False, na=False)
            ].head(limit)
            
            news_list = []
            for _, row in symbol_announcements.iterrows():
                news = CompanyNews(
                    ticker=symbol,
                    title=row.get('subject', ''),
                    content=row.get('subject', ''),
                    url='',  # NSE doesn't provide direct URLs
                    published_date=row.get('date', ''),
                    author='NSE',
                    sentiment=None
                )
                news_list.append(news)
            
            return news_list
            
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []
    
    def get_insider_trades(self, symbol: str, limit: int = 10) -> List[InsiderTrade]:
        """Get insider trading data for a symbol."""
        try:
            insider_data = self.nse.get_insider_trading()
            if insider_data is None or insider_data.empty:
                return []
            
            # Filter insider trades for the specific symbol
            symbol_insider = insider_data[
                insider_data['symbol'].str.contains(symbol, case=False, na=False)
            ].head(limit)
            
            trades_list = []
            for _, row in symbol_insider.iterrows():
                trade = InsiderTrade(
                    ticker=symbol,
                    insider_name=row.get('insiderName', ''),
                    insider_title=row.get('insiderTitle', ''),
                    trade_type=row.get('tradeType', ''),
                    shares_traded=row.get('sharesTraded', 0),
                    shares_owned=row.get('sharesOwned', 0),
                    price_per_share=row.get('pricePerShare', 0),
                    total_value=row.get('totalValue', 0),
                    trade_date=row.get('tradeDate', '')
                )
                trades_list.append(trade)
            
            return trades_list
            
        except Exception as e:
            logger.error(f"Error fetching insider trades for {symbol}: {e}")
            return []
    
    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "NSEUtility"
    
    def supports_ticker(self, ticker: str) -> bool:
        """Check if this provider supports the given ticker."""
        # NSEUtility supports Indian stocks
        return _is_indian_ticker(ticker)


def _is_indian_ticker(ticker: str) -> bool:
    """Check if ticker is an Indian stock."""
    ticker = ticker.upper()
    return any(ticker.endswith(suffix) for suffix in ['.NS', '.BO', '.NSE', '.BSE'])


class NSEIntradayProvider:
    """Provider for intraday data using NSEUtility."""
    
    def __init__(self):
        """Initialize intraday provider."""
        self.nse = NseUtils()
    
    def get_intraday_prices(self, symbol: str, interval: str = "1min") -> List[Price]:
        """Get intraday price data for a symbol."""
        try:
            # Get current price info (real-time)
            price_info = self.nse.price_info(symbol)
            if not price_info:
                return []
            
            # Create intraday price record
            price = Price(
                time=datetime.now(),
                open=price_info['Open'],
                high=price_info['High'],
                low=price_info['Low'],
                close=price_info['LastTradedPrice'],
                volume=price_info.get('Volume', 0)
            )
            
            return [price]
            
        except Exception as e:
            logger.error(f"Error fetching intraday prices for {symbol}: {e}")
            return []
    
    def get_market_depth(self, symbol: str) -> Dict[str, Any]:
        """Get market depth for a symbol."""
        try:
            depth_data = self.nse.get_market_depth(symbol)
            return depth_data
        except Exception as e:
            logger.error(f"Error fetching market depth for {symbol}: {e}")
            return {}
    
    def get_live_option_chain(self, symbol: str, expiry_date: str = None) -> pd.DataFrame:
        """Get live option chain for a symbol."""
        try:
            option_chain = self.nse.get_live_option_chain(symbol, expiry_date)
            return option_chain
        except Exception as e:
            logger.error(f"Error fetching option chain for {symbol}: {e}")
            return pd.DataFrame()


class NSEMarketDataProvider:
    """Provider for market-wide data using NSEUtility."""
    
    def __init__(self):
        """Initialize market data provider."""
        self.nse = NseUtils()
    
    def get_top_gainers_losers(self) -> Dict[str, List[str]]:
        """Get top gainers and losers."""
        try:
            gainers, losers = self.nse.get_gainers_losers()
            return {
                'gainers': gainers.get('Nifty Gainer', []),
                'losers': losers.get('Nifty Loser', [])
            }
        except Exception as e:
            logger.error(f"Error fetching gainers/losers: {e}")
            return {'gainers': [], 'losers': []}
    
    def get_most_active_stocks(self, by: str = "volume") -> pd.DataFrame:
        """Get most active stocks by volume or value."""
        try:
            if by == "volume":
                return self.nse.most_active_equity_stocks_by_volume()
            elif by == "value":
                return self.nse.most_active_equity_stocks_by_value()
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching most active stocks: {e}")
            return pd.DataFrame()
    
    def get_fii_dii_activity(self) -> pd.DataFrame:
        """Get FII/DII activity."""
        try:
            return self.nse.fii_dii_activity()
        except Exception as e:
            logger.error(f"Error fetching FII/DII activity: {e}")
            return pd.DataFrame()
    
    def get_corporate_actions(self, from_date: str = None, to_date: str = None) -> pd.DataFrame:
        """Get corporate actions."""
        try:
            return self.nse.get_corporate_action(from_date, to_date)
        except Exception as e:
            logger.error(f"Error fetching corporate actions: {e}")
            return pd.DataFrame()
    
    def get_index_pe_ratio(self) -> pd.DataFrame:
        """Get index PE ratios."""
        try:
            return self.nse.get_index_pe_ratio()
        except Exception as e:
            logger.error(f"Error fetching index PE ratios: {e}")
            return pd.DataFrame()
    
    def get_advance_decline(self) -> pd.DataFrame:
        """Get advance/decline data."""
        try:
            return self.nse.get_advance_decline()
        except Exception as e:
            logger.error(f"Error fetching advance/decline data: {e}")
            return pd.DataFrame()
    
    def is_trading_holiday(self, date_str: str = None) -> bool:
        """Check if a date is a trading holiday."""
        try:
            return self.nse.is_nse_trading_holiday(date_str)
        except Exception as e:
            logger.error(f"Error checking trading holiday: {e}")
            return False 