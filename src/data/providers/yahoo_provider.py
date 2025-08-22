import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import time
import logging

from .base_provider import BaseDataProvider, DataProviderError
from src.data.models import Price, FinancialMetrics, LineItem, CompanyNews, InsiderTrade

logger = logging.getLogger(__name__)


class YahooFinanceProvider(BaseDataProvider):
    """Yahoo Finance data provider supporting US and Indian stocks."""
    
    def __init__(self):
        self.name = "Yahoo Finance"
        self.supported_suffixes = ['.NS', '.BO', '.NSE', '.BSE']  # Indian market suffixes
        self.rate_limit_delay = 0.1  # 100ms delay between requests
        
    def get_provider_name(self) -> str:
        return self.name
    
    def supports_ticker(self, ticker: str) -> bool:
        """Check if this provider supports the ticker format."""
        # Support all tickers (Yahoo Finance is very comprehensive)
        return True
    
    def _rate_limit(self):
        """Implement rate limiting to avoid API restrictions."""
        time.sleep(self.rate_limit_delay)
    
    def _normalize_ticker(self, ticker: str) -> str:
        """Normalize ticker symbol for Yahoo Finance."""
        # Remove any existing suffixes and ensure proper format
        ticker = ticker.upper()
        
        # Handle Indian tickers
        if ticker.endswith('.NS') or ticker.endswith('.NSE'):
            return ticker.replace('.NSE', '.NS')
        elif ticker.endswith('.BO') or ticker.endswith('.BSE'):
            return ticker.replace('.BSE', '.BO')
        
        return ticker
    
    def get_prices(self, ticker: str, start_date: str, end_date: str) -> List[Price]:
        """Get historical price data from Yahoo Finance."""
        try:
            self._rate_limit()
            normalized_ticker = self._normalize_ticker(ticker)
            
            # Download data from Yahoo Finance
            stock = yf.Ticker(normalized_ticker)
            hist = stock.history(start=start_date, end=end_date, interval="1d")
            
            if hist.empty:
                logger.warning(f"No price data found for {ticker}")
                return []
            
            prices = []
            for date, row in hist.iterrows():
                price = Price(
                    open=float(row['Open']),
                    close=float(row['Close']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    volume=int(row['Volume']),
                    time=date.strftime('%Y-%m-%d')
                )
                prices.append(price)
            
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching price data for {ticker}: {str(e)}")
            raise DataProviderError(f"Failed to fetch price data for {ticker}: {str(e)}")
    
    def get_financial_metrics(self, ticker: str, end_date: str, period: str = "ttm", limit: int = 10) -> List[FinancialMetrics]:
        """Get financial metrics from Yahoo Finance."""
        try:
            self._rate_limit()
            normalized_ticker = self._normalize_ticker(ticker)
            
            stock = yf.Ticker(normalized_ticker)
            
            # Get financial info
            info = stock.info
            
            # Get financial statements
            if period.lower() == "ttm":
                financials = stock.financials
                balance_sheet = stock.balance_sheet
                cash_flow = stock.cashflow
            else:
                financials = stock.quarterly_financials
                balance_sheet = stock.quarterly_balance_sheet
                cash_flow = stock.quarterly_cashflow
            
            if financials.empty:
                logger.warning(f"No financial data found for {ticker}")
                return []
            
            metrics_list = []
            
            # Create financial metrics from available data
            for i, (date, row) in enumerate(financials.head(limit).iterrows()):
                if i >= limit:
                    break
                
                # Get corresponding balance sheet and cash flow data
                bs_row = balance_sheet.loc[date] if date in balance_sheet.index else pd.Series()
                cf_row = cash_flow.loc[date] if date in cash_flow.index else pd.Series()
                
                # Calculate metrics
                revenue = row.get('Total Revenue', 0)
                net_income = row.get('Net Income', 0)
                total_assets = bs_row.get('Total Assets', 0)
                total_equity = bs_row.get('Total Stockholder Equity', 0)
                total_debt = bs_row.get('Total Debt', 0)
                free_cash_flow = cf_row.get('Free Cash Flow', 0)
                
                # Calculate ratios
                roe = (net_income / total_equity * 100) if total_equity and total_equity != 0 else None
                roa = (net_income / total_assets * 100) if total_assets and total_assets != 0 else None
                debt_to_equity = (total_debt / total_equity) if total_equity and total_equity != 0 else None
                net_margin = (net_income / revenue * 100) if revenue and revenue != 0 else None
                
                # Get market cap from info
                market_cap = info.get('marketCap')
                if market_cap:
                    pe_ratio = (market_cap / net_income) if net_income and net_income != 0 else None
                    pb_ratio = (market_cap / total_equity) if total_equity and total_equity != 0 else None
                else:
                    pe_ratio = None
                    pb_ratio = None
                
                # Handle date conversion properly
                if hasattr(date, 'strftime'):
                    report_period = date.strftime('%Y-%m-%d')
                else:
                    report_period = str(date)
                
                metrics = FinancialMetrics(
                    ticker=normalized_ticker,
                    report_period=report_period,
                    period=period,
                    currency=info.get('currency', 'USD'),
                    market_cap=market_cap,
                    enterprise_value=info.get('enterpriseValue'),
                    price_to_earnings_ratio=pe_ratio,
                    price_to_book_ratio=pb_ratio,
                    price_to_sales_ratio=None,  # Would need to calculate
                    enterprise_value_to_ebitda_ratio=None,
                    enterprise_value_to_revenue_ratio=None,
                    free_cash_flow_yield=None,
                    peg_ratio=None,
                    gross_margin=None,
                    operating_margin=None,
                    net_margin=net_margin,
                    return_on_equity=roe,
                    return_on_assets=roa,
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
                    debt_to_equity=debt_to_equity,
                    debt_to_assets=None,
                    interest_coverage=None,
                    revenue_growth=None,
                    earnings_growth=None,
                    book_value_growth=None,
                    earnings_per_share_growth=None,
                    free_cash_flow_growth=None,
                    operating_income_growth=None,
                    ebitda_growth=None,
                    payout_ratio=None,
                    earnings_per_share=info.get('trailingEps'),
                    book_value_per_share=info.get('bookValue'),
                    free_cash_flow_per_share=None
                )
                
                metrics_list.append(metrics)
            
            return metrics_list
            
        except Exception as e:
            logger.error(f"Error fetching financial metrics for {ticker}: {str(e)}")
            raise DataProviderError(f"Failed to fetch financial metrics for {ticker}: {str(e)}")
    
    def get_line_items(self, ticker: str, line_items: List[str], end_date: str, period: str = "ttm", limit: int = 10) -> List[LineItem]:
        """Get specific financial line items from Yahoo Finance."""
        try:
            self._rate_limit()
            normalized_ticker = self._normalize_ticker(ticker)
            
            stock = yf.Ticker(normalized_ticker)
            
            # Get financial statements
            if period.lower() == "ttm":
                financials = stock.financials
                balance_sheet = stock.balance_sheet
                cash_flow = stock.cashflow
            else:
                financials = stock.quarterly_financials
                balance_sheet = stock.quarterly_balance_sheet
                cash_flow = stock.quarterly_cashflow
            
            if financials.empty:
                return []
            
            line_items_list = []
            
            for i, (date, row) in enumerate(financials.head(limit).iterrows()):
                if i >= limit:
                    break
                
                # Handle date conversion properly
                if hasattr(date, 'strftime'):
                    report_period = date.strftime('%Y-%m-%d')
                else:
                    report_period = str(date)
                
                # Create line item with available data
                line_item_data = {
                    'ticker': normalized_ticker,
                    'report_period': report_period,
                    'period': period,
                    'currency': stock.info.get('currency', 'USD')
                }
                
                # Map common line items to Yahoo Finance columns
                line_item_mapping = {
                    'revenue': 'Total Revenue',
                    'net_income': 'Net Income',
                    'operating_income': 'Operating Income',
                    'gross_profit': 'Gross Profit',
                    'total_assets': 'Total Assets',
                    'total_liabilities': 'Total Liab',
                    'current_assets': 'Current Assets',
                    'current_liabilities': 'Current Liabilities',
                    'cash_and_equivalents': 'Cash And Cash Equivalents',
                    'total_debt': 'Total Debt',
                    'shareholders_equity': 'Total Stockholder Equity',
                    'outstanding_shares': 'Shares Outstanding',
                    'free_cash_flow': 'Free Cash Flow',
                    'capital_expenditure': 'Capital Expenditure',
                    'research_and_development': 'Research And Development',
                    'operating_expense': 'Operating Expense',
                    'ebit': 'EBIT',
                    'ebitda': 'EBITDA'
                }
                
                # Add available line items
                for requested_item in line_items:
                    yf_column = line_item_mapping.get(requested_item.lower())
                    if yf_column and yf_column in row.index:
                        line_item_data[requested_item] = float(row[yf_column])
                    else:
                        line_item_data[requested_item] = None
                
                line_item = LineItem(**line_item_data)
                line_items_list.append(line_item)
            
            return line_items_list
            
        except Exception as e:
            logger.error(f"Error fetching line items for {ticker}: {str(e)}")
            raise DataProviderError(f"Failed to fetch line items for {ticker}: {str(e)}")
    
    def get_market_cap(self, ticker: str, end_date: str) -> Optional[float]:
        """Get market capitalization from Yahoo Finance."""
        try:
            self._rate_limit()
            normalized_ticker = self._normalize_ticker(ticker)
            
            stock = yf.Ticker(normalized_ticker)
            info = stock.info
            
            return info.get('marketCap')
            
        except Exception as e:
            logger.error(f"Error fetching market cap for {ticker}: {str(e)}")
            return None
    
    def get_company_news(self, ticker: str, end_date: str, start_date: Optional[str] = None, limit: int = 1000) -> List[CompanyNews]:
        """Get company news from Yahoo Finance."""
        try:
            self._rate_limit()
            normalized_ticker = self._normalize_ticker(ticker)
            
            stock = yf.Ticker(normalized_ticker)
            news = stock.news
            
            if not news:
                return []
            
            # Filter news by date if start_date is provided
            if start_date:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                filtered_news = []
                
                for item in news:
                    news_date = datetime.fromtimestamp(item.get('providerPublishTime', 0))
                    if start_dt <= news_date <= end_dt:
                        filtered_news.append(item)
                
                news = filtered_news
            
            # Limit results
            news = news[:limit]
            
            company_news_list = []
            for item in news:
                news_item = CompanyNews(
                    ticker=normalized_ticker,
                    title=item.get('title', ''),
                    summary=item.get('summary', ''),
                    url=item.get('link', ''),
                    date=datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%Y-%m-%d'),
                    source=item.get('publisher', ''),
                    sentiment=None  # Yahoo Finance doesn't provide sentiment
                )
                company_news_list.append(news_item)
            
            return company_news_list
            
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {str(e)}")
            return []
    
    def get_insider_trades(self, ticker: str, end_date: str, start_date: Optional[str] = None, limit: int = 1000) -> List[InsiderTrade]:
        """Get insider trades from Yahoo Finance."""
        # Yahoo Finance doesn't provide insider trading data
        # This would need to be implemented with a different data source
        logger.warning(f"Insider trading data not available from Yahoo Finance for {ticker}")
        return [] 