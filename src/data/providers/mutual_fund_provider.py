import requests
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from src.data.providers.base_provider import DataProviderError

logger = logging.getLogger(__name__)


@dataclass
class MutualFund:
    """Mutual fund data model."""
    fund_code: str
    fund_name: str
    fund_house: str
    category: str
    nav: float
    nav_date: str
    aum: float
    expense_ratio: float
    min_investment: float
    fund_type: str  # Equity, Debt, Hybrid, etc.
    benchmark: str
    risk_level: str  # Low, Moderate, High
    rating: Optional[int] = None
    returns_1y: Optional[float] = None
    returns_3y: Optional[float] = None
    returns_5y: Optional[float] = None


@dataclass
class FundPerformance:
    """Mutual fund performance data."""
    fund_code: str
    period: str
    returns: float
    benchmark_returns: float
    excess_returns: float
    rank: Optional[int] = None
    total_funds: Optional[int] = None


@dataclass
class FundHolding:
    """Mutual fund portfolio holding."""
    fund_code: str
    stock_name: str
    stock_symbol: str
    weight: float
    quantity: Optional[int] = None
    value: Optional[float] = None
    sector: Optional[str] = None


class IndianMutualFundProvider:
    """Provider for Indian mutual fund data."""
    
    def __init__(self):
        self.name = "Indian Mutual Fund Provider"
        self.session = requests.Session()
        self.rate_limit_delay = 2.0
        self.cache = {}
        self.cache_expiry = 3600  # 1 hour cache
        
        # Setup session headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        })
        
        # Popular Indian mutual fund houses
        self.fund_houses = [
            'HDFC Mutual Fund',
            'ICICI Prudential Mutual Fund',
            'SBI Mutual Fund',
            'Axis Mutual Fund',
            'Kotak Mutual Fund',
            'Aditya Birla Sun Life Mutual Fund',
            'Nippon India Mutual Fund',
            'UTI Mutual Fund',
            'Tata Mutual Fund',
            'Mirae Asset Mutual Fund'
        ]
        
        # Fund categories
        self.categories = [
            'Equity - Large Cap',
            'Equity - Mid Cap',
            'Equity - Small Cap',
            'Equity - Multi Cap',
            'Debt - Liquid',
            'Debt - Short Term',
            'Debt - Medium Term',
            'Hybrid - Balanced',
            'Hybrid - Conservative',
            'Hybrid - Aggressive',
            'Index Funds',
            'ETF'
        ]
    
    def _rate_limit(self):
        """Implement rate limiting."""
        time.sleep(self.rate_limit_delay)
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key].get('timestamp', 0)
        return (time.time() - cache_time) < self.cache_expiry
    
    def _get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Get cached data if valid."""
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key].get('data')
        return None
    
    def _cache_data(self, cache_key: str, data: Any):
        """Cache data with timestamp."""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }
    
    def get_top_performing_funds(self, category: Optional[str] = None, limit: int = 10) -> List[MutualFund]:
        """Get top performing mutual funds by category."""
        cache_key = f"top_funds_{category}_{limit}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            self._rate_limit()
            
            # Simulate top performing funds data
            # In production, this would fetch from AMFI or fund house APIs
            funds = []
            
            # Sample top performing funds
            sample_funds = [
                {
                    'fund_code': 'HDFC0001',
                    'fund_name': 'HDFC Top 100 Fund',
                    'fund_house': 'HDFC Mutual Fund',
                    'category': 'Equity - Large Cap',
                    'nav': 45.67,
                    'nav_date': datetime.now().strftime('%Y-%m-%d'),
                    'aum': 15000.0,  # in crores
                    'expense_ratio': 1.75,
                    'min_investment': 5000.0,
                    'fund_type': 'Equity',
                    'benchmark': 'NIFTY 100',
                    'risk_level': 'Moderate',
                    'rating': 4,
                    'returns_1y': 18.5,
                    'returns_3y': 15.2,
                    'returns_5y': 12.8
                },
                {
                    'fund_code': 'ICIC0001',
                    'fund_name': 'ICICI Prudential Bluechip Fund',
                    'fund_house': 'ICICI Prudential Mutual Fund',
                    'category': 'Equity - Large Cap',
                    'nav': 52.34,
                    'nav_date': datetime.now().strftime('%Y-%m-%d'),
                    'aum': 12000.0,
                    'expense_ratio': 1.85,
                    'min_investment': 5000.0,
                    'fund_type': 'Equity',
                    'benchmark': 'NIFTY 50',
                    'risk_level': 'Moderate',
                    'rating': 4,
                    'returns_1y': 16.8,
                    'returns_3y': 14.5,
                    'returns_5y': 11.9
                },
                {
                    'fund_code': 'SBI0001',
                    'fund_name': 'SBI Bluechip Fund',
                    'fund_house': 'SBI Mutual Fund',
                    'category': 'Equity - Large Cap',
                    'nav': 38.92,
                    'nav_date': datetime.now().strftime('%Y-%m-%d'),
                    'aum': 8500.0,
                    'expense_ratio': 1.95,
                    'min_investment': 5000.0,
                    'fund_type': 'Equity',
                    'benchmark': 'NIFTY 50',
                    'risk_level': 'Moderate',
                    'rating': 3,
                    'returns_1y': 15.2,
                    'returns_3y': 13.8,
                    'returns_5y': 10.5
                }
            ]
            
            for fund_data in sample_funds:
                if category is None or fund_data['category'] == category:
                    fund = MutualFund(**fund_data)
                    funds.append(fund)
            
            # Sort by 1-year returns
            funds.sort(key=lambda x: x.returns_1y or 0, reverse=True)
            funds = funds[:limit]
            
            self._cache_data(cache_key, funds)
            return funds
            
        except Exception as e:
            logger.error(f"Error fetching top performing funds: {str(e)}")
            raise DataProviderError(f"Failed to fetch top performing funds: {str(e)}")
    
    def get_fund_details(self, fund_code: str) -> Optional[MutualFund]:
        """Get detailed information about a specific mutual fund."""
        cache_key = f"fund_details_{fund_code}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            self._rate_limit()
            
            # Simulate fund details
            # In production, this would fetch from AMFI API
            fund_data = {
                'fund_code': fund_code,
                'fund_name': f'Sample Fund {fund_code}',
                'fund_house': 'Sample Fund House',
                'category': 'Equity - Large Cap',
                'nav': 45.67,
                'nav_date': datetime.now().strftime('%Y-%m-%d'),
                'aum': 15000.0,
                'expense_ratio': 1.75,
                'min_investment': 5000.0,
                'fund_type': 'Equity',
                'benchmark': 'NIFTY 100',
                'risk_level': 'Moderate',
                'rating': 4,
                'returns_1y': 18.5,
                'returns_3y': 15.2,
                'returns_5y': 12.8
            }
            
            fund = MutualFund(**fund_data)
            self._cache_data(cache_key, fund)
            return fund
            
        except Exception as e:
            logger.error(f"Error fetching fund details for {fund_code}: {str(e)}")
            return None
    
    def get_fund_performance(self, fund_code: str, period: str = '1y') -> Optional[FundPerformance]:
        """Get performance data for a specific fund."""
        try:
            self._rate_limit()
            
            # Simulate performance data
            performance_data = {
                'fund_code': fund_code,
                'period': period,
                'returns': 18.5,
                'benchmark_returns': 15.2,
                'excess_returns': 3.3,
                'rank': 5,
                'total_funds': 50
            }
            
            return FundPerformance(**performance_data)
            
        except Exception as e:
            logger.error(f"Error fetching fund performance for {fund_code}: {str(e)}")
            return None
    
    def get_fund_holdings(self, fund_code: str) -> List[FundHolding]:
        """Get portfolio holdings for a specific fund."""
        cache_key = f"fund_holdings_{fund_code}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            self._rate_limit()
            
            # Simulate fund holdings
            # In production, this would fetch from fund house APIs
            holdings = [
                FundHolding(
                    fund_code=fund_code,
                    stock_name='Reliance Industries',
                    stock_symbol='RELIANCE.NS',
                    weight=8.5,
                    quantity=100000,
                    value=142480000.0,
                    sector='Oil & Gas'
                ),
                FundHolding(
                    fund_code=fund_code,
                    stock_name='TCS',
                    stock_symbol='TCS.NS',
                    weight=7.2,
                    quantity=50000,
                    value=155130000.0,
                    sector='IT'
                ),
                FundHolding(
                    fund_code=fund_code,
                    stock_name='HDFC Bank',
                    stock_symbol='HDFCBANK.NS',
                    weight=6.8,
                    quantity=75000,
                    value=149340000.0,
                    sector='Banking'
                )
            ]
            
            self._cache_data(cache_key, holdings)
            return holdings
            
        except Exception as e:
            logger.error(f"Error fetching fund holdings for {fund_code}: {str(e)}")
            return []
    
    def search_funds(self, query: str, category: Optional[str] = None) -> List[MutualFund]:
        """Search mutual funds by name or fund house."""
        try:
            self._rate_limit()
            
            # Simulate fund search
            # In production, this would search AMFI database
            funds = []
            
            # Sample search results
            sample_funds = [
                {
                    'fund_code': 'HDFC0001',
                    'fund_name': 'HDFC Top 100 Fund',
                    'fund_house': 'HDFC Mutual Fund',
                    'category': 'Equity - Large Cap',
                    'nav': 45.67,
                    'nav_date': datetime.now().strftime('%Y-%m-%d'),
                    'aum': 15000.0,
                    'expense_ratio': 1.75,
                    'min_investment': 5000.0,
                    'fund_type': 'Equity',
                    'benchmark': 'NIFTY 100',
                    'risk_level': 'Moderate',
                    'rating': 4,
                    'returns_1y': 18.5,
                    'returns_3y': 15.2,
                    'returns_5y': 12.8
                }
            ]
            
            for fund_data in sample_funds:
                if (query.lower() in fund_data['fund_name'].lower() or 
                    query.lower() in fund_data['fund_house'].lower()):
                    if category is None or fund_data['category'] == category:
                        fund = MutualFund(**fund_data)
                        funds.append(fund)
            
            return funds
            
        except Exception as e:
            logger.error(f"Error searching funds: {str(e)}")
            return []
    
    def get_fund_categories(self) -> List[str]:
        """Get list of available fund categories."""
        return self.categories
    
    def get_fund_houses(self) -> List[str]:
        """Get list of available fund houses."""
        return self.fund_houses


# Global mutual fund provider instance
_mutual_fund_provider = None


def get_mutual_fund_provider() -> IndianMutualFundProvider:
    """Get the global mutual fund provider instance."""
    global _mutual_fund_provider
    if _mutual_fund_provider is None:
        _mutual_fund_provider = IndianMutualFundProvider()
    return _mutual_fund_provider 