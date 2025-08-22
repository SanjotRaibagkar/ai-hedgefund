import requests
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from src.data.providers.base_provider import DataProviderError

logger = logging.getLogger(__name__)


@dataclass
class CorporateAction:
    """Corporate action data model."""
    symbol: str
    company_name: str
    action_type: str  # Dividend, Split, Bonus, Rights, Buyback, etc.
    announcement_date: str
    ex_date: str
    record_date: str
    payment_date: Optional[str] = None
    details: str = ""
    status: str = "Announced"  # Announced, Ex-Date, Record Date, Paid, Completed
    amount: Optional[float] = None
    ratio: Optional[str] = None  # e.g., "1:1" for bonus, "2:1" for split


@dataclass
class Dividend:
    """Dividend data model."""
    symbol: str
    company_name: str
    dividend_type: str  # Interim, Final, Special
    amount_per_share: float
    announcement_date: str
    ex_date: str
    record_date: str
    payment_date: str
    dividend_yield: float
    status: str = "Announced"


@dataclass
class StockSplit:
    """Stock split data model."""
    symbol: str
    company_name: str
    split_ratio: str  # e.g., "2:1", "5:1"
    announcement_date: str
    ex_date: str
    record_date: str
    status: str = "Announced"


@dataclass
class BonusIssue:
    """Bonus issue data model."""
    symbol: str
    company_name: str
    bonus_ratio: str  # e.g., "1:1", "1:2"
    announcement_date: str
    ex_date: str
    record_date: str
    status: str = "Announced"


class IndianCorporateActionsProvider:
    """Provider for Indian corporate actions data."""
    
    def __init__(self):
        self.name = "Indian Corporate Actions Provider"
        self.session = requests.Session()
        self.rate_limit_delay = 2.0
        self.cache = {}
        self.cache_expiry = 3600  # 1 hour cache
        
        # Setup session headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        })
        
        # Corporate action types
        self.action_types = [
            'Dividend',
            'Stock Split',
            'Bonus Issue',
            'Rights Issue',
            'Buyback',
            'Merger',
            'Acquisition',
            'Delisting',
            'IPO',
            'FPO'
        ]
        
        # Major companies for sample data
        self.major_companies = {
            'RELIANCE': 'Reliance Industries Limited',
            'TCS': 'Tata Consultancy Services Limited',
            'INFY': 'Infosys Limited',
            'HDFCBANK': 'HDFC Bank Limited',
            'ICICIBANK': 'ICICI Bank Limited',
            'HINDUNILVR': 'Hindustan Unilever Limited',
            'ITC': 'ITC Limited',
            'SBIN': 'State Bank of India',
            'BHARTIARTL': 'Bharti Airtel Limited',
            'AXISBANK': 'Axis Bank Limited'
        }
    
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
    
    def get_corporate_actions(self, symbol: Optional[str] = None, action_type: Optional[str] = None, limit: int = 20) -> List[CorporateAction]:
        """Get corporate actions for a symbol or all actions."""
        cache_key = f"corp_actions_{symbol}_{action_type}_{limit}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            self._rate_limit()
            
            # Simulate corporate actions data
            # In production, this would fetch from NSE/BSE or company announcements
            actions = []
            
            # Sample corporate actions
            sample_actions = [
                {
                    'symbol': 'RELIANCE',
                    'company_name': 'Reliance Industries Limited',
                    'action_type': 'Dividend',
                    'announcement_date': '2024-01-15',
                    'ex_date': '2024-02-15',
                    'record_date': '2024-02-17',
                    'payment_date': '2024-03-15',
                    'details': 'Interim Dividend of ₹9.00 per share',
                    'status': 'Paid',
                    'amount': 9.00
                },
                {
                    'symbol': 'TCS',
                    'company_name': 'Tata Consultancy Services Limited',
                    'action_type': 'Dividend',
                    'announcement_date': '2024-01-20',
                    'ex_date': '2024-02-20',
                    'record_date': '2024-02-22',
                    'payment_date': '2024-03-20',
                    'details': 'Interim Dividend of ₹24.00 per share',
                    'status': 'Paid',
                    'amount': 24.00
                },
                {
                    'symbol': 'INFY',
                    'company_name': 'Infosys Limited',
                    'action_type': 'Stock Split',
                    'announcement_date': '2024-02-01',
                    'ex_date': '2024-03-01',
                    'record_date': '2024-03-03',
                    'details': 'Stock Split in the ratio of 1:2',
                    'status': 'Completed',
                    'ratio': '1:2'
                },
                {
                    'symbol': 'HDFCBANK',
                    'company_name': 'HDFC Bank Limited',
                    'action_type': 'Bonus Issue',
                    'announcement_date': '2024-02-10',
                    'ex_date': '2024-03-10',
                    'record_date': '2024-03-12',
                    'details': 'Bonus Issue in the ratio of 1:1',
                    'status': 'Completed',
                    'ratio': '1:1'
                },
                {
                    'symbol': 'ITC',
                    'company_name': 'ITC Limited',
                    'action_type': 'Dividend',
                    'announcement_date': '2024-03-01',
                    'ex_date': '2024-04-01',
                    'record_date': '2024-04-03',
                    'payment_date': '2024-05-01',
                    'details': 'Final Dividend of ₹6.25 per share',
                    'status': 'Ex-Date',
                    'amount': 6.25
                }
            ]
            
            for action_data in sample_actions:
                if symbol is None or action_data['symbol'] == symbol:
                    if action_type is None or action_data['action_type'] == action_type:
                        action = CorporateAction(**action_data)
                        actions.append(action)
            
            actions = actions[:limit]
            self._cache_data(cache_key, actions)
            return actions
            
        except Exception as e:
            logger.error(f"Error fetching corporate actions: {str(e)}")
            raise DataProviderError(f"Failed to fetch corporate actions: {str(e)}")
    
    def get_dividends(self, symbol: Optional[str] = None, limit: int = 20) -> List[Dividend]:
        """Get dividend information."""
        try:
            self._rate_limit()
            
            # Simulate dividend data
            dividends = []
            
            # Sample dividends
            sample_dividends = [
                {
                    'symbol': 'RELIANCE',
                    'company_name': 'Reliance Industries Limited',
                    'dividend_type': 'Interim',
                    'amount_per_share': 9.00,
                    'announcement_date': '2024-01-15',
                    'ex_date': '2024-02-15',
                    'record_date': '2024-02-17',
                    'payment_date': '2024-03-15',
                    'dividend_yield': 0.63,  # 9.00 / 1424.80 * 100
                    'status': 'Paid'
                },
                {
                    'symbol': 'TCS',
                    'company_name': 'Tata Consultancy Services Limited',
                    'dividend_type': 'Interim',
                    'amount_per_share': 24.00,
                    'announcement_date': '2024-01-20',
                    'ex_date': '2024-02-20',
                    'record_date': '2024-02-22',
                    'payment_date': '2024-03-20',
                    'dividend_yield': 0.77,  # 24.00 / 3102.60 * 100
                    'status': 'Paid'
                },
                {
                    'symbol': 'ITC',
                    'company_name': 'ITC Limited',
                    'dividend_type': 'Final',
                    'amount_per_share': 6.25,
                    'announcement_date': '2024-03-01',
                    'ex_date': '2024-04-01',
                    'record_date': '2024-04-03',
                    'payment_date': '2024-05-01',
                    'dividend_yield': 1.25,  # 6.25 / 500.00 * 100 (estimated price)
                    'status': 'Ex-Date'
                }
            ]
            
            for dividend_data in sample_dividends:
                if symbol is None or dividend_data['symbol'] == symbol:
                    dividend = Dividend(**dividend_data)
                    dividends.append(dividend)
            
            return dividends[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching dividends: {str(e)}")
            return []
    
    def get_stock_splits(self, symbol: Optional[str] = None, limit: int = 20) -> List[StockSplit]:
        """Get stock split information."""
        try:
            self._rate_limit()
            
            # Simulate stock split data
            splits = []
            
            # Sample stock splits
            sample_splits = [
                {
                    'symbol': 'INFY',
                    'company_name': 'Infosys Limited',
                    'split_ratio': '1:2',
                    'announcement_date': '2024-02-01',
                    'ex_date': '2024-03-01',
                    'record_date': '2024-03-03',
                    'status': 'Completed'
                },
                {
                    'symbol': 'HINDUNILVR',
                    'company_name': 'Hindustan Unilever Limited',
                    'split_ratio': '1:1',
                    'announcement_date': '2023-12-15',
                    'ex_date': '2024-01-15',
                    'record_date': '2024-01-17',
                    'status': 'Completed'
                }
            ]
            
            for split_data in sample_splits:
                if symbol is None or split_data['symbol'] == symbol:
                    split = StockSplit(**split_data)
                    splits.append(split)
            
            return splits[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching stock splits: {str(e)}")
            return []
    
    def get_bonus_issues(self, symbol: Optional[str] = None, limit: int = 20) -> List[BonusIssue]:
        """Get bonus issue information."""
        try:
            self._rate_limit()
            
            # Simulate bonus issue data
            bonus_issues = []
            
            # Sample bonus issues
            sample_bonuses = [
                {
                    'symbol': 'HDFCBANK',
                    'company_name': 'HDFC Bank Limited',
                    'bonus_ratio': '1:1',
                    'announcement_date': '2024-02-10',
                    'ex_date': '2024-03-10',
                    'record_date': '2024-03-12',
                    'status': 'Completed'
                },
                {
                    'symbol': 'BHARTIARTL',
                    'company_name': 'Bharti Airtel Limited',
                    'bonus_ratio': '1:5',
                    'announcement_date': '2023-11-20',
                    'ex_date': '2023-12-20',
                    'record_date': '2023-12-22',
                    'status': 'Completed'
                }
            ]
            
            for bonus_data in sample_bonuses:
                if symbol is None or bonus_data['symbol'] == symbol:
                    bonus = BonusIssue(**bonus_data)
                    bonus_issues.append(bonus)
            
            return bonus_issues[:limit]
            
        except Exception as e:
            logger.error(f"Error fetching bonus issues: {str(e)}")
            return []
    
    def get_upcoming_actions(self, days: int = 30) -> List[CorporateAction]:
        """Get upcoming corporate actions in the next N days."""
        try:
            self._rate_limit()
            
            all_actions = self.get_corporate_actions()
            upcoming_actions = []
            
            cutoff_date = datetime.now() + timedelta(days=days)
            
            for action in all_actions:
                try:
                    ex_date = datetime.strptime(action.ex_date, '%Y-%m-%d')
                    if ex_date >= datetime.now() and ex_date <= cutoff_date:
                        upcoming_actions.append(action)
                except ValueError:
                    continue
            
            return upcoming_actions
            
        except Exception as e:
            logger.error(f"Error fetching upcoming actions: {str(e)}")
            return []
    
    def get_action_history(self, symbol: str, years: int = 5) -> List[CorporateAction]:
        """Get corporate action history for a symbol."""
        try:
            self._rate_limit()
            
            all_actions = self.get_corporate_actions(symbol)
            historical_actions = []
            
            cutoff_date = datetime.now() - timedelta(days=years * 365)
            
            for action in all_actions:
                try:
                    announcement_date = datetime.strptime(action.announcement_date, '%Y-%m-%d')
                    if announcement_date >= cutoff_date:
                        historical_actions.append(action)
                except ValueError:
                    continue
            
            return historical_actions
            
        except Exception as e:
            logger.error(f"Error fetching action history for {symbol}: {str(e)}")
            return []
    
    def search_actions(self, query: str) -> List[CorporateAction]:
        """Search corporate actions by company name or symbol."""
        try:
            all_actions = self.get_corporate_actions()
            results = []
            
            for action in all_actions:
                if (query.lower() in action.symbol.lower() or 
                    query.lower() in action.company_name.lower() or
                    query.lower() in action.action_type.lower()):
                    results.append(action)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching corporate actions: {str(e)}")
            return []
    
    def get_action_types(self) -> List[str]:
        """Get list of available corporate action types."""
        return self.action_types
    
    def get_major_companies(self) -> Dict[str, str]:
        """Get list of major companies."""
        return self.major_companies


# Global corporate actions provider instance
_corporate_actions_provider = None


def get_corporate_actions_provider() -> IndianCorporateActionsProvider:
    """Get the global corporate actions provider instance."""
    global _corporate_actions_provider
    if _corporate_actions_provider is None:
        _corporate_actions_provider = IndianCorporateActionsProvider()
    return _corporate_actions_provider 