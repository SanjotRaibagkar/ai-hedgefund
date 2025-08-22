from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from src.data.models import Price, FinancialMetrics, LineItem, CompanyNews, InsiderTrade


class BaseDataProvider(ABC):
    """Abstract base class for all data providers."""
    
    @abstractmethod
    def get_prices(self, ticker: str, start_date: str, end_date: str) -> List[Price]:
        """Get historical price data for a ticker."""
        pass
    
    @abstractmethod
    def get_financial_metrics(self, ticker: str, end_date: str, period: str = "ttm", limit: int = 10) -> List[FinancialMetrics]:
        """Get financial metrics for a ticker."""
        pass
    
    @abstractmethod
    def get_line_items(self, ticker: str, line_items: List[str], end_date: str, period: str = "ttm", limit: int = 10) -> List[LineItem]:
        """Get specific financial line items for a ticker."""
        pass
    
    @abstractmethod
    def get_market_cap(self, ticker: str, end_date: str) -> Optional[float]:
        """Get market capitalization for a ticker."""
        pass
    
    @abstractmethod
    def get_company_news(self, ticker: str, end_date: str, start_date: Optional[str] = None, limit: int = 1000) -> List[CompanyNews]:
        """Get company news for a ticker."""
        pass
    
    @abstractmethod
    def get_insider_trades(self, ticker: str, end_date: str, start_date: Optional[str] = None, limit: int = 1000) -> List[InsiderTrade]:
        """Get insider trades for a ticker."""
        pass
    
    @abstractmethod
    def supports_ticker(self, ticker: str) -> bool:
        """Check if this provider supports the given ticker format."""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of this provider."""
        pass


class DataProviderError(Exception):
    """Custom exception for data provider errors."""
    pass 