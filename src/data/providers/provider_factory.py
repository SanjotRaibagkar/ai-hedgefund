from typing import List, Dict, Type
import logging

from .base_provider import BaseDataProvider
from .yahoo_provider import YahooFinanceProvider
from .nse_provider import NSEIndiaProvider
from .nse_utility_provider import NSEUtilityProvider, NSEIntradayProvider, NSEMarketDataProvider
from .indian_news_provider import IndianNewsProvider
from .currency_provider import get_currency_provider
from .indian_market_calendar import get_indian_market_calendar
from .mutual_fund_provider import get_mutual_fund_provider
from .bond_provider import get_bond_provider
from .commodity_provider import get_commodity_provider
from .forex_provider import get_forex_provider
from .derivatives_provider import get_derivatives_provider
from .corporate_actions_provider import get_corporate_actions_provider
from .eod_data_provider import get_eod_data_provider

logger = logging.getLogger(__name__)


class DataProviderFactory:
    """Factory class to manage multiple data providers and select appropriate ones."""
    
    def __init__(self):
        self.providers: List[BaseDataProvider] = []
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all available data providers."""
        # Add NSEUtility provider as default for Indian stocks
        self.providers.append(NSEUtilityProvider())
        
        # Add Yahoo Finance provider as fallback
        self.providers.append(YahooFinanceProvider())
        
        # Add NSE India provider for additional Indian market specifics
        self.providers.append(NSEIndiaProvider())
        
        # Initialize specialized providers
        self.intraday_provider = NSEIntradayProvider()
        self.market_data_provider = NSEMarketDataProvider()
        
        # Initialize Phase 3 providers
        self.mutual_fund_provider = get_mutual_fund_provider()
        self.bond_provider = get_bond_provider()
        self.commodity_provider = get_commodity_provider()
        self.forex_provider = get_forex_provider()
        self.derivatives_provider = get_derivatives_provider()
        self.corporate_actions_provider = get_corporate_actions_provider()
        self.eod_data_provider = get_eod_data_provider()
        
        # Future providers can be added here:
        # self.providers.append(AlphaVantageProvider())
        
        logger.info(f"Initialized {len(self.providers)} data providers with NSEUtility as default")
    
    def get_provider_for_ticker(self, ticker: str) -> BaseDataProvider:
        """Get the best provider for a given ticker."""
        ticker = ticker.upper()
        
        # Find providers that support this ticker
        supported_providers = [p for p in self.providers if p.supports_ticker(ticker)]
        
        if not supported_providers:
            raise ValueError(f"No data provider found for ticker: {ticker}")
        
        # For now, return the first supported provider
        # In the future, we could implement provider ranking/priority
        provider = supported_providers[0]
        logger.debug(f"Selected provider '{provider.get_provider_name()}' for ticker '{ticker}'")
        
        return provider
    
    def get_all_providers(self) -> List[BaseDataProvider]:
        """Get all available providers."""
        return self.providers.copy()
    
    def add_provider(self, provider: BaseDataProvider):
        """Add a new data provider."""
        self.providers.append(provider)
        logger.info(f"Added new data provider: {provider.get_provider_name()}")
    
    def remove_provider(self, provider_name: str):
        """Remove a data provider by name."""
        self.providers = [p for p in self.providers if p.get_provider_name() != provider_name]
        logger.info(f"Removed data provider: {provider_name}")


# Global factory instance
_provider_factory = None


def get_provider_factory() -> DataProviderFactory:
    """Get the global provider factory instance."""
    global _provider_factory
    if _provider_factory is None:
        _provider_factory = DataProviderFactory()
    return _provider_factory


def get_provider_for_ticker(ticker: str) -> BaseDataProvider:
    """Convenience function to get provider for a ticker."""
    factory = get_provider_factory()
    return factory.get_provider_for_ticker(ticker)


def get_indian_news_provider() -> IndianNewsProvider:
    """Get the Indian news provider instance."""
    return IndianNewsProvider()


def get_intraday_provider() -> NSEIntradayProvider:
    """Get the intraday data provider instance."""
    factory = get_provider_factory()
    return factory.intraday_provider


def get_market_data_provider() -> NSEMarketDataProvider:
    """Get the market data provider instance."""
    factory = get_provider_factory()
    return factory.market_data_provider


def get_nse_utility_provider() -> NSEUtilityProvider:
    """Get the NSEUtility provider instance."""
    factory = get_provider_factory()
    return factory.providers[0]  # NSEUtility is first in the list


def get_currency_service():
    """Get the currency conversion service."""
    return get_currency_provider()


def get_market_calendar():
    """Get the Indian market calendar service."""
    return get_indian_market_calendar()


def get_mutual_fund_service():
    """Get the mutual fund service."""
    return get_mutual_fund_provider()


def get_bond_service():
    """Get the bond service."""
    return get_bond_provider()


def get_commodity_service():
    """Get the commodity service."""
    return get_commodity_provider()


def get_forex_service():
    """Get the forex service."""
    return get_forex_provider()


def get_derivatives_service():
    """Get the derivatives service."""
    return get_derivatives_provider()


def get_corporate_actions_service():
    """Get the corporate actions service."""
    return get_corporate_actions_provider()


def get_eod_data_service():
    """Get the EOD data service."""
    factory = get_provider_factory()
    return factory.eod_data_provider 