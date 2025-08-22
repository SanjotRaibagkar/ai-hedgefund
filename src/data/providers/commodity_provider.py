import requests
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from src.data.providers.base_provider import DataProviderError

logger = logging.getLogger(__name__)


@dataclass
class Commodity:
    """Commodity data model."""
    symbol: str
    name: str
    category: str  # Precious Metals, Energy, Agriculture, etc.
    current_price: float
    currency: str
    unit: str  # per gram, per kg, per barrel, etc.
    change: float
    change_percent: float
    high_24h: float
    low_24h: float
    volume: Optional[int] = None
    open_interest: Optional[int] = None


@dataclass
class CommodityPrice:
    """Historical commodity price data."""
    symbol: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[int] = None


@dataclass
class FuturesContract:
    """Commodity futures contract data."""
    symbol: str
    contract_month: str  # e.g., "2024-12"
    expiry_date: str
    last_price: float
    change: float
    change_percent: float
    open_interest: int
    volume: int
    high: float
    low: float
    open: float


class IndianCommodityProvider:
    """Provider for Indian commodity market data."""
    
    def __init__(self):
        self.name = "Indian Commodity Provider"
        self.session = requests.Session()
        self.rate_limit_delay = 2.0
        self.cache = {}
        self.cache_expiry = 900  # 15 minutes cache
        
        # Setup session headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        })
        
        # Commodity categories
        self.categories = [
            'Precious Metals',
            'Energy',
            'Agriculture',
            'Base Metals',
            'Soft Commodities'
        ]
        
        # Major Indian commodities
        self.major_commodities = {
            'GOLD': {
                'name': 'Gold',
                'category': 'Precious Metals',
                'unit': 'per 10 grams',
                'currency': 'INR'
            },
            'SILVER': {
                'name': 'Silver',
                'category': 'Precious Metals',
                'unit': 'per kg',
                'currency': 'INR'
            },
            'CRUDEOIL': {
                'name': 'Crude Oil',
                'category': 'Energy',
                'unit': 'per barrel',
                'currency': 'USD'
            },
            'COPPER': {
                'name': 'Copper',
                'category': 'Base Metals',
                'unit': 'per kg',
                'currency': 'INR'
            },
            'ZINC': {
                'name': 'Zinc',
                'category': 'Base Metals',
                'unit': 'per kg',
                'currency': 'INR'
            },
            'NICKEL': {
                'name': 'Nickel',
                'category': 'Base Metals',
                'unit': 'per kg',
                'currency': 'INR'
            },
            'COTTON': {
                'name': 'Cotton',
                'category': 'Agriculture',
                'unit': 'per bale',
                'currency': 'INR'
            },
            'SUGAR': {
                'name': 'Sugar',
                'category': 'Agriculture',
                'unit': 'per quintal',
                'currency': 'INR'
            }
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
    
    def get_commodity_price(self, symbol: str) -> Optional[Commodity]:
        """Get current price for a specific commodity."""
        cache_key = f"commodity_price_{symbol}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            self._rate_limit()
            
            # Simulate commodity price data
            # In production, this would fetch from MCX or other exchanges
            if symbol not in self.major_commodities:
                logger.warning(f"Unknown commodity symbol: {symbol}")
                return None
            
            commodity_info = self.major_commodities[symbol]
            
            # Simulate price data based on commodity type
            base_prices = {
                'GOLD': 65000.0,
                'SILVER': 75000.0,
                'CRUDEOIL': 85.0,
                'COPPER': 850.0,
                'ZINC': 320.0,
                'NICKEL': 1800.0,
                'COTTON': 55000.0,
                'SUGAR': 3800.0
            }
            
            base_price = base_prices.get(symbol, 1000.0)
            change = (symbol.__hash__() % 1000) / 100.0 - 5.0  # Simulate change
            change_percent = (change / base_price) * 100
            
            commodity_data = {
                'symbol': symbol,
                'name': commodity_info['name'],
                'category': commodity_info['category'],
                'current_price': base_price + change,
                'currency': commodity_info['currency'],
                'unit': commodity_info['unit'],
                'change': change,
                'change_percent': change_percent,
                'high_24h': base_price + 100,
                'low_24h': base_price - 100,
                'volume': 1000000 + (symbol.__hash__() % 500000),
                'open_interest': 50000 + (symbol.__hash__() % 25000)
            }
            
            commodity = Commodity(**commodity_data)
            self._cache_data(cache_key, commodity)
            return commodity
            
        except Exception as e:
            logger.error(f"Error fetching commodity price for {symbol}: {str(e)}")
            return None
    
    def get_all_commodities(self) -> List[Commodity]:
        """Get current prices for all major commodities."""
        cache_key = "all_commodities"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            self._rate_limit()
            
            commodities = []
            for symbol in self.major_commodities.keys():
                commodity = self.get_commodity_price(symbol)
                if commodity:
                    commodities.append(commodity)
            
            self._cache_data(cache_key, commodities)
            return commodities
            
        except Exception as e:
            logger.error(f"Error fetching all commodities: {str(e)}")
            return []
    
    def get_commodity_prices(self, symbol: str, days: int = 30) -> List[CommodityPrice]:
        """Get historical price data for a commodity."""
        try:
            self._rate_limit()
            
            # Simulate historical price data
            prices = []
            base_price = 1000.0
            
            if symbol in self.major_commodities:
                base_prices = {
                    'GOLD': 65000.0,
                    'SILVER': 75000.0,
                    'CRUDEOIL': 85.0,
                    'COPPER': 850.0,
                    'ZINC': 320.0,
                    'NICKEL': 1800.0,
                    'COTTON': 55000.0,
                    'SUGAR': 3800.0
                }
                base_price = base_prices.get(symbol, 1000.0)
            
            for i in range(days):
                date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                
                # Simulate price movement
                price_variation = (i % 7) * 0.02 - 0.07
                current_price = base_price * (1 + price_variation)
                
                price_data = CommodityPrice(
                    symbol=symbol,
                    date=date,
                    open=current_price * 0.999,
                    high=current_price * 1.005,
                    low=current_price * 0.995,
                    close=current_price,
                    volume=1000000 + (i * 50000)
                )
                prices.append(price_data)
            
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching commodity prices for {symbol}: {str(e)}")
            return []
    
    def get_futures_contracts(self, symbol: str) -> List[FuturesContract]:
        """Get futures contracts for a commodity."""
        try:
            self._rate_limit()
            
            # Simulate futures contracts
            contracts = []
            base_price = 1000.0
            
            if symbol in self.major_commodities:
                base_prices = {
                    'GOLD': 65000.0,
                    'SILVER': 75000.0,
                    'CRUDEOIL': 85.0,
                    'COPPER': 850.0,
                    'ZINC': 320.0,
                    'NICKEL': 1800.0,
                    'COTTON': 55000.0,
                    'SUGAR': 3800.0
                }
                base_price = base_prices.get(symbol, 1000.0)
            
            # Generate contracts for next 6 months
            for i in range(6):
                contract_date = datetime.now() + timedelta(days=30 * (i + 1))
                contract_month = contract_date.strftime('%Y-%m')
                expiry_date = contract_date.strftime('%Y-%m-%d')
                
                # Simulate futures pricing (contango/backwardation)
                futures_price = base_price * (1 + (i * 0.01))  # Contango
                change = (i % 3) * 10 - 15
                change_percent = (change / futures_price) * 100
                
                contract = FuturesContract(
                    symbol=symbol,
                    contract_month=contract_month,
                    expiry_date=expiry_date,
                    last_price=futures_price,
                    change=change,
                    change_percent=change_percent,
                    open_interest=50000 + (i * 10000),
                    volume=100000 + (i * 20000),
                    high=futures_price + 50,
                    low=futures_price - 50,
                    open=futures_price - 10
                )
                contracts.append(contract)
            
            return contracts
            
        except Exception as e:
            logger.error(f"Error fetching futures contracts for {symbol}: {str(e)}")
            return []
    
    def get_commodities_by_category(self, category: str) -> List[Commodity]:
        """Get commodities filtered by category."""
        try:
            all_commodities = self.get_all_commodities()
            return [c for c in all_commodities if c.category == category]
        except Exception as e:
            logger.error(f"Error fetching commodities by category {category}: {str(e)}")
            return []
    
    def search_commodities(self, query: str) -> List[Commodity]:
        """Search commodities by name."""
        try:
            all_commodities = self.get_all_commodities()
            return [c for c in all_commodities if query.lower() in c.name.lower()]
        except Exception as e:
            logger.error(f"Error searching commodities: {str(e)}")
            return []
    
    def get_commodity_categories(self) -> List[str]:
        """Get list of available commodity categories."""
        return self.categories
    
    def get_major_commodities(self) -> Dict[str, Dict[str, str]]:
        """Get list of major commodities with their details."""
        return self.major_commodities


# Global commodity provider instance
_commodity_provider = None


def get_commodity_provider() -> IndianCommodityProvider:
    """Get the global commodity provider instance."""
    global _commodity_provider
    if _commodity_provider is None:
        _commodity_provider = IndianCommodityProvider()
    return _commodity_provider 