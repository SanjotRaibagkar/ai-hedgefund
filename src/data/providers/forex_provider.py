import requests
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from src.data.providers.base_provider import DataProviderError

logger = logging.getLogger(__name__)


@dataclass
class ForexRate:
    """Forex rate data model."""
    pair: str  # e.g., "USDINR", "EURINR"
    base_currency: str
    quote_currency: str
    bid: float
    ask: float
    mid_rate: float
    spread: float
    change: float
    change_percent: float
    high_24h: float
    low_24h: float
    timestamp: str
    source: str = "RBI"


@dataclass
class ForexPrice:
    """Historical forex price data."""
    pair: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[int] = None


@dataclass
class CrossRate:
    """Cross currency rate."""
    pair: str
    rate: float
    inverse_rate: float
    timestamp: str


class IndianForexProvider:
    """Provider for Indian forex market data."""
    
    def __init__(self):
        self.name = "Indian Forex Provider"
        self.session = requests.Session()
        self.rate_limit_delay = 2.0
        self.cache = {}
        self.cache_expiry = 300  # 5 minutes cache
        
        # Setup session headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        })
        
        # Major currency pairs with INR
        self.major_pairs = {
            'USDINR': {
                'base': 'USD',
                'quote': 'INR',
                'description': 'US Dollar / Indian Rupee'
            },
            'EURINR': {
                'base': 'EUR',
                'quote': 'INR',
                'description': 'Euro / Indian Rupee'
            },
            'GBPINR': {
                'base': 'GBP',
                'quote': 'INR',
                'description': 'British Pound / Indian Rupee'
            },
            'JPYINR': {
                'base': 'JPY',
                'quote': 'INR',
                'description': 'Japanese Yen / Indian Rupee'
            },
            'AUDINR': {
                'base': 'AUD',
                'quote': 'INR',
                'description': 'Australian Dollar / Indian Rupee'
            },
            'CADINR': {
                'base': 'CAD',
                'quote': 'INR',
                'description': 'Canadian Dollar / Indian Rupee'
            },
            'CHFINR': {
                'base': 'CHF',
                'quote': 'INR',
                'description': 'Swiss Franc / Indian Rupee'
            },
            'CNYINR': {
                'base': 'CNY',
                'quote': 'INR',
                'description': 'Chinese Yuan / Indian Rupee'
            }
        }
        
        # Cross currency pairs
        self.cross_pairs = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD',
            'USDCAD', 'USDCHF', 'EURGBP', 'EURJPY'
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
    
    def get_forex_rate(self, pair: str) -> Optional[ForexRate]:
        """Get current forex rate for a currency pair."""
        cache_key = f"forex_rate_{pair}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            self._rate_limit()
            
            # Simulate forex rate data
            # In production, this would fetch from RBI or other forex sources
            if pair not in self.major_pairs:
                logger.warning(f"Unknown forex pair: {pair}")
                return None
            
            pair_info = self.major_pairs[pair]
            
            # Simulate rate data based on pair
            base_rates = {
                'USDINR': 87.0,
                'EURINR': 95.0,
                'GBPINR': 110.0,
                'JPYINR': 0.58,
                'AUDINR': 57.0,
                'CADINR': 64.0,
                'CHFINR': 98.0,
                'CNYINR': 12.0
            }
            
            base_rate = base_rates.get(pair, 80.0)
            spread = base_rate * 0.001  # 0.1% spread
            bid = base_rate - spread / 2
            ask = base_rate + spread / 2
            
            # Simulate change
            change = (pair.__hash__() % 100) / 100.0 - 0.5
            change_percent = (change / base_rate) * 100
            
            forex_data = {
                'pair': pair,
                'base_currency': pair_info['base'],
                'quote_currency': pair_info['quote'],
                'bid': bid,
                'ask': ask,
                'mid_rate': base_rate,
                'spread': spread,
                'change': change,
                'change_percent': change_percent,
                'high_24h': base_rate + 0.5,
                'low_24h': base_rate - 0.5,
                'timestamp': datetime.now().isoformat(),
                'source': 'RBI'
            }
            
            forex_rate = ForexRate(**forex_data)
            self._cache_data(cache_key, forex_rate)
            return forex_rate
            
        except Exception as e:
            logger.error(f"Error fetching forex rate for {pair}: {str(e)}")
            return None
    
    def get_all_forex_rates(self) -> List[ForexRate]:
        """Get current rates for all major forex pairs."""
        cache_key = "all_forex_rates"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            self._rate_limit()
            
            rates = []
            for pair in self.major_pairs.keys():
                rate = self.get_forex_rate(pair)
                if rate:
                    rates.append(rate)
            
            self._cache_data(cache_key, rates)
            return rates
            
        except Exception as e:
            logger.error(f"Error fetching all forex rates: {str(e)}")
            return []
    
    def get_forex_prices(self, pair: str, days: int = 30) -> List[ForexPrice]:
        """Get historical forex price data."""
        try:
            self._rate_limit()
            
            # Simulate historical price data
            prices = []
            base_rate = 87.0  # Default USDINR rate
            
            if pair in self.major_pairs:
                base_rates = {
                    'USDINR': 87.0,
                    'EURINR': 95.0,
                    'GBPINR': 110.0,
                    'JPYINR': 0.58,
                    'AUDINR': 57.0,
                    'CADINR': 64.0,
                    'CHFINR': 98.0,
                    'CNYINR': 12.0
                }
                base_rate = base_rates.get(pair, 87.0)
            
            for i in range(days):
                date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                
                # Simulate price movement
                price_variation = (i % 7) * 0.005 - 0.02
                current_rate = base_rate * (1 + price_variation)
                
                price_data = ForexPrice(
                    pair=pair,
                    date=date,
                    open=current_rate * 0.999,
                    high=current_rate * 1.002,
                    low=current_rate * 0.998,
                    close=current_rate,
                    volume=1000000 + (i * 50000)
                )
                prices.append(price_data)
            
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching forex prices for {pair}: {str(e)}")
            return []
    
    def get_cross_rates(self) -> List[CrossRate]:
        """Get cross currency rates."""
        try:
            self._rate_limit()
            
            # Simulate cross rates
            cross_rates = []
            
            # Sample cross rates
            sample_crosses = [
                ('EURUSD', 1.09),
                ('GBPUSD', 1.26),
                ('USDJPY', 150.0),
                ('AUDUSD', 0.66),
                ('USDCAD', 1.35),
                ('USDCHF', 0.88),
                ('EURGBP', 0.86),
                ('EURJPY', 163.5)
            ]
            
            for pair, rate in sample_crosses:
                cross_rate = CrossRate(
                    pair=pair,
                    rate=rate,
                    inverse_rate=1.0 / rate,
                    timestamp=datetime.now().isoformat()
                )
                cross_rates.append(cross_rate)
            
            return cross_rates
            
        except Exception as e:
            logger.error(f"Error fetching cross rates: {str(e)}")
            return []
    
    def convert_currency(self, amount: float, from_currency: str, to_currency: str) -> Optional[float]:
        """Convert amount from one currency to another."""
        try:
            if from_currency == to_currency:
                return amount
            
            # Direct pair
            direct_pair = f"{from_currency}{to_currency}"
            if direct_pair in self.major_pairs:
                rate = self.get_forex_rate(direct_pair)
                if rate:
                    return amount * rate.mid_rate
            
            # Inverse pair
            inverse_pair = f"{to_currency}{from_currency}"
            if inverse_pair in self.major_pairs:
                rate = self.get_forex_rate(inverse_pair)
                if rate:
                    return amount / rate.mid_rate
            
            # Cross conversion via USD
            if from_currency != 'USD' and to_currency != 'USD':
                # Convert from_currency to USD
                usd_amount = self.convert_currency(amount, from_currency, 'USD')
                if usd_amount:
                    # Convert USD to to_currency
                    return self.convert_currency(usd_amount, 'USD', to_currency)
            
            return None
            
        except Exception as e:
            logger.error(f"Error converting currency: {str(e)}")
            return None
    
    def get_currency_pairs(self) -> Dict[str, Dict[str, str]]:
        """Get list of available currency pairs."""
        return self.major_pairs
    
    def get_cross_pairs(self) -> List[str]:
        """Get list of available cross currency pairs."""
        return self.cross_pairs
    
    def search_currency_pairs(self, query: str) -> List[ForexRate]:
        """Search currency pairs by currency code."""
        try:
            all_rates = self.get_all_forex_rates()
            return [r for r in all_rates if query.upper() in r.pair]
        except Exception as e:
            logger.error(f"Error searching currency pairs: {str(e)}")
            return []


# Global forex provider instance
_forex_provider = None


def get_forex_provider() -> IndianForexProvider:
    """Get the global forex provider instance."""
    global _forex_provider
    if _forex_provider is None:
        _forex_provider = IndianForexProvider()
    return _forex_provider 