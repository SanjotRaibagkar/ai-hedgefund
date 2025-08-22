import requests
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json

logger = logging.getLogger(__name__)


class CurrencyProvider:
    """Provider for currency conversion rates, specifically INR/USD."""
    
    def __init__(self):
        self.name = "Currency Provider"
        self.session = requests.Session()
        self.rate_limit_delay = 1.0
        self.cache = {}
        self.cache_expiry = 300  # 5 minutes cache
        
        # Free currency APIs (in order of preference)
        self.apis = [
            {
                'name': 'ExchangeRate-API',
                'url': 'https://api.exchangerate-api.com/v4/latest/USD',
                'enabled': True
            },
            {
                'name': 'Fixer.io',
                'url': 'https://api.fixer.io/latest?base=USD&symbols=INR',
                'enabled': False  # Requires API key
            },
            {
                'name': 'CurrencyAPI',
                'url': 'https://api.currencyapi.com/v3/latest?apikey=YOUR_API_KEY&currencies=INR&base_currency=USD',
                'enabled': False  # Requires API key
            }
        ]
        
        # Fallback exchange rate (approximate)
        self.fallback_usd_inr = 83.0
        
        # Setup session headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        })
    
    def _rate_limit(self):
        """Implement rate limiting."""
        time.sleep(self.rate_limit_delay)
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key].get('timestamp', 0)
        return (time.time() - cache_time) < self.cache_expiry
    
    def _get_cached_rate(self, cache_key: str) -> Optional[float]:
        """Get cached exchange rate if valid."""
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key].get('rate')
        return None
    
    def _cache_rate(self, cache_key: str, rate: float):
        """Cache exchange rate with timestamp."""
        self.cache[cache_key] = {
            'rate': rate,
            'timestamp': time.time()
        }
    
    def _fetch_from_exchangerate_api(self) -> Optional[float]:
        """Fetch USD to INR rate from ExchangeRate-API."""
        try:
            self._rate_limit()
            response = self.session.get(self.apis[0]['url'], timeout=10)
            response.raise_for_status()
            
            data = response.json()
            rates = data.get('rates', {})
            inr_rate = rates.get('INR')
            
            if inr_rate:
                logger.info(f"Fetched USD/INR rate from ExchangeRate-API: {inr_rate}")
                return float(inr_rate)
            
        except Exception as e:
            logger.error(f"Error fetching from ExchangeRate-API: {str(e)}")
        
        return None
    
    def _fetch_from_backup_sources(self) -> Optional[float]:
        """Try backup currency sources."""
        # Could implement other free APIs here
        # For now, return None to trigger fallback
        return None
    
    def get_usd_to_inr_rate(self) -> float:
        """Get current USD to INR exchange rate."""
        cache_key = 'USD_INR'
        
        # Check cache first
        cached_rate = self._get_cached_rate(cache_key)
        if cached_rate:
            return cached_rate
        
        # Try primary API
        rate = self._fetch_from_exchangerate_api()
        
        # Try backup sources if primary fails
        if not rate:
            rate = self._fetch_from_backup_sources()
        
        # Use fallback rate if all APIs fail
        if not rate:
            logger.warning(f"All currency APIs failed, using fallback rate: {self.fallback_usd_inr}")
            rate = self.fallback_usd_inr
        
        # Cache the rate
        self._cache_rate(cache_key, rate)
        
        return rate
    
    def get_inr_to_usd_rate(self) -> float:
        """Get current INR to USD exchange rate."""
        usd_inr_rate = self.get_usd_to_inr_rate()
        return 1.0 / usd_inr_rate
    
    def convert_usd_to_inr(self, usd_amount: float) -> float:
        """Convert USD amount to INR."""
        rate = self.get_usd_to_inr_rate()
        return usd_amount * rate
    
    def convert_inr_to_usd(self, inr_amount: float) -> float:
        """Convert INR amount to USD."""
        rate = self.get_inr_to_usd_rate()
        return inr_amount * rate
    
    def get_exchange_rate_info(self) -> Dict[str, Any]:
        """Get detailed exchange rate information."""
        try:
            usd_inr_rate = self.get_usd_to_inr_rate()
            inr_usd_rate = 1.0 / usd_inr_rate
            
            return {
                'USD_INR': usd_inr_rate,
                'INR_USD': inr_usd_rate,
                'timestamp': datetime.now().isoformat(),
                'source': 'ExchangeRate-API',
                'cache_status': 'cached' if self._is_cache_valid('USD_INR') else 'fresh'
            }
            
        except Exception as e:
            logger.error(f"Error getting exchange rate info: {str(e)}")
            return {
                'USD_INR': self.fallback_usd_inr,
                'INR_USD': 1.0 / self.fallback_usd_inr,
                'timestamp': datetime.now().isoformat(),
                'source': 'fallback',
                'error': str(e)
            }
    
    def normalize_market_cap(self, market_cap: float, from_currency: str, to_currency: str = 'USD') -> float:
        """Normalize market cap to target currency."""
        if from_currency == to_currency:
            return market_cap
        
        if from_currency == 'INR' and to_currency == 'USD':
            return self.convert_inr_to_usd(market_cap)
        elif from_currency == 'USD' and to_currency == 'INR':
            return self.convert_usd_to_inr(market_cap)
        else:
            logger.warning(f"Unsupported currency conversion: {from_currency} to {to_currency}")
            return market_cap
    
    def format_currency(self, amount: float, currency: str = 'INR') -> str:
        """Format currency amount with appropriate symbols and formatting."""
        if currency == 'INR':
            # Indian numbering system (crores, lakhs)
            if amount >= 10000000:  # 1 crore
                crores = amount / 10000000
                return f"₹{crores:.2f} Cr"
            elif amount >= 100000:  # 1 lakh
                lakhs = amount / 100000
                return f"₹{lakhs:.2f} L"
            else:
                return f"₹{amount:,.2f}"
        elif currency == 'USD':
            # US formatting
            if amount >= 1000000000:  # 1 billion
                billions = amount / 1000000000
                return f"${billions:.2f}B"
            elif amount >= 1000000:  # 1 million
                millions = amount / 1000000
                return f"${millions:.2f}M"
            else:
                return f"${amount:,.2f}"
        else:
            return f"{amount:,.2f} {currency}"


# Global currency provider instance
_currency_provider = None


def get_currency_provider() -> CurrencyProvider:
    """Get the global currency provider instance."""
    global _currency_provider
    if _currency_provider is None:
        _currency_provider = CurrencyProvider()
    return _currency_provider