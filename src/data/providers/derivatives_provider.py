import requests
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from src.data.providers.base_provider import DataProviderError

logger = logging.getLogger(__name__)


@dataclass
class OptionContract:
    """Options contract data model."""
    symbol: str
    strike_price: float
    expiry_date: str
    option_type: str  # CE (Call), PE (Put)
    underlying: str  # NIFTY, BANKNIFTY, RELIANCE, etc.
    last_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    change: float
    change_percent: float
    high: float
    low: float
    open: float


@dataclass
class FuturesContract:
    """Futures contract data model."""
    symbol: str
    expiry_date: str
    underlying: str
    last_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    change: float
    change_percent: float
    high: float
    low: float
    open: float
    basis: float  # Difference from spot


@dataclass
class OptionChain:
    """Complete option chain for an underlying."""
    underlying: str
    expiry_date: str
    spot_price: float
    call_options: List[OptionContract]
    put_options: List[OptionContract]
    timestamp: str


class IndianDerivativesProvider:
    """Provider for Indian derivatives market data."""
    
    def __init__(self):
        self.name = "Indian Derivatives Provider"
        self.session = requests.Session()
        self.rate_limit_delay = 2.0
        self.cache = {}
        self.cache_expiry = 300  # 5 minutes cache
        
        # Setup session headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        })
        
        # Major underlyings
        self.major_underlyings = {
            'NIFTY': {
                'name': 'NIFTY 50',
                'type': 'Index',
                'lot_size': 50,
                'tick_size': 0.05
            },
            'BANKNIFTY': {
                'name': 'NIFTY BANK',
                'type': 'Index',
                'lot_size': 25,
                'tick_size': 0.05
            },
            'FINNIFTY': {
                'name': 'NIFTY FINANCIAL SERVICES',
                'type': 'Index',
                'lot_size': 40,
                'tick_size': 0.05
            },
            'RELIANCE': {
                'name': 'Reliance Industries',
                'type': 'Stock',
                'lot_size': 250,
                'tick_size': 0.05
            },
            'TCS': {
                'name': 'Tata Consultancy Services',
                'type': 'Stock',
                'lot_size': 300,
                'tick_size': 0.05
            },
            'INFY': {
                'name': 'Infosys',
                'type': 'Stock',
                'lot_size': 600,
                'tick_size': 0.05
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
    
    def get_option_chain(self, underlying: str, expiry_date: Optional[str] = None) -> Optional[OptionChain]:
        """Get complete option chain for an underlying."""
        cache_key = f"option_chain_{underlying}_{expiry_date}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            self._rate_limit()
            
            # Simulate option chain data
            # In production, this would fetch from NSE
            if underlying not in self.major_underlyings:
                logger.warning(f"Unknown underlying: {underlying}")
                return None
            
            # Generate expiry date if not provided
            if not expiry_date:
                # Next Thursday
                today = datetime.now()
                days_until_thursday = (3 - today.weekday()) % 7
                if days_until_thursday == 0:
                    days_until_thursday = 7
                expiry_date = (today + timedelta(days=days_until_thursday)).strftime('%Y-%m-%d')
            
            # Simulate spot price
            spot_prices = {
                'NIFTY': 22000.0,
                'BANKNIFTY': 48000.0,
                'FINNIFTY': 20000.0,
                'RELIANCE': 1424.80,
                'TCS': 3102.60,
                'INFY': 1496.40
            }
            spot_price = spot_prices.get(underlying, 1000.0)
            
            # Generate strike prices around spot
            strikes = []
            for i in range(-10, 11):  # 21 strikes around spot
                if underlying in ['NIFTY', 'BANKNIFTY', 'FINNIFTY']:
                    strike = spot_price + (i * 100)
                else:
                    strike = spot_price + (i * 10)
                strikes.append(strike)
            
            # Generate call options
            call_options = []
            for strike in strikes:
                if strike <= spot_price:
                    # ITM call
                    intrinsic_value = spot_price - strike
                    time_value = max(0, (strike / spot_price) * 50)
                else:
                    # OTM call
                    intrinsic_value = 0
                    time_value = max(0, (spot_price / strike) * 30)
                
                option_price = intrinsic_value + time_value
                
                # Calculate Greeks (simplified)
                delta = max(0, min(1, (spot_price - strike) / (spot_price * 0.1)))
                gamma = 0.001
                theta = -option_price * 0.01
                vega = option_price * 0.1
                
                call_option = OptionContract(
                    symbol=f"{underlying}{expiry_date.replace('-', '')}CE{int(strike)}",
                    strike_price=strike,
                    expiry_date=expiry_date,
                    option_type='CE',
                    underlying=underlying,
                    last_price=option_price,
                    bid=option_price * 0.99,
                    ask=option_price * 1.01,
                    volume=1000 + (abs(strike - spot_price) * 10),
                    open_interest=5000 + (abs(strike - spot_price) * 50),
                    implied_volatility=0.25 + (abs(strike - spot_price) / spot_price) * 0.1,
                    delta=delta,
                    gamma=gamma,
                    theta=theta,
                    vega=vega,
                    change=option_price * 0.02,
                    change_percent=2.0,
                    high=option_price * 1.05,
                    low=option_price * 0.95,
                    open=option_price * 1.02
                )
                call_options.append(call_option)
            
            # Generate put options
            put_options = []
            for strike in strikes:
                if strike >= spot_price:
                    # ITM put
                    intrinsic_value = strike - spot_price
                    time_value = max(0, (spot_price / strike) * 50)
                else:
                    # OTM put
                    intrinsic_value = 0
                    time_value = max(0, (strike / spot_price) * 30)
                
                option_price = intrinsic_value + time_value
                
                # Calculate Greeks (simplified)
                delta = min(0, max(-1, (strike - spot_price) / (spot_price * 0.1)))
                gamma = 0.001
                theta = -option_price * 0.01
                vega = option_price * 0.1
                
                put_option = OptionContract(
                    symbol=f"{underlying}{expiry_date.replace('-', '')}PE{int(strike)}",
                    strike_price=strike,
                    expiry_date=expiry_date,
                    option_type='PE',
                    underlying=underlying,
                    last_price=option_price,
                    bid=option_price * 0.99,
                    ask=option_price * 1.01,
                    volume=1000 + (abs(strike - spot_price) * 10),
                    open_interest=5000 + (abs(strike - spot_price) * 50),
                    implied_volatility=0.25 + (abs(strike - spot_price) / spot_price) * 0.1,
                    delta=delta,
                    gamma=gamma,
                    theta=theta,
                    vega=vega,
                    change=option_price * 0.02,
                    change_percent=2.0,
                    high=option_price * 1.05,
                    low=option_price * 0.95,
                    open=option_price * 1.02
                )
                put_options.append(put_option)
            
            option_chain = OptionChain(
                underlying=underlying,
                expiry_date=expiry_date,
                spot_price=spot_price,
                call_options=call_options,
                put_options=put_options,
                timestamp=datetime.now().isoformat()
            )
            
            self._cache_data(cache_key, option_chain)
            return option_chain
            
        except Exception as e:
            logger.error(f"Error fetching option chain for {underlying}: {str(e)}")
            return None
    
    def get_futures_contracts(self, underlying: str) -> List[FuturesContract]:
        """Get futures contracts for an underlying."""
        try:
            self._rate_limit()
            
            # Simulate futures contracts
            contracts = []
            spot_prices = {
                'NIFTY': 22000.0,
                'BANKNIFTY': 48000.0,
                'FINNIFTY': 20000.0,
                'RELIANCE': 1424.80,
                'TCS': 3102.60,
                'INFY': 1496.40
            }
            spot_price = spot_prices.get(underlying, 1000.0)
            
            # Generate contracts for next 3 months
            for i in range(3):
                contract_date = datetime.now() + timedelta(days=30 * (i + 1))
                expiry_date = contract_date.strftime('%Y-%m-%d')
                
                # Simulate futures pricing (contango)
                futures_price = spot_price * (1 + (i * 0.005))  # 0.5% contango per month
                basis = futures_price - spot_price
                
                contract = FuturesContract(
                    symbol=f"{underlying}{expiry_date.replace('-', '')}",
                    expiry_date=expiry_date,
                    underlying=underlying,
                    last_price=futures_price,
                    bid=futures_price * 0.999,
                    ask=futures_price * 1.001,
                    volume=100000 + (i * 50000),
                    open_interest=50000 + (i * 25000),
                    change=futures_price * 0.01,
                    change_percent=1.0,
                    high=futures_price * 1.02,
                    low=futures_price * 0.98,
                    open=futures_price * 1.01,
                    basis=basis
                )
                contracts.append(contract)
            
            return contracts
            
        except Exception as e:
            logger.error(f"Error fetching futures contracts for {underlying}: {str(e)}")
            return []
    
    def get_option_contract(self, symbol: str) -> Optional[OptionContract]:
        """Get specific option contract details."""
        try:
            self._rate_limit()
            
            # Parse symbol to extract underlying and strike
            # Format: NIFTY20241226CE22000
            if len(symbol) < 12:
                return None
            
            # Extract components (simplified parsing)
            underlying = symbol[:6] if symbol.startswith('NIFTY') else symbol[:4]
            expiry_str = symbol[6:14] if symbol.startswith('NIFTY') else symbol[4:12]
            option_type = symbol[14:16] if symbol.startswith('NIFTY') else symbol[12:14]
            strike_str = symbol[16:] if symbol.startswith('NIFTY') else symbol[14:]
            
            try:
                expiry_date = f"{expiry_str[:4]}-{expiry_str[4:6]}-{expiry_str[6:8]}"
                strike_price = float(strike_str)
            except (ValueError, IndexError):
                return None
            
            # Get option chain and find the specific contract
            option_chain = self.get_option_chain(underlying, expiry_date)
            if not option_chain:
                return None
            
            # Find the contract
            if option_type == 'CE':
                for option in option_chain.call_options:
                    if abs(option.strike_price - strike_price) < 0.01:
                        return option
            elif option_type == 'PE':
                for option in option_chain.put_options:
                    if abs(option.strike_price - strike_price) < 0.01:
                        return option
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching option contract {symbol}: {str(e)}")
            return None
    
    def get_implied_volatility_smile(self, underlying: str, expiry_date: str) -> Dict[str, List[float]]:
        """Get implied volatility smile for an underlying."""
        try:
            option_chain = self.get_option_chain(underlying, expiry_date)
            if not option_chain:
                return {}
            
            strikes = []
            call_ivs = []
            put_ivs = []
            
            # Get strikes and IVs from call options
            for option in option_chain.call_options:
                strikes.append(option.strike_price)
                call_ivs.append(option.implied_volatility)
            
            # Get IVs from put options (same strikes)
            for option in option_chain.put_options:
                put_ivs.append(option.implied_volatility)
            
            return {
                'strikes': strikes,
                'call_ivs': call_ivs,
                'put_ivs': put_ivs
            }
            
        except Exception as e:
            logger.error(f"Error fetching IV smile for {underlying}: {str(e)}")
            return {}
    
    def get_underlyings(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available underlyings."""
        return self.major_underlyings
    
    def search_options(self, query: str) -> List[OptionContract]:
        """Search options by symbol or underlying."""
        try:
            # Simple search implementation
            results = []
            
            # Search in major underlyings
            for underlying in self.major_underlyings.keys():
                if query.upper() in underlying.upper():
                    option_chain = self.get_option_chain(underlying)
                    if option_chain:
                        results.extend(option_chain.call_options[:5])  # First 5 calls
                        results.extend(option_chain.put_options[:5])   # First 5 puts
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching options: {str(e)}")
            return []


# Global derivatives provider instance
_derivatives_provider = None


def get_derivatives_provider() -> IndianDerivativesProvider:
    """Get the global derivatives provider instance."""
    global _derivatives_provider
    if _derivatives_provider is None:
        _derivatives_provider = IndianDerivativesProvider()
    return _derivatives_provider 