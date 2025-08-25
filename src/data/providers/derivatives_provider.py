import requests
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from src.data.providers.base_provider import DataProviderError
from src.nsedata.NseUtility import NseUtils

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
        
        # Initialize NSE utility for real data
        try:
            self.nse = NseUtils()
            logger.info("✅ NSE utility initialized for real derivatives data")
        except Exception as e:
            logger.error(f"❌ Failed to initialize NSE utility: {e}")
            self.nse = None
        
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
        """Get complete option chain for an underlying using real NSE data."""
        cache_key = f"option_chain_{underlying}_{expiry_date}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            self._rate_limit()
            
            # Check if NSE utility is available
            if not self.nse:
                logger.error("❌ NSE utility not available, cannot fetch real data")
                return None
            
            # Validate underlying
            if underlying not in self.major_underlyings:
                logger.warning(f"Unknown underlying: {underlying}")
                return None
            
            logger.info(f"🔄 Fetching real option chain data for {underlying}")
            
            # Get real option chain data from NSE
            options_df = self.nse.get_live_option_chain(underlying, expiry_date, oi_mode="full", indices=True)
            
            if options_df is None or options_df.empty:
                logger.warning(f"❌ No option chain data available for {underlying}")
                return None
            
            # Get real spot price from futures data
            try:
                futures_df = self.nse.futures_data(underlying, indices=True)
                if futures_df is not None and not futures_df.empty:
                    # Get the current month futures price as spot
                    spot_price = futures_df['lastPrice'].iloc[0] if 'lastPrice' in futures_df.columns else None
                    if spot_price is None:
                        # Fallback to first available price
                        price_columns = [col for col in futures_df.columns if 'price' in col.lower() or 'ltp' in col.lower()]
                        if price_columns:
                            spot_price = futures_df[price_columns[0]].iloc[0]
                        else:
                            spot_price = 0.0
                else:
                    spot_price = 0.0
            except Exception as e:
                logger.warning(f"⚠️ Could not fetch futures data for spot price: {e}")
                spot_price = 0.0
            
            # Convert DataFrame to OptionChain format
            call_options = []
            put_options = []
            
            for _, row in options_df.iterrows():
                # Create call option
                if row.get('CALLS_LTP', 0) > 0:
                    call_option = OptionContract(
                        symbol=f"{underlying}{row['Expiry_Date'].replace('-', '')}CE{int(row['Strike_Price'])}",
                        strike_price=float(row['Strike_Price']),
                        expiry_date=row['Expiry_Date'],
                        option_type='CE',
                        underlying=underlying,
                        last_price=float(row.get('CALLS_LTP', 0)),
                        bid=float(row.get('CALLS_Bid_Price', 0)),
                        ask=float(row.get('CALLS_Ask_Price', 0)),
                        volume=int(row.get('CALLS_Volume', 0)),
                        open_interest=int(row.get('CALLS_OI', 0)),
                        implied_volatility=float(row.get('CALLS_IV', 0)),
                        delta=0.0,  # NSE doesn't provide Greeks
                        gamma=0.0,
                        theta=0.0,
                        vega=0.0,
                        change=float(row.get('CALLS_Net_Chng', 0)),
                        change_percent=0.0,  # Calculate if needed
                        high=0.0,  # Not available in NSE data
                        low=0.0,
                        open=0.0
                    )
                    call_options.append(call_option)
                
                # Create put option
                if row.get('PUTS_LTP', 0) > 0:
                    put_option = OptionContract(
                        symbol=f"{underlying}{row['Expiry_Date'].replace('-', '')}PE{int(row['Strike_Price'])}",
                        strike_price=float(row['Strike_Price']),
                        expiry_date=row['Expiry_Date'],
                        option_type='PE',
                        underlying=underlying,
                        last_price=float(row.get('PUTS_LTP', 0)),
                        bid=float(row.get('PUTS_Bid_Price', 0)),
                        ask=float(row.get('PUTS_Ask_Price', 0)),
                        volume=int(row.get('PUTS_Volume', 0)),
                        open_interest=int(row.get('PUTS_OI', 0)),
                        implied_volatility=float(row.get('PUTS_IV', 0)),
                        delta=0.0,  # NSE doesn't provide Greeks
                        gamma=0.0,
                        theta=0.0,
                        vega=0.0,
                        change=float(row.get('PUTS_Net_Chng', 0)),
                        change_percent=0.0,  # Calculate if needed
                        high=0.0,  # Not available in NSE data
                        low=0.0,
                        open=0.0
                    )
                    put_options.append(put_option)
            
            # Create OptionChain object
            option_chain = OptionChain(
                underlying=underlying,
                expiry_date=expiry_date or options_df['Expiry_Date'].iloc[0] if not options_df.empty else None,
                spot_price=spot_price,
                call_options=call_options,
                put_options=put_options,
                timestamp=datetime.now().isoformat()
            )
            
            # Cache the result
            self._cache_data(cache_key, option_chain)
            
            logger.info(f"✅ Successfully fetched real option chain for {underlying}: {len(call_options)} calls, {len(put_options)} puts")
            return option_chain
            
        except Exception as e:
            logger.error(f"❌ Error fetching real option chain for {underlying}: {e}")
            return None
    
    def get_futures_contracts(self, underlying: str) -> List[FuturesContract]:
        """Get futures contracts for an underlying using real NSE data."""
        try:
            self._rate_limit()
            
            # Check if NSE utility is available
            if not self.nse:
                logger.error("❌ NSE utility not available, cannot fetch real futures data")
                return []
            
            # Validate underlying
            if underlying not in self.major_underlyings:
                logger.warning(f"Unknown underlying: {underlying}")
                return []
            
            logger.info(f"🔄 Fetching real futures contracts for {underlying}")
            
            # Get real futures data from NSE
            futures_df = self.nse.futures_data(underlying, indices=True)
            
            if futures_df is None or futures_df.empty:
                logger.warning(f"❌ No futures data available for {underlying}")
                return []
            
            contracts = []
            
            for _, row in futures_df.iterrows():
                try:
                    # Extract contract details
                    expiry_date = row.get('expiryDate', '')
                    last_price = float(row.get('lastPrice', 0))
                    bid_price = float(row.get('bidPrice', 0))
                    ask_price = float(row.get('askPrice', 0))
                    volume = int(row.get('totalTradedVolume', 0))
                    open_interest = int(row.get('openInterest', 0))
                    change = float(row.get('change', 0))
                    change_percent = float(row.get('pChange', 0))
                    high = float(row.get('dayHigh', 0))
                    low = float(row.get('dayLow', 0))
                    open_price = float(row.get('open', 0))
                    
                    # Calculate basis (difference from spot - simplified)
                    basis = 0.0  # Would need spot price for accurate calculation
                    
                    contract = FuturesContract(
                        symbol=row.get('identifier', f"{underlying}{expiry_date}"),
                        expiry_date=expiry_date,
                        underlying=underlying,
                        last_price=last_price,
                        bid=bid_price,
                        ask=ask_price,
                        volume=volume,
                        open_interest=open_interest,
                        change=change,
                        change_percent=change_percent,
                        high=high,
                        low=low,
                        open=open_price,
                        basis=basis
                    )
                    contracts.append(contract)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error processing futures contract row: {e}")
                    continue
            
            logger.info(f"✅ Successfully fetched {len(contracts)} real futures contracts for {underlying}")
            return contracts
            
        except Exception as e:
            logger.error(f"❌ Error fetching real futures contracts for {underlying}: {e}")
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