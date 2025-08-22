import requests
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from src.data.providers.base_provider import DataProviderError

logger = logging.getLogger(__name__)


@dataclass
class Bond:
    """Bond data model."""
    bond_code: str
    bond_name: str
    issuer: str
    bond_type: str  # Government, Corporate, Municipal
    face_value: float
    coupon_rate: float
    coupon_frequency: str  # Semi-annual, Annual
    maturity_date: str
    issue_date: str
    current_price: float
    yield_to_maturity: float
    modified_duration: float
    credit_rating: str
    sector: Optional[str] = None
    liquidity: Optional[str] = None  # High, Medium, Low


@dataclass
class BondYield:
    """Bond yield data."""
    bond_code: str
    date: str
    yield_rate: float
    price: float
    volume: Optional[int] = None
    trades: Optional[int] = None


@dataclass
class YieldCurve:
    """Government bond yield curve."""
    date: str
    tenures: List[str]  # 3M, 6M, 1Y, 2Y, 5Y, 10Y, 15Y, 20Y, 30Y
    yields: List[float]
    source: str = "RBI"


class IndianBondProvider:
    """Provider for Indian bond market data."""
    
    def __init__(self):
        self.name = "Indian Bond Provider"
        self.session = requests.Session()
        self.rate_limit_delay = 2.0
        self.cache = {}
        self.cache_expiry = 1800  # 30 minutes cache
        
        # Setup session headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        })
        
        # Bond types
        self.bond_types = [
            'Government Securities',
            'Corporate Bonds',
            'State Development Loans',
            'Treasury Bills',
            'Municipal Bonds'
        ]
        
        # Credit ratings
        self.credit_ratings = [
            'AAA', 'AA+', 'AA', 'AA-',
            'A+', 'A', 'A-',
            'BBB+', 'BBB', 'BBB-',
            'BB+', 'BB', 'BB-',
            'B+', 'B', 'B-',
            'CCC+', 'CCC', 'CCC-',
            'CC', 'C', 'D'
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
    
    def get_government_bonds(self, limit: int = 20) -> List[Bond]:
        """Get list of government bonds."""
        cache_key = f"gov_bonds_{limit}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            self._rate_limit()
            
            # Simulate government bonds data
            # In production, this would fetch from RBI or NSE
            bonds = []
            
            # Sample government bonds
            sample_bonds = [
                {
                    'bond_code': 'GSEC2025',
                    'bond_name': '7.26% GOI 2025',
                    'issuer': 'Government of India',
                    'bond_type': 'Government Securities',
                    'face_value': 100.0,
                    'coupon_rate': 7.26,
                    'coupon_frequency': 'Semi-annual',
                    'maturity_date': '2025-01-15',
                    'issue_date': '2020-01-15',
                    'current_price': 98.45,
                    'yield_to_maturity': 7.85,
                    'modified_duration': 0.85,
                    'credit_rating': 'AAA',
                    'liquidity': 'High'
                },
                {
                    'bond_code': 'GSEC2030',
                    'bond_name': '6.54% GOI 2030',
                    'issuer': 'Government of India',
                    'bond_type': 'Government Securities',
                    'face_value': 100.0,
                    'coupon_rate': 6.54,
                    'coupon_frequency': 'Semi-annual',
                    'maturity_date': '2030-05-15',
                    'issue_date': '2020-05-15',
                    'current_price': 95.32,
                    'yield_to_maturity': 7.12,
                    'modified_duration': 6.45,
                    'credit_rating': 'AAA',
                    'liquidity': 'High'
                },
                {
                    'bond_code': 'GSEC2035',
                    'bond_name': '6.18% GOI 2035',
                    'issuer': 'Government of India',
                    'bond_type': 'Government Securities',
                    'face_value': 100.0,
                    'coupon_rate': 6.18,
                    'coupon_frequency': 'Semi-annual',
                    'maturity_date': '2035-08-15',
                    'issue_date': '2020-08-15',
                    'current_price': 92.18,
                    'yield_to_maturity': 6.95,
                    'modified_duration': 12.34,
                    'credit_rating': 'AAA',
                    'liquidity': 'Medium'
                }
            ]
            
            for bond_data in sample_bonds:
                bond = Bond(**bond_data)
                bonds.append(bond)
            
            bonds = bonds[:limit]
            self._cache_data(cache_key, bonds)
            return bonds
            
        except Exception as e:
            logger.error(f"Error fetching government bonds: {str(e)}")
            raise DataProviderError(f"Failed to fetch government bonds: {str(e)}")
    
    def get_corporate_bonds(self, rating: Optional[str] = None, limit: int = 20) -> List[Bond]:
        """Get list of corporate bonds."""
        cache_key = f"corp_bonds_{rating}_{limit}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            self._rate_limit()
            
            # Simulate corporate bonds data
            bonds = []
            
            # Sample corporate bonds
            sample_bonds = [
                {
                    'bond_code': 'RELIANCE2026',
                    'bond_name': 'Reliance Industries 7.5% 2026',
                    'issuer': 'Reliance Industries Limited',
                    'bond_type': 'Corporate Bonds',
                    'face_value': 1000.0,
                    'coupon_rate': 7.5,
                    'coupon_frequency': 'Annual',
                    'maturity_date': '2026-03-15',
                    'issue_date': '2021-03-15',
                    'current_price': 985.50,
                    'yield_to_maturity': 8.25,
                    'modified_duration': 1.85,
                    'credit_rating': 'AAA',
                    'sector': 'Oil & Gas',
                    'liquidity': 'Medium'
                },
                {
                    'bond_code': 'TCS2027',
                    'bond_name': 'TCS 6.8% 2027',
                    'issuer': 'Tata Consultancy Services Limited',
                    'bond_type': 'Corporate Bonds',
                    'face_value': 1000.0,
                    'coupon_rate': 6.8,
                    'coupon_frequency': 'Annual',
                    'maturity_date': '2027-06-15',
                    'issue_date': '2022-06-15',
                    'current_price': 975.20,
                    'yield_to_maturity': 7.45,
                    'modified_duration': 3.25,
                    'credit_rating': 'AAA',
                    'sector': 'IT',
                    'liquidity': 'Medium'
                },
                {
                    'bond_code': 'HDFCBANK2028',
                    'bond_name': 'HDFC Bank 7.2% 2028',
                    'issuer': 'HDFC Bank Limited',
                    'bond_type': 'Corporate Bonds',
                    'face_value': 1000.0,
                    'coupon_rate': 7.2,
                    'coupon_frequency': 'Annual',
                    'maturity_date': '2028-09-15',
                    'issue_date': '2023-09-15',
                    'current_price': 965.80,
                    'yield_to_maturity': 7.85,
                    'modified_duration': 4.12,
                    'credit_rating': 'AA+',
                    'sector': 'Banking',
                    'liquidity': 'Low'
                }
            ]
            
            for bond_data in sample_bonds:
                if rating is None or bond_data['credit_rating'] == rating:
                    bond = Bond(**bond_data)
                    bonds.append(bond)
            
            bonds = bonds[:limit]
            self._cache_data(cache_key, bonds)
            return bonds
            
        except Exception as e:
            logger.error(f"Error fetching corporate bonds: {str(e)}")
            raise DataProviderError(f"Failed to fetch corporate bonds: {str(e)}")
    
    def get_bond_details(self, bond_code: str) -> Optional[Bond]:
        """Get detailed information about a specific bond."""
        cache_key = f"bond_details_{bond_code}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            self._rate_limit()
            
            # Simulate bond details
            bond_data = {
                'bond_code': bond_code,
                'bond_name': f'Sample Bond {bond_code}',
                'issuer': 'Sample Issuer',
                'bond_type': 'Government Securities',
                'face_value': 100.0,
                'coupon_rate': 7.26,
                'coupon_frequency': 'Semi-annual',
                'maturity_date': '2025-01-15',
                'issue_date': '2020-01-15',
                'current_price': 98.45,
                'yield_to_maturity': 7.85,
                'modified_duration': 0.85,
                'credit_rating': 'AAA',
                'liquidity': 'High'
            }
            
            bond = Bond(**bond_data)
            self._cache_data(cache_key, bond)
            return bond
            
        except Exception as e:
            logger.error(f"Error fetching bond details for {bond_code}: {str(e)}")
            return None
    
    def get_bond_yields(self, bond_code: str, days: int = 30) -> List[BondYield]:
        """Get historical yield data for a bond."""
        try:
            self._rate_limit()
            
            # Simulate yield data
            yields = []
            base_yield = 7.85
            base_price = 98.45
            
            for i in range(days):
                date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
                # Simulate some yield variation
                yield_variation = (i % 7) * 0.1 - 0.3
                price_variation = yield_variation * -0.5
                
                yield_data = BondYield(
                    bond_code=bond_code,
                    date=date,
                    yield_rate=base_yield + yield_variation,
                    price=base_price + price_variation,
                    volume=1000000 + (i * 50000),
                    trades=50 + (i % 10)
                )
                yields.append(yield_data)
            
            return yields
            
        except Exception as e:
            logger.error(f"Error fetching bond yields for {bond_code}: {str(e)}")
            return []
    
    def get_yield_curve(self) -> YieldCurve:
        """Get current government bond yield curve."""
        cache_key = "yield_curve"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        try:
            self._rate_limit()
            
            # Simulate yield curve data
            # In production, this would fetch from RBI
            tenures = ['3M', '6M', '1Y', '2Y', '5Y', '10Y', '15Y', '20Y', '30Y']
            yields = [6.85, 7.05, 7.25, 7.45, 7.65, 7.85, 8.05, 8.25, 8.45]
            
            yield_curve = YieldCurve(
                date=datetime.now().strftime('%Y-%m-%d'),
                tenures=tenures,
                yields=yields,
                source="RBI"
            )
            
            self._cache_data(cache_key, yield_curve)
            return yield_curve
            
        except Exception as e:
            logger.error(f"Error fetching yield curve: {str(e)}")
            raise DataProviderError(f"Failed to fetch yield curve: {str(e)}")
    
    def search_bonds(self, query: str, bond_type: Optional[str] = None) -> List[Bond]:
        """Search bonds by name or issuer."""
        try:
            self._rate_limit()
            
            # Simulate bond search
            bonds = []
            
            # Sample search results
            sample_bonds = [
                {
                    'bond_code': 'GSEC2025',
                    'bond_name': '7.26% GOI 2025',
                    'issuer': 'Government of India',
                    'bond_type': 'Government Securities',
                    'face_value': 100.0,
                    'coupon_rate': 7.26,
                    'coupon_frequency': 'Semi-annual',
                    'maturity_date': '2025-01-15',
                    'issue_date': '2020-01-15',
                    'current_price': 98.45,
                    'yield_to_maturity': 7.85,
                    'modified_duration': 0.85,
                    'credit_rating': 'AAA',
                    'liquidity': 'High'
                }
            ]
            
            for bond_data in sample_bonds:
                if (query.lower() in bond_data['bond_name'].lower() or 
                    query.lower() in bond_data['issuer'].lower()):
                    if bond_type is None or bond_data['bond_type'] == bond_type:
                        bond = Bond(**bond_data)
                        bonds.append(bond)
            
            return bonds
            
        except Exception as e:
            logger.error(f"Error searching bonds: {str(e)}")
            return []
    
    def get_bond_types(self) -> List[str]:
        """Get list of available bond types."""
        return self.bond_types
    
    def get_credit_ratings(self) -> List[str]:
        """Get list of available credit ratings."""
        return self.credit_ratings


# Global bond provider instance
_bond_provider = None


def get_bond_provider() -> IndianBondProvider:
    """Get the global bond provider instance."""
    global _bond_provider
    if _bond_provider is None:
        _bond_provider = IndianBondProvider()
    return _bond_provider 