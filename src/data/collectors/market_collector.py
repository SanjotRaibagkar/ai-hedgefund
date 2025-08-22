"""
Market Data Collector for AI Hedge Fund
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import yfinance as yf
from loguru import logger

from ..database.models import MarketData
from ...tools.enhanced_api import get_market_cap

class MarketDataCollector:
    """Collects market data including market cap, beta, and valuation metrics."""
    
    def __init__(self):
        """Initialize market data collector."""
        self.metrics = {
            'market_cap': self._get_market_cap,
            'beta': self._get_beta,
            'dividend_yield': self._get_dividend_yield,
            'payout_ratio': self._get_payout_ratio,
            'price_to_book': self._get_price_to_book,
            'price_to_sales': self._get_price_to_sales,
            'price_to_earnings': self._get_price_to_earnings,
            'forward_pe': self._get_forward_pe,
            'peg_ratio': self._get_peg_ratio
        }
    
    def collect_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Collect market data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with market data
        """
        try:
            logger.info(f"Collecting market data for {ticker} from {start_date} to {end_date}")
            
            # Get market data using yfinance
            stock = yf.Ticker(ticker)
            
            # Get info
            info = stock.info
            
            # Create market data record
            market_data = self._create_market_record(ticker, info, end_date)
            
            if market_data:
                df = pd.DataFrame([market_data])
                logger.info(f"Collected market data for {ticker}")
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to collect market data for {ticker}: {e}")
            return None
    
    def _create_market_record(self, ticker: str, info: Dict[str, Any], date: str) -> Optional[Dict[str, Any]]:
        """Create a market data record from stock info."""
        try:
            record = {
                'ticker': ticker,
                'date': datetime.strptime(date, '%Y-%m-%d').date(),
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            
            # Extract market metrics
            if info:
                # Market cap
                if 'marketCap' in info and info['marketCap']:
                    record['market_cap'] = info['marketCap']
                
                # Enterprise value
                if 'enterpriseValue' in info and info['enterpriseValue']:
                    record['enterprise_value'] = info['enterpriseValue']
                
                # Beta
                if 'beta' in info and info['beta']:
                    record['beta'] = info['beta']
                
                # Dividend yield
                if 'dividendYield' in info and info['dividendYield']:
                    record['dividend_yield'] = info['dividendYield']
                
                # Payout ratio
                if 'payoutRatio' in info and info['payoutRatio']:
                    record['payout_ratio'] = info['payoutRatio']
                
                # Price to book
                if 'priceToBook' in info and info['priceToBook']:
                    record['price_to_book'] = info['priceToBook']
                
                # Price to sales
                if 'priceToSalesTrailing12Months' in info and info['priceToSalesTrailing12Months']:
                    record['price_to_sales'] = info['priceToSalesTrailing12Months']
                
                # Price to earnings
                if 'trailingPE' in info and info['trailingPE']:
                    record['price_to_earnings'] = info['trailingPE']
                
                # Forward PE
                if 'forwardPE' in info and info['forwardPE']:
                    record['forward_pe'] = info['forwardPE']
                
                # PEG ratio
                if 'pegRatio' in info and info['pegRatio']:
                    record['peg_ratio'] = info['pegRatio']
            
            return record
            
        except Exception as e:
            logger.error(f"Failed to create market record: {e}")
            return None
    
    def _get_market_cap(self, ticker: str) -> Optional[float]:
        """Get market capitalization."""
        try:
            market_cap = get_market_cap(ticker, datetime.now().strftime('%Y-%m-%d'))
            return market_cap if market_cap else None
        except Exception as e:
            logger.error(f"Failed to get market cap for {ticker}: {e}")
            return None
    
    def _get_beta(self, ticker: str) -> Optional[float]:
        """Get beta coefficient."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get('beta', None)
        except Exception as e:
            logger.error(f"Failed to get beta for {ticker}: {e}")
            return None
    
    def _get_dividend_yield(self, ticker: str) -> Optional[float]:
        """Get dividend yield."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get('dividendYield', None)
        except Exception as e:
            logger.error(f"Failed to get dividend yield for {ticker}: {e}")
            return None
    
    def _get_payout_ratio(self, ticker: str) -> Optional[float]:
        """Get payout ratio."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get('payoutRatio', None)
        except Exception as e:
            logger.error(f"Failed to get payout ratio for {ticker}: {e}")
            return None
    
    def _get_price_to_book(self, ticker: str) -> Optional[float]:
        """Get price to book ratio."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get('priceToBook', None)
        except Exception as e:
            logger.error(f"Failed to get price to book for {ticker}: {e}")
            return None
    
    def _get_price_to_sales(self, ticker: str) -> Optional[float]:
        """Get price to sales ratio."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get('priceToSalesTrailing12Months', None)
        except Exception as e:
            logger.error(f"Failed to get price to sales for {ticker}: {e}")
            return None
    
    def _get_price_to_earnings(self, ticker: str) -> Optional[float]:
        """Get price to earnings ratio."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get('trailingPE', None)
        except Exception as e:
            logger.error(f"Failed to get price to earnings for {ticker}: {e}")
            return None
    
    def _get_forward_pe(self, ticker: str) -> Optional[float]:
        """Get forward price to earnings ratio."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get('forwardPE', None)
        except Exception as e:
            logger.error(f"Failed to get forward PE for {ticker}: {e}")
            return None
    
    def _get_peg_ratio(self, ticker: str) -> Optional[float]:
        """Get PEG ratio."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get('pegRatio', None)
        except Exception as e:
            logger.error(f"Failed to get PEG ratio for {ticker}: {e}")
            return None
    
    def validate_market_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate market data quality."""
        validation_result = {
            'total_records': len(df),
            'missing_values': {},
            'data_quality_score': 0.0,
            'issues': [],
            'metric_coverage': {}
        }
        
        # Check for missing values
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                validation_result['missing_values'][column] = missing_count
        
        # Check for negative values where inappropriate
        positive_metrics = ['market_cap', 'enterprise_value', 'price_to_book', 'price_to_sales', 'price_to_earnings']
        for col in positive_metrics:
            if col in df.columns:
                negative_values = (df[col] < 0).sum()
                if negative_values > 0:
                    validation_result['issues'].append(f"Found {negative_values} negative values in {col}")
        
        # Check metric coverage
        metric_columns = ['market_cap', 'beta', 'dividend_yield', 'price_to_book', 'price_to_earnings']
        for col in metric_columns:
            if col in df.columns:
                coverage = (df[col].notna().sum() / len(df)) * 100
                validation_result['metric_coverage'][col] = f"{coverage:.1f}%"
        
        # Calculate data quality score
        total_cells = len(df) * len(df.columns)
        missing_cells = sum(validation_result['missing_values'].values())
        validation_result['data_quality_score'] = ((total_cells - missing_cells) / total_cells) * 100
        
        return validation_result
    
    def get_market_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for market data."""
        summary = {
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d') if not df.empty else None,
                'end': df['date'].max().strftime('%Y-%m-%d') if not df.empty else None,
                'total_records': len(df)
            },
            'market_metrics': {},
            'valuation_metrics': {},
            'metric_coverage': {}
        }
        
        if not df.empty:
            # Market metrics statistics
            market_columns = ['market_cap', 'enterprise_value', 'beta']
            for col in market_columns:
                if col in df.columns:
                    summary['market_metrics'][col] = {
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'mean': df[col].mean(),
                        'std': df[col].std()
                    }
            
            # Valuation metrics statistics
            valuation_columns = ['price_to_book', 'price_to_sales', 'price_to_earnings', 'forward_pe', 'peg_ratio']
            for col in valuation_columns:
                if col in df.columns:
                    summary['valuation_metrics'][col] = {
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'mean': df[col].mean(),
                        'std': df[col].std()
                    }
            
            # Metric coverage
            all_metrics = market_columns + valuation_columns + ['dividend_yield', 'payout_ratio']
            for col in all_metrics:
                if col in df.columns:
                    coverage = (df[col].notna().sum() / len(df)) * 100
                    summary['metric_coverage'][col] = f"{coverage:.1f}%"
        
        return summary 