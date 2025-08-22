"""
Fundamental Data Collector for AI Hedge Fund
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import yfinance as yf
from loguru import logger

from ..database.models import FundamentalData
from ...tools.enhanced_api import get_financial_metrics, get_line_items

class FundamentalDataCollector:
    """Collects fundamental data including financial statements and ratios."""
    
    def __init__(self):
        """Initialize fundamental data collector."""
        self.ratio_calculators = {
            'debt_to_equity': self._calculate_debt_to_equity,
            'roe': self._calculate_roe,
            'roa': self._calculate_roa,
            'pe_ratio': self._calculate_pe_ratio,
            'pb_ratio': self._calculate_pb_ratio,
            'ps_ratio': self._calculate_ps_ratio
        }
    
    def collect_annual_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Collect annual fundamental data.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with annual fundamental data
        """
        try:
            logger.info(f"Collecting annual fundamental data for {ticker}")
            
            # Get financial metrics
            metrics = get_financial_metrics(ticker, end_date)
            if not metrics:
                logger.warning(f"No financial metrics found for {ticker}")
                return None
            
            # Get line items
            line_items = get_line_items(ticker, end_date)
            
            # Create fundamental data record
            fundamental_data = self._create_fundamental_record(
                ticker, metrics, line_items, 'annual', end_date
            )
            
            if fundamental_data:
                df = pd.DataFrame([fundamental_data])
                logger.info(f"Collected annual fundamental data for {ticker}")
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to collect annual fundamental data for {ticker}: {e}")
            return None
    
    def collect_quarterly_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Collect quarterly fundamental data.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with quarterly fundamental data
        """
        try:
            logger.info(f"Collecting quarterly fundamental data for {ticker}")
            
            # For quarterly data, we'll use yfinance as fallback
            stock = yf.Ticker(ticker)
            
            # Get quarterly financials
            quarterly_data = []
            
            # Get quarterly earnings
            earnings = stock.quarterly_earnings
            if not earnings.empty:
                for date, row in earnings.iterrows():
                    if start_date <= date.strftime('%Y-%m-%d') <= end_date:
                        quarterly_data.append({
                            'ticker': ticker,
                            'date': date.date(),
                            'period_type': 'quarterly',
                            'revenue': row.get('Revenue', np.nan),
                            'net_income': row.get('Earnings', np.nan),
                            'created_at': datetime.now(),
                            'updated_at': datetime.now()
                        })
            
            # Get quarterly financials
            financials = stock.quarterly_financials
            if not financials.empty:
                # Process financial statements
                for date in financials.columns:
                    if start_date <= date.strftime('%Y-%m-%d') <= end_date:
                        financial_row = financials[date]
                        
                        # Extract key metrics
                        revenue = financial_row.get('Total Revenue', np.nan)
                        net_income = financial_row.get('Net Income', np.nan)
                        total_assets = financial_row.get('Total Assets', np.nan)
                        total_equity = financial_row.get('Total Stockholder Equity', np.nan)
                        
                        quarterly_data.append({
                            'ticker': ticker,
                            'date': date.date(),
                            'period_type': 'quarterly',
                            'revenue': revenue,
                            'net_income': net_income,
                            'total_assets': total_assets,
                            'total_equity': total_equity,
                            'created_at': datetime.now(),
                            'updated_at': datetime.now()
                        })
            
            if quarterly_data:
                df = pd.DataFrame(quarterly_data)
                df = df.drop_duplicates(subset=['date'])
                df = df.sort_values('date')
                
                # Calculate ratios
                df = self._calculate_ratios(df)
                
                logger.info(f"Collected {len(df)} quarterly fundamental records for {ticker}")
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to collect quarterly fundamental data for {ticker}: {e}")
            return None
    
    def _create_fundamental_record(self, ticker: str, metrics: Any, line_items: Any, 
                                 period_type: str, date: str) -> Optional[Dict[str, Any]]:
        """Create a fundamental data record from metrics and line items."""
        try:
            record = {
                'ticker': ticker,
                'date': datetime.strptime(date, '%Y-%m-%d').date(),
                'period_type': period_type,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            
            # Extract metrics if available
            if metrics:
                if hasattr(metrics, 'price_to_earnings_ratio'):
                    record['pe_ratio'] = metrics.price_to_earnings_ratio
                if hasattr(metrics, 'price_to_book_ratio'):
                    record['pb_ratio'] = metrics.price_to_book_ratio
                if hasattr(metrics, 'price_to_sales_ratio'):
                    record['ps_ratio'] = metrics.price_to_sales_ratio
                if hasattr(metrics, 'return_on_equity'):
                    record['roe'] = metrics.return_on_equity
                if hasattr(metrics, 'return_on_assets'):
                    record['roa'] = metrics.return_on_assets
                if hasattr(metrics, 'debt_to_equity_ratio'):
                    record['debt_to_equity'] = metrics.debt_to_equity_ratio
                if hasattr(metrics, 'dividend_yield'):
                    record['dividend_yield'] = metrics.dividend_yield
            
            # Extract line items if available
            if line_items:
                # Income statement items
                if hasattr(line_items, 'revenue'):
                    record['revenue'] = line_items.revenue
                if hasattr(line_items, 'net_income'):
                    record['net_income'] = line_items.net_income
                
                # Balance sheet items
                if hasattr(line_items, 'total_assets'):
                    record['total_assets'] = line_items.total_assets
                if hasattr(line_items, 'total_liabilities'):
                    record['total_liabilities'] = line_items.total_liabilities
                if hasattr(line_items, 'total_equity'):
                    record['total_equity'] = line_items.total_equity
                
                # Cash flow items
                if hasattr(line_items, 'operating_cash_flow'):
                    record['operating_cash_flow'] = line_items.operating_cash_flow
                if hasattr(line_items, 'free_cash_flow'):
                    record['free_cash_flow'] = line_items.free_cash_flow
            
            return record
            
        except Exception as e:
            logger.error(f"Failed to create fundamental record: {e}")
            return None
    
    def _calculate_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate financial ratios for the dataframe."""
        try:
            # Calculate debt to equity ratio
            if 'total_liabilities' in df.columns and 'total_equity' in df.columns:
                df['debt_to_equity'] = df['total_liabilities'] / df['total_equity']
            
            # Calculate ROE
            if 'net_income' in df.columns and 'total_equity' in df.columns:
                df['roe'] = df['net_income'] / df['total_equity']
            
            # Calculate ROA
            if 'net_income' in df.columns and 'total_assets' in df.columns:
                df['roa'] = df['net_income'] / df['total_assets']
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to calculate ratios: {e}")
            return df
    
    def _calculate_debt_to_equity(self, total_liabilities: float, total_equity: float) -> float:
        """Calculate debt to equity ratio."""
        if total_equity and total_equity != 0:
            return total_liabilities / total_equity
        return np.nan
    
    def _calculate_roe(self, net_income: float, total_equity: float) -> float:
        """Calculate return on equity."""
        if total_equity and total_equity != 0:
            return net_income / total_equity
        return np.nan
    
    def _calculate_roa(self, net_income: float, total_assets: float) -> float:
        """Calculate return on assets."""
        if total_assets and total_assets != 0:
            return net_income / total_assets
        return np.nan
    
    def _calculate_pe_ratio(self, market_price: float, earnings_per_share: float) -> float:
        """Calculate price to earnings ratio."""
        if earnings_per_share and earnings_per_share != 0:
            return market_price / earnings_per_share
        return np.nan
    
    def _calculate_pb_ratio(self, market_price: float, book_value_per_share: float) -> float:
        """Calculate price to book ratio."""
        if book_value_per_share and book_value_per_share != 0:
            return market_price / book_value_per_share
        return np.nan
    
    def _calculate_ps_ratio(self, market_price: float, sales_per_share: float) -> float:
        """Calculate price to sales ratio."""
        if sales_per_share and sales_per_share != 0:
            return market_price / sales_per_share
        return np.nan
    
    def validate_fundamental_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate fundamental data quality."""
        validation_result = {
            'total_records': len(df),
            'missing_values': {},
            'data_quality_score': 0.0,
            'issues': [],
            'ratio_coverage': {}
        }
        
        # Check for missing values
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                validation_result['missing_values'][column] = missing_count
        
        # Check for negative values where inappropriate
        financial_columns = ['revenue', 'net_income', 'total_assets', 'total_equity']
        for col in financial_columns:
            if col in df.columns:
                negative_values = (df[col] < 0).sum()
                if negative_values > 0:
                    validation_result['issues'].append(f"Found {negative_values} negative values in {col}")
        
        # Check ratio coverage
        ratio_columns = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'roe', 'roa', 'debt_to_equity']
        for col in ratio_columns:
            if col in df.columns:
                coverage = (df[col].notna().sum() / len(df)) * 100
                validation_result['ratio_coverage'][col] = f"{coverage:.1f}%"
        
        # Calculate data quality score
        total_cells = len(df) * len(df.columns)
        missing_cells = sum(validation_result['missing_values'].values())
        validation_result['data_quality_score'] = ((total_cells - missing_cells) / total_cells) * 100
        
        return validation_result
    
    def get_fundamental_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for fundamental data."""
        summary = {
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d') if not df.empty else None,
                'end': df['date'].max().strftime('%Y-%m-%d') if not df.empty else None,
                'total_periods': len(df)
            },
            'financial_metrics': {},
            'ratio_statistics': {},
            'period_type_distribution': {}
        }
        
        if not df.empty:
            # Financial metrics statistics
            financial_columns = ['revenue', 'net_income', 'total_assets', 'total_equity']
            for col in financial_columns:
                if col in df.columns:
                    summary['financial_metrics'][col] = {
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'mean': df[col].mean(),
                        'std': df[col].std()
                    }
            
            # Ratio statistics
            ratio_columns = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'roe', 'roa', 'debt_to_equity']
            for col in ratio_columns:
                if col in df.columns:
                    summary['ratio_statistics'][col] = {
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'mean': df[col].mean(),
                        'std': df[col].std()
                    }
            
            # Period type distribution
            if 'period_type' in df.columns:
                period_counts = df['period_type'].value_counts()
                summary['period_type_distribution'] = period_counts.to_dict()
        
        return summary 