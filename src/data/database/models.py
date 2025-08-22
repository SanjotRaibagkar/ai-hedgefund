"""
Data models for AI Hedge Fund database.
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
import pandas as pd

class TechnicalData(BaseModel):
    """Technical data model for stock price and indicators."""
    
    ticker: str = Field(..., description="Stock ticker symbol")
    trade_date: date = Field(..., description="Trading date")
    open_price: float = Field(..., description="Opening price")
    high_price: float = Field(..., description="High price")
    low_price: float = Field(..., description="Low price")
    close_price: float = Field(..., description="Closing price")
    volume: int = Field(..., description="Trading volume")
    adjusted_close: Optional[float] = Field(None, description="Adjusted closing price")
    
    # Technical indicators
    sma_20: Optional[float] = Field(None, description="20-day Simple Moving Average")
    sma_50: Optional[float] = Field(None, description="50-day Simple Moving Average")
    sma_200: Optional[float] = Field(None, description="200-day Simple Moving Average")
    rsi_14: Optional[float] = Field(None, description="14-day Relative Strength Index")
    macd: Optional[float] = Field(None, description="MACD line")
    macd_signal: Optional[float] = Field(None, description="MACD signal line")
    macd_histogram: Optional[float] = Field(None, description="MACD histogram")
    bollinger_upper: Optional[float] = Field(None, description="Bollinger Bands upper")
    bollinger_lower: Optional[float] = Field(None, description="Bollinger Bands lower")
    bollinger_middle: Optional[float] = Field(None, description="Bollinger Bands middle")
    atr_14: Optional[float] = Field(None, description="14-day Average True Range")
    
    # Metadata
    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(default_factory=datetime.now)
    
    @validator('volume')
    def validate_volume(cls, v):
        if v < 0:
            raise ValueError('Volume must be non-negative')
        return v
    
    @validator('open_price', 'high_price', 'low_price', 'close_price')
    def validate_prices(cls, v):
        if v <= 0:
            raise ValueError('Price must be positive')
        return v
    
    @validator('high_price')
    def validate_high_price(cls, v, values):
        if 'low_price' in values and v < values['low_price']:
            raise ValueError('High price cannot be less than low price')
        return v

class FundamentalData(BaseModel):
    """Fundamental data model for financial statements and ratios."""
    
    ticker: str = Field(..., description="Stock ticker symbol")
    report_date: date = Field(..., description="Financial period end date")
    period_type: str = Field(..., description="Period type: annual, quarterly, ttm")
    
    # Income Statement
    revenue: Optional[float] = Field(None, description="Total revenue")
    net_income: Optional[float] = Field(None, description="Net income")
    
    # Balance Sheet
    total_assets: Optional[float] = Field(None, description="Total assets")
    total_liabilities: Optional[float] = Field(None, description="Total liabilities")
    total_equity: Optional[float] = Field(None, description="Total equity")
    
    # Cash Flow
    operating_cash_flow: Optional[float] = Field(None, description="Operating cash flow")
    free_cash_flow: Optional[float] = Field(None, description="Free cash flow")
    
    # Financial Ratios
    debt_to_equity: Optional[float] = Field(None, description="Debt to equity ratio")
    roe: Optional[float] = Field(None, description="Return on equity")
    roa: Optional[float] = Field(None, description="Return on assets")
    pe_ratio: Optional[float] = Field(None, description="Price to earnings ratio")
    pb_ratio: Optional[float] = Field(None, description="Price to book ratio")
    ps_ratio: Optional[float] = Field(None, description="Price to sales ratio")
    dividend_yield: Optional[float] = Field(None, description="Dividend yield")
    
    # Market Data
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    enterprise_value: Optional[float] = Field(None, description="Enterprise value")
    
    # Metadata
    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(default_factory=datetime.now)
    
    @validator('period_type')
    def validate_period_type(cls, v):
        allowed_types = ['annual', 'quarterly', 'ttm']
        if v not in allowed_types:
            raise ValueError(f'Period type must be one of: {allowed_types}')
        return v

class MarketData(BaseModel):
    """Market data model for market-specific metrics."""
    
    ticker: str = Field(..., description="Stock ticker symbol")
    market_date: date = Field(..., description="Market data date")
    
    # Market Metrics
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    enterprise_value: Optional[float] = Field(None, description="Enterprise value")
    beta: Optional[float] = Field(None, description="Beta coefficient")
    
    # Valuation Metrics
    dividend_yield: Optional[float] = Field(None, description="Dividend yield")
    payout_ratio: Optional[float] = Field(None, description="Payout ratio")
    price_to_book: Optional[float] = Field(None, description="Price to book ratio")
    price_to_sales: Optional[float] = Field(None, description="Price to sales ratio")
    price_to_earnings: Optional[float] = Field(None, description="Price to earnings ratio")
    forward_pe: Optional[float] = Field(None, description="Forward P/E ratio")
    peg_ratio: Optional[float] = Field(None, description="PEG ratio")
    
    # Metadata
    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(default_factory=datetime.now)

class CorporateActions(BaseModel):
    """Corporate actions model for dividends, splits, etc."""
    
    ticker: str = Field(..., description="Stock ticker symbol")
    action_date: date = Field(..., description="Action announcement date")
    action_type: str = Field(..., description="Type of corporate action")
    description: Optional[str] = Field(None, description="Action description")
    value: Optional[float] = Field(None, description="Action value")
    ratio: Optional[str] = Field(None, description="Action ratio (e.g., 2:1 for splits)")
    
    # Important dates
    ex_date: Optional[date] = Field(None, description="Ex-dividend date")
    record_date: Optional[date] = Field(None, description="Record date")
    payment_date: Optional[date] = Field(None, description="Payment date")
    
    # Metadata
    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(default_factory=datetime.now)
    
    @validator('action_type')
    def validate_action_type(cls, v):
        allowed_types = ['dividend', 'split', 'bonus', 'rights', 'merger', 'acquisition']
        if v not in allowed_types:
            raise ValueError(f'Action type must be one of: {allowed_types}')
        return v

class DataQualityMetrics(BaseModel):
    """Data quality metrics model for monitoring data quality."""
    
    ticker: str = Field(..., description="Stock ticker symbol")
    quality_date: date = Field(..., description="Data date")
    data_type: str = Field(..., description="Type of data: technical, fundamental, market")
    
    # Quality scores (0-100)
    completeness_score: float = Field(..., ge=0, le=100, description="Data completeness score")
    accuracy_score: float = Field(..., ge=0, le=100, description="Data accuracy score")
    timeliness_score: float = Field(..., ge=0, le=100, description="Data timeliness score")
    consistency_score: float = Field(..., ge=0, le=100, description="Data consistency score")
    
    # Record counts
    total_records: int = Field(..., ge=0, description="Total expected records")
    missing_records: int = Field(..., ge=0, description="Number of missing records")
    error_count: int = Field(..., ge=0, description="Number of errors")
    
    # Metadata
    last_updated: Optional[datetime] = Field(default_factory=datetime.now)
    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    
    @validator('data_type')
    def validate_data_type(cls, v):
        allowed_types = ['technical', 'fundamental', 'market']
        if v not in allowed_types:
            raise ValueError(f'Data type must be one of: {allowed_types}')
        return v
    
    @property
    def overall_score(self) -> float:
        """Calculate overall quality score."""
        return (self.completeness_score + self.accuracy_score + 
                self.timeliness_score + self.consistency_score) / 4

class DataCollectionConfig(BaseModel):
    """Configuration for data collection."""
    
    ticker: str = Field(..., description="Stock ticker symbol")
    start_date: date = Field(..., description="Start date for data collection")
    end_date: date = Field(..., description="End date for data collection")
    data_types: List[str] = Field(default=['technical', 'fundamental'], 
                                 description="Types of data to collect")
    parallel: bool = Field(default=True, description="Use parallel processing")
    max_workers: int = Field(default=5, description="Maximum parallel workers")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    retry_delay: int = Field(default=5, description="Delay between retries (seconds)")
    
    @validator('data_types')
    def validate_data_types(cls, v):
        allowed_types = ['technical', 'fundamental', 'market', 'corporate_actions']
        for data_type in v:
            if data_type not in allowed_types:
                raise ValueError(f'Data type must be one of: {allowed_types}')
        return v
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        if 'start_date' in values and v < values['start_date']:
            raise ValueError('End date cannot be before start date')
        return v

class DataCollectionResult(BaseModel):
    """Result of data collection operation."""
    
    ticker: str = Field(..., description="Stock ticker symbol")
    success: bool = Field(..., description="Whether collection was successful")
    data_type: str = Field(..., description="Type of data collected")
    records_collected: int = Field(default=0, description="Number of records collected")
    records_expected: int = Field(default=0, description="Number of records expected")
    start_date: date = Field(..., description="Collection start date")
    end_date: date = Field(..., description="Collection end date")
    duration_seconds: float = Field(..., description="Collection duration in seconds")
    errors: List[str] = Field(default_factory=list, description="List of errors")
    warnings: List[str] = Field(default_factory=list, description="List of warnings")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.records_expected == 0:
            return 0.0
        return (self.records_collected / self.records_expected) * 100
    
    @property
    def is_complete(self) -> bool:
        """Check if collection is complete."""
        return self.success and self.records_collected >= self.records_expected 