import pytz
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class IndianMarketCalendar:
    """Indian stock market calendar with trading hours and holidays."""
    
    def __init__(self):
        self.name = "Indian Market Calendar"
        self.ist_timezone = pytz.timezone('Asia/Kolkata')
        
        # NSE/BSE trading hours (IST)
        self.market_open_time = time(9, 15)  # 9:15 AM
        self.market_close_time = time(15, 30)  # 3:30 PM
        
        # Pre-market and after-market hours
        self.pre_market_start = time(9, 0)   # 9:00 AM
        self.pre_market_end = time(9, 15)    # 9:15 AM
        self.after_market_start = time(15, 30)  # 3:30 PM
        self.after_market_end = time(16, 0)   # 4:00 PM
        
        # Indian market holidays for 2024-2025 (approximate)
        # Note: These should be updated annually from official NSE/BSE sources
        self.market_holidays = {
            2024: [
                date(2024, 1, 26),  # Republic Day
                date(2024, 3, 8),   # Holi
                date(2024, 3, 29),  # Good Friday
                date(2024, 4, 11),  # Id-Ul-Fitr
                date(2024, 4, 17),  # Ram Navami
                date(2024, 5, 1),   # Maharashtra Day
                date(2024, 6, 17),  # Bakri Id
                date(2024, 8, 15),  # Independence Day
                date(2024, 8, 26),  # Janmashtami
                date(2024, 10, 2),  # Gandhi Jayanti
                date(2024, 11, 1),  # Diwali Balipratipada
                date(2024, 11, 15), # Guru Nanak Jayanti
                date(2024, 12, 25), # Christmas
            ],
            2025: [
                date(2025, 1, 26),  # Republic Day
                date(2025, 3, 14),  # Holi
                date(2025, 4, 18),  # Good Friday
                date(2025, 5, 1),   # Maharashtra Day
                date(2025, 8, 15),  # Independence Day
                date(2025, 10, 2),  # Gandhi Jayanti
                date(2025, 12, 25), # Christmas
                # Note: Add more holidays as they are announced
            ]
        }
    
    def get_current_ist_time(self) -> datetime:
        """Get current time in IST."""
        return datetime.now(self.ist_timezone)
    
    def is_market_day(self, check_date: Optional[date] = None) -> bool:
        """Check if given date is a market trading day."""
        if check_date is None:
            check_date = self.get_current_ist_time().date()
        
        # Check if it's a weekend
        if check_date.weekday() >= 5:  # Saturday (5) or Sunday (6)
            return False
        
        # Check if it's a market holiday
        year = check_date.year
        if year in self.market_holidays:
            if check_date in self.market_holidays[year]:
                return False
        
        return True
    
    def is_market_open(self, check_time: Optional[datetime] = None) -> bool:
        """Check if market is currently open."""
        if check_time is None:
            check_time = self.get_current_ist_time()
        
        # Convert to IST if timezone-aware
        if check_time.tzinfo is not None:
            check_time = check_time.astimezone(self.ist_timezone)
        
        # Check if it's a market day
        if not self.is_market_day(check_time.date()):
            return False
        
        # Check if current time is within market hours
        current_time = check_time.time()
        return self.market_open_time <= current_time <= self.market_close_time
    
    def is_pre_market_open(self, check_time: Optional[datetime] = None) -> bool:
        """Check if pre-market session is open."""
        if check_time is None:
            check_time = self.get_current_ist_time()
        
        if check_time.tzinfo is not None:
            check_time = check_time.astimezone(self.ist_timezone)
        
        if not self.is_market_day(check_time.date()):
            return False
        
        current_time = check_time.time()
        return self.pre_market_start <= current_time < self.pre_market_end
    
    def is_after_market_open(self, check_time: Optional[datetime] = None) -> bool:
        """Check if after-market session is open."""
        if check_time is None:
            check_time = self.get_current_ist_time()
        
        if check_time.tzinfo is not None:
            check_time = check_time.astimezone(self.ist_timezone)
        
        if not self.is_market_day(check_time.date()):
            return False
        
        current_time = check_time.time()
        return self.after_market_start <= current_time <= self.after_market_end
    
    def get_market_status(self) -> Dict[str, Any]:
        """Get comprehensive market status information."""
        now = self.get_current_ist_time()
        
        return {
            'current_time_ist': now.isoformat(),
            'is_market_day': self.is_market_day(),
            'is_market_open': self.is_market_open(),
            'is_pre_market_open': self.is_pre_market_open(),
            'is_after_market_open': self.is_after_market_open(),
            'market_session': self._get_current_session(),
            'next_market_open': self.get_next_market_open().isoformat(),
            'next_market_close': self.get_next_market_close().isoformat(),
            'time_to_open': self._get_time_until_market_open(),
            'time_to_close': self._get_time_until_market_close()
        }
    
    def _get_current_session(self) -> str:
        """Get current market session status."""
        if self.is_market_open():
            return "market_open"
        elif self.is_pre_market_open():
            return "pre_market"
        elif self.is_after_market_open():
            return "after_market"
        else:
            return "market_closed"
    
    def get_next_market_open(self) -> datetime:
        """Get the next market opening time."""
        now = self.get_current_ist_time()
        
        # If market is currently open or in pre-market, return today's open
        if self.is_market_open() or self.is_pre_market_open():
            return now.replace(hour=9, minute=15, second=0, microsecond=0)
        
        # Otherwise, find next market day
        check_date = now.date()
        if now.time() >= self.market_close_time:
            check_date += timedelta(days=1)
        
        while not self.is_market_day(check_date):
            check_date += timedelta(days=1)
        
        return self.ist_timezone.localize(
            datetime.combine(check_date, self.market_open_time)
        )
    
    def get_next_market_close(self) -> datetime:
        """Get the next market closing time."""
        now = self.get_current_ist_time()
        
        # If market is open, return today's close
        if self.is_market_open():
            return now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        # Otherwise, find next market day
        check_date = now.date()
        if not self.is_market_day(check_date) or now.time() >= self.market_close_time:
            check_date += timedelta(days=1)
            while not self.is_market_day(check_date):
                check_date += timedelta(days=1)
        
        return self.ist_timezone.localize(
            datetime.combine(check_date, self.market_close_time)
        )
    
    def _get_time_until_market_open(self) -> Optional[str]:
        """Get time remaining until market opens."""
        if self.is_market_open():
            return None
        
        now = self.get_current_ist_time()
        next_open = self.get_next_market_open()
        time_diff = next_open - now
        
        return self._format_timedelta(time_diff)
    
    def _get_time_until_market_close(self) -> Optional[str]:
        """Get time remaining until market closes."""
        if not self.is_market_open():
            return None
        
        now = self.get_current_ist_time()
        next_close = self.get_next_market_close()
        time_diff = next_close - now
        
        return self._format_timedelta(time_diff)
    
    def _format_timedelta(self, td: timedelta) -> str:
        """Format timedelta as human-readable string."""
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m"
        else:
            return f"{seconds}s"
    
    def get_trading_days_between(self, start_date: date, end_date: date) -> List[date]:
        """Get list of trading days between two dates."""
        trading_days = []
        current_date = start_date
        
        while current_date <= end_date:
            if self.is_market_day(current_date):
                trading_days.append(current_date)
            current_date += timedelta(days=1)
        
        return trading_days
    
    def get_market_holidays(self, year: int) -> List[date]:
        """Get list of market holidays for a given year."""
        return self.market_holidays.get(year, [])
    
    def add_market_holiday(self, holiday_date: date, description: str = ""):
        """Add a new market holiday."""
        year = holiday_date.year
        if year not in self.market_holidays:
            self.market_holidays[year] = []
        
        if holiday_date not in self.market_holidays[year]:
            self.market_holidays[year].append(holiday_date)
            self.market_holidays[year].sort()
            logger.info(f"Added market holiday: {holiday_date} - {description}")
    
    def get_market_timings(self) -> Dict[str, str]:
        """Get market timing information."""
        return {
            'pre_market': f"{self.pre_market_start.strftime('%H:%M')} - {self.pre_market_end.strftime('%H:%M')} IST",
            'regular_market': f"{self.market_open_time.strftime('%H:%M')} - {self.market_close_time.strftime('%H:%M')} IST",
            'after_market': f"{self.after_market_start.strftime('%H:%M')} - {self.after_market_end.strftime('%H:%M')} IST",
            'timezone': 'Asia/Kolkata (IST)',
            'trading_days': 'Monday to Friday (excluding holidays)'
        }


# Global market calendar instance
_market_calendar = None


def get_indian_market_calendar() -> IndianMarketCalendar:
    """Get the global Indian market calendar instance."""
    global _market_calendar
    if _market_calendar is None:
        _market_calendar = IndianMarketCalendar()
    return _market_calendar