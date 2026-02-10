"""Georgian public holidays calculation.

Includes fixed dates and Orthodox Easter (Julian calendar converted to Gregorian).
Algorithm valid for years 1900-2099.
"""

from datetime import date, timedelta
from typing import Set


def orthodox_easter(year: int) -> date:
    """
    Calculate Orthodox Easter date using Julian calendar algorithm, then convert to Gregorian.
    
    Orthodox Easter is calculated using the Julian calendar, then converted to Gregorian.
    For years 1900-2099, the algorithm uses the standard Orthodox Easter calculation.
    
    Algorithm:
    1. Calculate days after March 22 (Julian calendar) using Orthodox Easter formula
    2. Convert to Gregorian: March 22 (Julian) = April 4 (Gregorian) for 1900-2099
    3. Add the calculated days to get the Gregorian date
    
    Args:
        year: Year (1900-2099)
        
    Returns:
        Gregorian date of Orthodox Easter Sunday
    """
    # Orthodox Easter calculation (Julian calendar method)
    # This is the algorithm used by Orthodox churches
    a = year % 19
    b = year % 7
    c = year % 4
    
    d = (19 * a + 16) % 30
    e = (2 * c + 4 * b + 6 * d) % 7
    f = d + e  # Days after March 22 (Julian)
    
    # March 22 in Julian calendar = April 4 in Gregorian (for 1900-2099)
    # The offset is 13 days: March 22 + 13 = April 4
    base = date(year, 4, 4)  # Gregorian equivalent of March 22 (Julian)
    
    # Add f days, but subtract 1 to account for formula off-by-one
    easter = base + timedelta(days=f - 1)
    
    return easter


def georgian_holidays(year: int) -> Set[date]:
    """
    Get all Georgian public holidays for a given year.
    
    Fixed holidays:
    - Jan 1, Jan 2, Jan 7, Jan 19, Mar 3, Mar 8, Apr 9, May 9, May 12, May 26,
      Aug 28, Oct 14, Nov 23
    
    Movable (Orthodox Easter based):
    - Good Friday (Easter - 2)
    - Holy Saturday (Easter - 1)
    - Easter Sunday
    - Easter Monday (Easter + 1)
    
    Args:
        year: Year
        
    Returns:
        Set of holiday dates
    """
    holidays = set()
    
    # Fixed holidays
    fixed = [
        (1, 1),   # New Year
        (1, 2),   # New Year (2nd day)
        (1, 7),   # Orthodox Christmas
        (1, 19),  # Epiphany
        (3, 3),   # Mother's Day
        (3, 8),   # International Women's Day
        (4, 9),   # National Unity Day
        (5, 9),   # Victory Day
        (5, 12),  # St. Andrew's Day
        (5, 26),  # Independence Day
        (8, 28),  # Dormition of the Theotokos
        (10, 14), # Mtskhetoba
        (11, 23), # St. George's Day
    ]
    
    for month, day in fixed:
        holidays.add(date(year, month, day))
    
    # Movable holidays (Orthodox Easter)
    easter = orthodox_easter(year)
    holidays.add(easter - timedelta(days=2))  # Good Friday
    holidays.add(easter - timedelta(days=1))  # Holy Saturday
    holidays.add(easter)                       # Easter Sunday
    holidays.add(easter + timedelta(days=1))  # Easter Monday
    
    return holidays


def georgian_holidays_range(start_date: date, end_date: date) -> Set[date]:
    """
    Get all Georgian holidays in a date range.
    
    Args:
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        
    Returns:
        Set of holiday dates in the range
    """
    all_holidays = set()
    for year in range(start_date.year, end_date.year + 1):
        all_holidays.update(georgian_holidays(year))
    
    # Filter to range
    return {d for d in all_holidays if start_date <= d <= end_date}
