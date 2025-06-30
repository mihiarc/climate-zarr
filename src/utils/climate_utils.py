"""Common utilities for climate data processing.

This module contains shared functions used across different calculator implementations
to reduce code duplication and ensure consistency.
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, List, Tuple, Any, Optional


def create_annual_record(
    county_info: Dict[str, Any],
    scenario: str,
    year: int,
    indicators: Dict[str, xr.DataArray],
    time_index: int
) -> Dict[str, Any]:
    """Create a single annual record from climate indicators.
    
    Args:
        county_info: Dictionary containing geoid, name, and state
        scenario: Climate scenario name
        year: Year for this record
        indicators: Dictionary of indicator DataArrays
        time_index: Index into the time dimension
        
    Returns:
        Dictionary containing the annual record
    """
    record = {
        'GEOID': county_info['geoid'],
        'NAME': county_info['name'],
        'STATE': county_info['state'],
        'scenario': scenario,
        'year': year
    }
    
    # Add indicator values with appropriate transformations
    for ind_name, ind_data in indicators.items():
        # Check if time_index is valid
        if hasattr(ind_data, 'time') and time_index >= len(ind_data.time):
            # Skip this indicator if index is out of bounds
            continue
            
        try:
            if ind_name == 'tg_mean':
                # Convert from Kelvin to Celsius
                record[ind_name + '_C'] = float(ind_data.isel(time=time_index).values) - 273.15
            elif ind_name == 'precip_accumulation':
                # Add units to column name
                record[ind_name + '_mm'] = float(ind_data.isel(time=time_index).values)
            elif ind_name in ['tx90p', 'tn10p']:
                # Convert count to percentage
                count = float(ind_data.isel(time=time_index).values)
                year_start = pd.Timestamp(f'{year}-01-01')
                year_end = pd.Timestamp(f'{year}-12-31')
                n_days = (year_end - year_start).days + 1
                percentage = (count / n_days) * 100
                record[ind_name + '_percent'] = percentage
            else:
                record[ind_name] = float(ind_data.isel(time=time_index).values)
        except IndexError:
            # Skip if we can't access this time index
            continue
    
    return record


def adjust_longitude_bounds(bounds: List[float]) -> Tuple[float, float, float, float]:
    """Adjust bounds for dateline crossing.
    
    Args:
        bounds: List of [min_lon, min_lat, max_lon, max_lat]
        
    Returns:
        Tuple of (min_lon, min_lat, max_lon, max_lat) with adjusted longitudes
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    
    # Convert longitude to 0-360 if needed
    if min_lon < 0:
        min_lon = min_lon % 360
    if max_lon < 0:
        max_lon = max_lon % 360
    
    return min_lon, min_lat, max_lon, max_lat


def calculate_area_weighted_mean(
    data: xr.Dataset,
    lat_dim: str = 'lat',
    lon_dim: str = 'lon'
) -> xr.Dataset:
    """Calculate area-weighted mean for spatial data.
    
    Args:
        data: xarray Dataset or DataArray with spatial dimensions
        lat_dim: Name of latitude dimension
        lon_dim: Name of longitude dimension
        
    Returns:
        Dataset with spatial dimensions reduced using area weighting
    """
    # Calculate weights based on cosine of latitude
    weights = np.cos(np.deg2rad(data[lat_dim]))
    
    # Apply weighted mean
    weighted_data = data.weighted(weights)
    result = weighted_data.mean(dim=[lat_dim, lon_dim])
    
    # Preserve attributes (including units) for each variable
    for var in result.data_vars:
        if var in data.data_vars and 'units' in data[var].attrs:
            result[var].attrs['units'] = data[var].attrs['units']
    
    return result


def format_results_for_county(
    county_info: Dict[str, Any],
    scenarios: List[str],
    indicators: Dict[str, Dict[str, xr.DataArray]],
    years: np.ndarray
) -> List[Dict[str, Any]]:
    """Format calculation results into records for a single county.
    
    Args:
        county_info: Dictionary containing geoid, name, and state
        scenarios: List of scenario names
        indicators: Nested dict of {scenario: {indicator: DataArray}}
        years: Array of years
        
    Returns:
        List of annual records for all scenarios
    """
    all_records = []
    
    for scenario in scenarios:
        if scenario not in indicators:
            continue
            
        scenario_indicators = indicators[scenario]
        
        # Get the actual length of data from one of the indicators
        max_time_steps = 0
        for ind_name, ind_data in scenario_indicators.items():
            if hasattr(ind_data, 'time'):
                max_time_steps = len(ind_data.time)
                break
        
        # Create annual records only for available time steps
        for i in range(min(len(years), max_time_steps)):
            record = create_annual_record(
                county_info=county_info,
                scenario=scenario,
                year=int(years[i]),
                indicators=scenario_indicators,
                time_index=i
            )
            all_records.append(record)
    
    return all_records


def validate_county_data(
    data: xr.Dataset,
    expected_vars: Optional[List[str]] = None,
    min_time_steps: int = 365
) -> Tuple[bool, Optional[str]]:
    """Validate county data for completeness and quality.
    
    Args:
        data: xarray Dataset to validate
        expected_vars: List of expected variable names
        min_time_steps: Minimum expected time steps
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if data is None:
        return False, "Data is None"
    
    # Check if it's empty
    if len(data.dims) == 0:
        return False, "Data has no dimensions"
    
    # Check expected variables
    if expected_vars:
        missing_vars = [var for var in expected_vars if var not in data.data_vars]
        if missing_vars:
            return False, f"Missing variables: {missing_vars}"
    
    # Check time dimension
    if 'time' in data.dims:
        if len(data.time) < min_time_steps:
            return False, f"Insufficient time steps: {len(data.time)} < {min_time_steps}"
    
    # Check for all NaN data
    for var in data.data_vars:
        if data[var].isnull().all():
            return False, f"Variable {var} contains all NaN values"
    
    return True, None


def get_percentile_doy_index(date: pd.Timestamp, window_days: int = 5) -> np.ndarray:
    """Get day-of-year indices for percentile calculation with window.
    
    Args:
        date: Date to calculate window around
        window_days: Number of days on each side of the date
        
    Returns:
        Array of day-of-year values for the window
    """
    doy = date.dayofyear
    window = np.arange(doy - window_days, doy + window_days + 1)
    
    # Handle year boundaries
    window = np.where(window < 1, window + 365, window)
    window = np.where(window > 365, window - 365, window)
    
    return window


def celsius_to_fahrenheit(temp_c: float) -> float:
    """Convert temperature from Celsius to Fahrenheit."""
    return temp_c * 9/5 + 32


def mm_to_inches(mm: float) -> float:
    """Convert precipitation from millimeters to inches."""
    return mm / 25.4