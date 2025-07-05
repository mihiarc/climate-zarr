#!/usr/bin/env python
"""Data processing utilities for climate statistics."""

import numpy as np
from typing import Dict, Any
from rich.console import Console

console = Console()


def convert_units(data: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
    """Convert data between different units.
    
    Args:
        data: Input data array
        from_unit: Source unit
        to_unit: Target unit
        
    Returns:
        Converted data array
    """
    # Precipitation conversions
    if from_unit == "kg/m2/s" and to_unit == "mm/day":
        # 1 kg/m²/s = 86400 mm/day (86400 seconds per day, 1 kg/m² = 1 mm)
        console.print("[yellow]Converting precipitation units from kg/m²/s to mm/day[/yellow]")
        return data * 86400
    
    # Temperature conversions
    elif from_unit == "K" and to_unit == "C":
        console.print("[yellow]Converting temperature units from Kelvin to Celsius[/yellow]")
        return data - 273.15
    
    elif from_unit == "F" and to_unit == "C":
        console.print("[yellow]Converting temperature units from Fahrenheit to Celsius[/yellow]")
        return (data - 32) * 5.0/9.0
    
    elif from_unit == "C" and to_unit == "F":
        console.print("[yellow]Converting temperature units from Celsius to Fahrenheit[/yellow]")
        return data * 9.0/5.0 + 32
    
    else:
        console.print(f"[yellow]No conversion needed from {from_unit} to {to_unit}[/yellow]")
        return data


def calculate_precipitation_stats(
    daily_values: np.ndarray,
    threshold_mm: float,
    year: int,
    scenario: str,
    county_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate precipitation statistics for a county.
    
    Args:
        daily_values: Array of daily precipitation values
        threshold_mm: Threshold for heavy precipitation days
        year: Year of the data
        scenario: Climate scenario name
        county_info: County metadata
        
    Returns:
        Dictionary of precipitation statistics
    """
    valid_days = daily_values[~np.isnan(daily_values)]
    
    if len(valid_days) == 0:
        return None
    
    return {
        'year': year,
        'scenario': scenario,
        'county_id': county_info['county_id'],
        'county_name': county_info['county_name'],
        'state': county_info['state'],
        'total_annual_precip_mm': float(np.sum(valid_days)),
        'days_above_threshold': int(np.sum(valid_days > threshold_mm)) if threshold_mm is not None else 0,
        'mean_daily_precip_mm': float(np.mean(valid_days)),
        'max_daily_precip_mm': float(np.max(valid_days)),
        'precip_std_mm': float(np.std(valid_days)),
        'dry_days': int(np.sum(valid_days < 0.1)),
        'wet_days': int(np.sum(valid_days >= 0.1)),
        'precip_percentile_95': float(np.percentile(valid_days, 95)),
        'precip_percentile_99': float(np.percentile(valid_days, 99)),
    }


def calculate_temperature_stats(
    daily_values: np.ndarray,
    year: int,
    scenario: str,
    county_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate temperature statistics for a county.
    
    Args:
        daily_values: Array of daily temperature values
        year: Year of the data
        scenario: Climate scenario name
        county_info: County metadata
        
    Returns:
        Dictionary of temperature statistics
    """
    valid_days = daily_values[~np.isnan(daily_values)]
    
    if len(valid_days) == 0:
        return None
    
    return {
        'year': year,
        'scenario': scenario,
        'county_id': county_info['county_id'],
        'county_name': county_info['county_name'],
        'state': county_info['state'],
        'mean_annual_temp_c': float(np.mean(valid_days)),
        'min_temp_c': float(np.min(valid_days)),
        'max_temp_c': float(np.max(valid_days)),
        'temp_range_c': float(np.max(valid_days) - np.min(valid_days)),
        'temp_std_c': float(np.std(valid_days)),
        'days_below_freezing': int(np.sum(valid_days < 0)),
        'days_above_30c': int(np.sum(valid_days > 30)),
        'growing_degree_days': float(np.sum(np.maximum(valid_days - 10, 0))),  # Base 10°C
        'cooling_degree_days': float(np.sum(np.maximum(valid_days - 18, 0))),  # Base 18°C
        'heating_degree_days': float(np.sum(np.maximum(18 - valid_days, 0))),  # Base 18°C
    }


def calculate_tasmax_stats(
    daily_values: np.ndarray,
    threshold_temp_c: float,
    year: int,
    scenario: str,
    county_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate daily maximum temperature statistics for a county.
    
    Args:
        daily_values: Array of daily maximum temperature values
        threshold_temp_c: Temperature threshold for hot days
        year: Year of the data
        scenario: Climate scenario name
        county_info: County metadata
        
    Returns:
        Dictionary of daily maximum temperature statistics
    """
    valid_days = daily_values[~np.isnan(daily_values)]
    
    if len(valid_days) == 0:
        return None
    
    return {
        'year': year,
        'scenario': scenario,
        'county_id': county_info['county_id'],
        'county_name': county_info['county_name'],
        'state': county_info['state'],
        'mean_annual_tasmax_c': float(np.mean(valid_days)),
        'min_tasmax_c': float(np.min(valid_days)),
        'max_tasmax_c': float(np.max(valid_days)),
        'tasmax_range_c': float(np.max(valid_days) - np.min(valid_days)),
        'tasmax_std_c': float(np.std(valid_days)),
        'days_above_threshold_c': int(np.sum(valid_days > threshold_temp_c)) if threshold_temp_c is not None else 0,
        'threshold_temp_c': float(threshold_temp_c) if threshold_temp_c is not None else 0.0,
        'days_above_30c': int(np.sum(valid_days > 30)),
        'days_above_35c': int(np.sum(valid_days > 35)),
        'days_above_40c': int(np.sum(valid_days > 40)),
        'growing_degree_days_max': float(np.sum(np.maximum(valid_days - 10, 0))),  # Base 10°C
        'heat_index_days': int(np.sum(valid_days > 32)),  # Days above 32°C (90°F)
    }


def calculate_tasmin_stats(
    daily_values: np.ndarray,
    year: int,
    scenario: str,
    county_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate daily minimum temperature statistics for a county.
    
    Args:
        daily_values: Array of daily minimum temperature values
        year: Year of the data
        scenario: Climate scenario name
        county_info: County metadata
        
    Returns:
        Dictionary of daily minimum temperature statistics
    """
    valid_days = daily_values[~np.isnan(daily_values)]
    
    if len(valid_days) == 0:
        return None
    
    return {
        'year': year,
        'scenario': scenario,
        'county_id': county_info['county_id'],
        'county_name': county_info['county_name'],
        'state': county_info['state'],
        'mean_annual_tasmin_c': float(np.mean(valid_days)),
        'min_tasmin_c': float(np.min(valid_days)),
        'max_tasmin_c': float(np.max(valid_days)),
        'tasmin_range_c': float(np.max(valid_days) - np.min(valid_days)),
        'tasmin_std_c': float(np.std(valid_days)),
        'cold_days': int(np.sum(valid_days < 0)),  # Days below 0°C (freezing)
        'extreme_cold_days': int(np.sum(valid_days < -10)),  # Days below -10°C
        'very_extreme_cold_days': int(np.sum(valid_days < -20)),  # Days below -20°C
        'days_above_freezing': int(np.sum(valid_days >= 0)),  # Days at or above 0°C
        'frost_free_days': int(np.sum(valid_days > 0)),  # Days above 0°C
        'growing_degree_days_min': float(np.sum(np.maximum(valid_days - 0, 0))),  # Base 0°C
        'heating_degree_days': float(np.sum(np.maximum(18 - valid_days, 0))),  # Base 18°C
    }


def calculate_statistics(
    data: np.ndarray,
    variable: str,
    threshold: float,
    year: int,
    scenario: str,
    county_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate statistics for any climate variable.
    
    Args:
        data: Array of daily values
        variable: Climate variable name
        threshold: Threshold value for the variable
        year: Year of the data
        scenario: Climate scenario name
        county_info: County metadata
        
    Returns:
        Dictionary of statistics
    """
    if variable == 'pr':
        return calculate_precipitation_stats(data, threshold, year, scenario, county_info)
    elif variable == 'tas':
        return calculate_temperature_stats(data, year, scenario, county_info)
    elif variable == 'tasmax':
        return calculate_tasmax_stats(data, threshold, year, scenario, county_info)
    elif variable == 'tasmin':
        return calculate_tasmin_stats(data, year, scenario, county_info)
    else:
        raise ValueError(f"Unsupported variable: {variable}") 