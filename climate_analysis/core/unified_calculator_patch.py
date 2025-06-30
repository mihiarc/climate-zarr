"""Patch for UnifiedClimateCalculator to handle variable/scenario directory structure."""

import logging
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, List, Any

from climate_analysis.utils.climate_utils import adjust_longitude_bounds, calculate_area_weighted_mean
from climate_analysis.utils.file_operations import load_netcdf

logger = logging.getLogger(__name__)


def extract_county_data_patched(
    self,
    scenario: str,
    county_bounds: List[float],
    county_info: Dict[str, Any]
) -> xr.Dataset:
    """Extract county data with proper variable/scenario directory structure."""
    # Adjust bounds for dateline
    min_lon, min_lat, max_lon, max_lat = adjust_longitude_bounds(county_bounds)
    
    # Buffer for edge effects
    buffer = 0.1
    lat_slice = slice(min_lat - buffer, max_lat + buffer)
    lon_slice = slice(min_lon - buffer, max_lon + buffer)
    
    # Variables to load
    variables = ['tas', 'tasmax', 'tasmin', 'pr']
    
    # Load each variable separately
    variable_datasets = {}
    
    for variable in variables:
        # Get files for this variable and scenario
        variable_path = self.base_path / variable / scenario
        
        if not variable_path.exists():
            logger.warning(f"Path does not exist: {variable_path}")
            continue
            
        # Find all NC files for this variable
        var_files = sorted(variable_path.glob(f"{variable}_*.nc"))
        
        if not var_files:
            logger.warning(f"No files found for {variable} in {scenario}")
            continue
        
        logger.info(f"Found {len(var_files)} files for {variable} in {scenario}")
        
        # Load and concatenate files for this variable
        var_datasets = []
        for file_path in var_files:
            try:
                ds = load_netcdf(
                    file_path,
                    preselect_bounds={'lat': lat_slice, 'lon': lon_slice},
                    chunks={'time': 365}
                )
                if ds is not None and variable in ds.data_vars:
                    # Keep only the variable we need
                    ds = ds[[variable]]
                    var_datasets.append(ds)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        if var_datasets:
            # Concatenate all time periods for this variable
            combined_var = xr.concat(var_datasets, dim='time')
            combined_var = combined_var.sortby('time')
            
            # Calculate area-weighted mean for this variable
            weighted_var = calculate_area_weighted_mean(combined_var)
            
            # Store in dictionary
            variable_datasets[variable] = weighted_var[variable]
    
    if not variable_datasets:
        logger.warning(f"No data loaded for county {county_info.get('geoid')} in scenario {scenario}")
        return xr.Dataset()
    
    # Combine all variables into a single dataset
    # First, find common time coordinates
    all_times = None
    for var_name, var_data in variable_datasets.items():
        if all_times is None:
            all_times = var_data.time
        else:
            # Find intersection of times
            all_times = xr.DataArray(
                data=pd.Index(all_times.values).intersection(pd.Index(var_data.time.values)),
                dims=['time']
            )
    
    # Create unified dataset with common times
    unified_data = xr.Dataset()
    for var_name, var_data in variable_datasets.items():
        # Select only common times
        unified_data[var_name] = var_data.sel(time=all_times)
    
    logger.info(f"Loaded data for county {county_info.get('geoid')}: "
                f"variables={list(unified_data.data_vars)}, "
                f"time_range={str(unified_data.time.min().values)[:10]} to {str(unified_data.time.max().values)[:10]}")
    
    return unified_data


def calculate_indicators_patched(
    self,
    scenarios: List[str],
    county_bounds: List[float],
    county_info: Dict[str, Any],
    indicators_config: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Calculate climate indicators for a county with cftime handling."""
    all_results = []
    
    # Process each scenario
    indicators_by_scenario = {}
    
    for scenario in scenarios:
        # Extract county data
        county_data = self.extract_county_data(scenario, county_bounds, county_info)
        
        if len(county_data.data_vars) == 0:
            logger.warning(f"No data found for county {county_info.get('geoid')} in scenario {scenario}")
            continue
        
        # Calculate baseline percentiles (uses merged data if available)
        baseline_percentiles = self.calculate_baseline_percentiles(
            county_data,
            percentiles=[10, 90],
            county_bounds=county_bounds,
            county_id=county_info.get('geoid')
        )
        
        # Calculate indicators
        scenario_indicators = {}
        
        # Handle cftime dates properly
        time_values = county_data.time.values
        if hasattr(time_values[0], 'year'):
            # cftime objects have year attribute
            years = [t.year for t in time_values]
        else:
            # Try pandas datetime conversion as fallback
            years = pd.to_datetime(time_values).year.tolist()
        
        years = sorted(set(years))
        
        for ind_name, ind_config in indicators_config.items():
            indicator = self._calculate_single_indicator(
                county_data,
                ind_name,
                ind_config,
                baseline_percentiles
            )
            
            if indicator is not None:
                scenario_indicators[ind_name] = indicator
        
        indicators_by_scenario[scenario] = scenario_indicators
    
    # Format results
    if indicators_by_scenario:
        from climate_analysis.utils.climate_utils import format_results_for_county
        all_results = format_results_for_county(
            county_info,
            list(indicators_by_scenario.keys()),
            indicators_by_scenario,
            years
        )
    
    return all_results


def apply_patch():
    """Apply the patch to UnifiedClimateCalculator."""
    from climate_analysis.core.unified_calculator import UnifiedClimateCalculator
    
    # Replace the methods with our patched versions
    UnifiedClimateCalculator.extract_county_data = extract_county_data_patched
    UnifiedClimateCalculator.calculate_indicators = calculate_indicators_patched
    
    logger.info("Applied patches to UnifiedClimateCalculator")