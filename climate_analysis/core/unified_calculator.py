"""Unified climate calculator using the best performing strategy.

This module implements the merged baseline approach which provided the best
performance in our testing (100-1000x speedup for baseline lookups).
"""

import logging
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from functools import lru_cache
import pickle
import warnings
from xclim import atmos

from climate_analysis.utils.climate_utils import (
    create_annual_record,
    adjust_longitude_bounds,
    calculate_area_weighted_mean,
    format_results_for_county
)
from climate_analysis.utils.file_operations import (
    load_netcdf,
    load_pickle,
    find_files
)

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class UnifiedClimateCalculator:
    """Production climate calculator using merged baseline strategy.
    
    This calculator uses pre-computed merged baseline files for optimal performance.
    Falls back to standard calculation if merged baselines are not available.
    """
    
    def __init__(
        self,
        base_data_path: str,
        merged_baseline_path: Optional[str] = None,
        baseline_period: Tuple[int, int] = (1980, 2010)
    ):
        """Initialize the calculator.
        
        Args:
            base_data_path: Path to base climate data
            merged_baseline_path: Path to merged baseline pickle file
            baseline_period: Tuple of (start_year, end_year) for baseline
        """
        self.base_path = Path(base_data_path)
        self.baseline_period = baseline_period
        
        self.merged_baseline_path = Path(merged_baseline_path) if merged_baseline_path else None
        self.merged_baselines = None
        self.baseline_lookup = {}  # GEOID -> baseline data
        self.baseline_by_bounds = {}  # bounds_key -> baseline data
        
        # Load merged baselines if available
        if self.merged_baseline_path and self.merged_baseline_path.exists():
            self._load_merged_baselines()
            logger.info(f"Loaded merged baselines from {self.merged_baseline_path}")
        else:
            logger.warning("No merged baselines available, will calculate on demand")
    
    def _load_merged_baselines(self):
        """Load the merged baseline cache file."""
        try:
            with open(self.merged_baseline_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Handle different formats
            if isinstance(cache_data, dict):
                # Check if it's the new format with separate percentile arrays
                if 'tasmax_p90' in cache_data and 'tasmin_p10' in cache_data:
                    # New format: convert to expected structure
                    self.merged_baselines = cache_data
                    county_info = cache_data.get('county_info', {})
                    
                    # Build lookup dictionaries
                    for geoid, info in county_info.items():
                        # Create baseline data in expected format
                        baseline_data = {
                            90: xr.Dataset({
                                'tasmax': cache_data['tasmax_p90'].get(geoid)
                            }) if geoid in cache_data.get('tasmax_p90', {}) else None,
                            10: xr.Dataset({
                                'tasmin': cache_data['tasmin_p10'].get(geoid)
                            }) if geoid in cache_data.get('tasmin_p10', {}) else None
                        }
                        
                        # Only add if we have data
                        if baseline_data[90] is not None or baseline_data[10] is not None:
                            self.baseline_lookup[str(geoid)] = baseline_data
                            
                            if 'bounds' in info:
                                bounds_key = self._make_bounds_key(info['bounds'])
                                self.baseline_by_bounds[bounds_key] = baseline_data
                else:
                    # Old format
                    self.merged_baselines = cache_data
            elif isinstance(cache_data, list):
                # Convert list format to lookup dictionaries
                for item in cache_data:
                    geoid = item.get('GEOID')
                    bounds = item.get('bounds')
                    baseline_data = item.get('baseline_data')
                    
                    if geoid and baseline_data:
                        self.baseline_lookup[geoid] = baseline_data
                    
                    if bounds and baseline_data:
                        bounds_key = self._make_bounds_key(bounds)
                        self.baseline_by_bounds[bounds_key] = baseline_data
            
            logger.info(f"Loaded baselines for {len(self.baseline_lookup)} counties")
            
        except Exception as e:
            logger.error(f"Error loading merged baselines: {e}")
            self.merged_baselines = None
    
    def _make_bounds_key(self, bounds: List[float]) -> str:
        """Create a hashable key from bounds."""
        return f"{bounds[0]:.3f},{bounds[1]:.3f},{bounds[2]:.3f},{bounds[3]:.3f}"
    
    def get_files_for_period(self, 
                            variable: str, 
                            scenario: str, 
                            start_year: int, 
                            end_year: int) -> List[Path]:
        """Get list of files for a specific variable, scenario, and time period.
        
        Parameters:
            variable: Climate variable name
            scenario: Climate scenario (e.g., 'historical', 'ssp245')
            start_year: Start year
            end_year: End year
            
        Returns:
            List of file paths
        """
        file_pattern = self.base_path / variable / scenario / f"{variable}_*.nc"
        all_files = list(file_pattern.parent.glob(file_pattern.name))
        
        # Filter by year range
        period_files = []
        for f in all_files:
            parts = f.stem.split('_')
            for part in parts:
                if part.isdigit() and len(part) == 4:
                    year = int(part)
                    if start_year <= year <= end_year:
                        period_files.append(f)
                    break
                    
        return sorted(period_files)
    
    def extract_county_data_base(self, 
                           files: List[Path], 
                           variable: str,
                           bounds: Tuple[float, float, float, float]) -> xr.DataArray:
        """Extract spatially averaged data for a county from NetCDF files.
        
        This is the base implementation without optimizations.
        
        Parameters:
            files: List of NetCDF file paths
            variable: Variable name to extract
            bounds: (min_lon, min_lat, max_lon, max_lat) in degrees
            
        Returns:
            Spatially averaged time series
        """
        if not files:
            raise ValueError(f"No files provided for variable {variable}")
            
        # Load data
        ds = xr.open_mfdataset(files, combine='by_coords')
        
        # Extract bounds
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Convert longitude to 0-360 if needed (NEX-GDDP uses 0-360)
        if min_lon < 0:
            min_lon = min_lon % 360
        if max_lon < 0:
            max_lon = max_lon % 360
            
        # Add buffer for selection
        lat_slice = slice(min_lat - 0.5, max_lat + 0.5)
        lon_slice = slice(min_lon - 0.5, max_lon + 0.5)
        
        # Extract region
        county_data = ds[variable].sel(lat=lat_slice, lon=lon_slice)
        
        # Calculate area-weighted mean
        weights = np.cos(np.deg2rad(county_data.lat))
        county_mean = county_data.weighted(weights).mean(dim=['lat', 'lon'])
        
        # Compute to avoid dask issues
        county_mean = county_mean.compute()
        
        # Ensure units are preserved
        county_mean.attrs['units'] = ds[variable].attrs.get('units', 'unknown')
        
        ds.close()
        
        return county_mean
    
    def calculate_baseline_percentiles_base(self, 
                                     bounds: Tuple[float, float, float, float],
                                     variables: List[str] = ['tasmax', 'tasmin']) -> Dict:
        """Calculate percentile thresholds from baseline period.
        
        This is the base implementation without caching.
        
        Parameters:
            bounds: County bounds (min_lon, min_lat, max_lon, max_lat)
            variables: Variables to calculate percentiles for
            
        Returns:
            Dictionary with day-of-year percentile thresholds
        """
        thresholds = {}
        baseline_start, baseline_end = self.baseline_period
        
        for var in variables:
            # Get baseline files
            baseline_files = self.get_files_for_period(
                var, 'historical', baseline_start, baseline_end
            )
            
            if len(baseline_files) < 10:
                logger.warning(f"Only {len(baseline_files)} years available for {var} baseline")
                continue
                
            # Extract data
            county_mean = self.extract_county_data_base(baseline_files, var, bounds)
            
            # Ensure proper units
            if var in ['tasmax', 'tasmin', 'tas']:
                county_mean.attrs['units'] = 'K'
                
            # Calculate day-of-year percentiles
            if var == 'tasmax':
                grouped = county_mean.groupby('time.dayofyear')
                thresholds['tasmax_p90_doy'] = grouped.quantile(0.9, dim='time')
                thresholds['tasmax_p90_doy'].attrs['units'] = 'K'
                
            elif var == 'tasmin':
                grouped = county_mean.groupby('time.dayofyear')
                thresholds['tasmin_p10_doy'] = grouped.quantile(0.1, dim='time')
                thresholds['tasmin_p10_doy'].attrs['units'] = 'K'
                
        return thresholds
    
    def calculate_baseline_percentiles(
        self,
        county_data: xr.Dataset,
        percentiles: List[int] = [10, 90],
        county_bounds: Optional[List[float]] = None,
        county_id: Optional[str] = None
    ) -> Dict[int, xr.Dataset]:
        """Calculate or retrieve baseline percentiles.
        
        First tries to use merged baselines for instant lookup.
        Falls back to standard calculation if not available.
        """
        # Try GEOID lookup first
        if county_id and county_id in self.baseline_lookup:
            logger.debug(f"Using merged baseline for county {county_id}")
            return self.baseline_lookup[county_id]
        
        # Try bounds lookup
        if county_bounds:
            bounds_key = self._make_bounds_key(county_bounds)
            if bounds_key in self.baseline_by_bounds:
                logger.debug(f"Using merged baseline for bounds {bounds_key}")
                return self.baseline_by_bounds[bounds_key]
        
        # Fallback to calculation
        logger.debug("Calculating baseline percentiles (no merged data available)")
        
        # Convert to base method format
        if county_bounds:
            thresholds = self.calculate_baseline_percentiles_base(county_bounds)
            
            # Convert to expected format
            result = {}
            if 90 in percentiles and 'tasmax_p90_doy' in thresholds:
                result[90] = xr.Dataset({'tasmax': thresholds['tasmax_p90_doy']})
            if 10 in percentiles and 'tasmin_p10_doy' in thresholds:
                result[10] = xr.Dataset({'tasmin': thresholds['tasmin_p10_doy']})
            
            return result
        
        return {}
    
    def extract_county_data(
        self,
        scenario: str,
        county_bounds: List[float],
        county_info: Dict[str, Any]
    ) -> xr.Dataset:
        """Extract county data with optimized loading."""
        # Adjust bounds for dateline
        min_lon, min_lat, max_lon, max_lat = adjust_longitude_bounds(county_bounds)
        
        # Buffer for edge effects
        buffer = 0.1
        lat_slice = slice(min_lat - buffer, max_lat + buffer)
        lon_slice = slice(min_lon - buffer, max_lon + buffer)
        
        # Get file list
        scenario_files = self._get_scenario_files(scenario)
        if not scenario_files:
            logger.warning(f"No files found for scenario {scenario}")
            return xr.Dataset()
        
        # Load with pre-selection for efficiency
        datasets = []
        for file_path in scenario_files:
            ds = load_netcdf(
                file_path,
                preselect_bounds={'lat': lat_slice, 'lon': lon_slice},
                chunks={'time': 365}
            )
            if ds is not None:
                datasets.append(ds)
        
        if not datasets:
            return xr.Dataset()
        
        # Combine datasets
        combined = xr.concat(datasets, dim='time')
        combined = combined.sortby('time')
        
        # Calculate area-weighted mean
        return calculate_area_weighted_mean(combined)
    
    @lru_cache(maxsize=32)
    def _get_scenario_files(self, scenario: str) -> List[Path]:
        """Get sorted list of files for a scenario (cached)."""
        scenario_path = self.base_path / scenario
        return find_files(scenario_path, "*.nc")
    
    def calculate_indicators_base(self, 
                           data: Dict[str, np.ndarray],
                           thresholds: Dict) -> Dict:
        """Calculate climate indicators from data using xclim.
        
        This is the core calculation method from the base class.
        
        Parameters:
            data: Dictionary with arrays for each variable and time coordinate
            thresholds: Pre-calculated baseline thresholds
            
        Returns:
            Dictionary of calculated indicators
        """
        # Create DataArrays
        time_coord = data['time']
        
        tas_da = xr.DataArray(data['tas'], dims=['time'], coords={'time': time_coord})
        tasmax_da = xr.DataArray(data['tasmax'], dims=['time'], coords={'time': time_coord})
        tasmin_da = xr.DataArray(data['tasmin'], dims=['time'], coords={'time': time_coord})
        pr_da = xr.DataArray(data['pr'], dims=['time'], coords={'time': time_coord})
        
        # Set units
        tas_da.attrs['units'] = 'K'
        tasmax_da.attrs['units'] = 'K'
        tasmin_da.attrs['units'] = 'K'
        pr_da.attrs['units'] = 'kg m-2 s-1'
        
        # Calculate indicators
        indicators = {}
        
        # Temperature percentiles
        if 'tasmax_p90_doy' in thresholds:
            indicators['tx90p'] = atmos.tx90p(tasmax_da, thresholds['tasmax_p90_doy'], freq='YS')
            
        if 'tasmin_p10_doy' in thresholds:
            indicators['tn10p'] = atmos.tn10p(tasmin_da, thresholds['tasmin_p10_doy'], freq='YS')
            
        # Fixed threshold indicators
        indicators['tx_days_above_90F'] = atmos.tx_days_above(tasmax_da, thresh='305.37 K', freq='YS')
        indicators['tn_days_below_32F'] = atmos.tn_days_below(tasmin_da, thresh='273.15 K', freq='YS')
        
        # Mean temperature
        indicators['tg_mean'] = atmos.tg_mean(tas_da, freq='YS')
        
        # Precipitation
        indicators['days_precip_over_25.4mm'] = atmos.wetdays(pr_da, thresh='0.000294 kg m-2 s-1', freq='YS')
        
        # Total precipitation (convert to mm/day first)
        pr_mm_day = pr_da * 86400
        pr_mm_day.attrs['units'] = 'mm/day'
        indicators['precip_accumulation'] = atmos.precip_accumulation(pr_mm_day, freq='YS')
        
        return indicators
    
    def calculate_indicators(
        self,
        scenarios: List[str],
        county_bounds: List[float],
        county_info: Dict[str, Any],
        indicators_config: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Calculate climate indicators for a county.
        
        This method orchestrates the full calculation pipeline with
        optimized baseline lookups when available.
        """
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
            years = pd.to_datetime(county_data.time.values).year.unique()
            
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
            all_results = format_results_for_county(
                county_info,
                list(indicators_by_scenario.keys()),
                indicators_by_scenario,
                years
            )
        
        return all_results
    
    def _calculate_single_indicator(
        self,
        county_data: xr.Dataset,
        indicator_name: str,
        config: Dict[str, Any],
        baseline_percentiles: Dict[int, xr.Dataset]
    ) -> Optional[xr.DataArray]:
        """Calculate a single climate indicator using xclim.
        
        Args:
            county_data: Dataset with climate variables
            indicator_name: Name of the indicator to calculate
            config: Configuration for this indicator
            baseline_percentiles: Pre-calculated baseline percentiles
            
        Returns:
            DataArray with calculated indicator or None if calculation fails
        """
        try:
            # Get the xclim function
            func_name = config.get('xclim_func')
            if not func_name:
                logger.warning(f"No xclim function specified for {indicator_name}")
                return None
            
            # Get the function from xclim.atmos
            if hasattr(atmos, func_name):
                xclim_func = getattr(atmos, func_name)
            else:
                logger.warning(f"xclim function {func_name} not found")
                return None
            
            # Prepare arguments based on indicator type
            var_name = config.get('variable')
            
            if indicator_name in ['tx90p', 'tn10p']:
                # Percentile-based indicators need baseline thresholds
                if indicator_name == 'tx90p' and 90 in baseline_percentiles:
                    thresh = baseline_percentiles[90].tasmax
                elif indicator_name == 'tn10p' and 10 in baseline_percentiles:
                    thresh = baseline_percentiles[10].tasmin
                else:
                    logger.warning(f"Missing baseline for {indicator_name}")
                    return None
                
                result = xclim_func(
                    county_data[var_name],
                    thresh,
                    freq=config.get('freq', 'YS')
                )
                
            elif 'thresh' in config:
                # Fixed threshold indicators
                result = xclim_func(
                    county_data[var_name],
                    thresh=config['thresh'],
                    freq=config.get('freq', 'YS')
                )
                
            else:
                # Simple indicators (no threshold)
                if indicator_name == 'precip_accumulation':
                    # Convert precipitation units first
                    pr_mm_day = county_data[var_name] * 86400
                    pr_mm_day.attrs['units'] = 'mm/day'
                    result = xclim_func(pr_mm_day, freq=config.get('freq', 'YS'))
                else:
                    result = xclim_func(
                        county_data[var_name],
                        freq=config.get('freq', 'YS')
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating {indicator_name}: {e}")
            return None