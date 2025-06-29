#!/usr/bin/env python3
"""
Optimized climate indicator calculator with caching and performance improvements.
"""

import xarray as xr
import numpy as np
import pandas as pd
from xclim import atmos
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import pickle
import hashlib
import os
from functools import lru_cache
import time

warnings.filterwarnings('ignore')

from climate_indicator_calculator import ClimateIndicatorCalculator


class OptimizedClimateCalculator(ClimateIndicatorCalculator):
    """
    Optimized version of climate calculator with caching and performance improvements.
    """
    
    def __init__(self, 
                 base_data_path: str, 
                 baseline_period: Tuple[int, int] = (1980, 2010),
                 cache_dir: Optional[str] = None,
                 enable_caching: bool = True):
        """
        Initialize optimized calculator with caching support.
        
        Parameters
        ----------
        base_data_path : str
            Base path to climate data
        baseline_period : tuple
            (start_year, end_year) for baseline
        cache_dir : str, optional
            Directory for cache files. If None, uses temp directory
        enable_caching : bool
            Whether to enable caching
        """
        super().__init__(base_data_path, baseline_period)
        
        self.enable_caching = enable_caching
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / '.climate_cache'
        
        if self.enable_caching:
            self.cache_dir.mkdir(exist_ok=True)
            print(f"Cache directory: {self.cache_dir}")
            
        # In-memory cache for current session
        self._memory_cache = {}
        
    def _get_cache_key(self, bounds: Tuple[float, float, float, float], 
                      prefix: str = 'baseline') -> str:
        """Generate unique cache key for bounds."""
        # Create hash of bounds and baseline period
        key_data = f"{prefix}_{bounds}_{self.baseline_period}".encode()
        return hashlib.md5(key_data).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load data from cache if available."""
        if not self.enable_caching:
            return None
            
        # Check memory cache first
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]
            
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                # Also store in memory cache
                self._memory_cache[cache_key] = data
                return data
            except Exception as e:
                print(f"Warning: Failed to load cache {cache_file}: {e}")
                
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict) -> None:
        """Save data to cache."""
        if not self.enable_caching:
            return
            
        # Save to memory cache
        self._memory_cache[cache_key] = data
        
        # Save to disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Warning: Failed to save cache {cache_file}: {e}")
    
    def calculate_baseline_percentiles(self, 
                                     bounds: Tuple[float, float, float, float],
                                     variables: List[str] = ['tasmax', 'tasmin']) -> Dict:
        """
        Calculate baseline percentiles with caching.
        """
        # Generate cache key
        cache_key = self._get_cache_key(bounds, 'baseline_percentiles')
        
        # Try to load from cache
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            print(f"  Loaded baseline percentiles from cache")
            return cached_data
            
        # Calculate if not cached
        print(f"  Calculating baseline percentiles...")
        start_time = time.time()
        thresholds = super().calculate_baseline_percentiles(bounds, variables)
        calc_time = time.time() - start_time
        print(f"  Baseline calculation took {calc_time:.2f}s")
        
        # Save to cache
        self._save_to_cache(cache_key, thresholds)
        
        return thresholds
    
    def extract_county_data_optimized(self, 
                                    files: List[Path], 
                                    variable: str,
                                    bounds: Tuple[float, float, float, float]) -> xr.DataArray:
        """
        Optimized data extraction with better I/O handling.
        """
        if not files:
            raise ValueError(f"No files provided for variable {variable}")
            
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Convert longitude to 0-360 if needed
        if min_lon < 0:
            min_lon = min_lon % 360
        if max_lon < 0:
            max_lon = max_lon % 360
            
        # Strategy 1: Pre-select region for each file before combining
        # This reduces memory usage and speeds up processing
        arrays = []
        
        for file_path in files:
            # Open single file
            with xr.open_dataset(file_path) as ds:
                # Select region immediately to reduce data volume
                regional = ds[variable].sel(
                    lat=slice(min_lat - 0.5, max_lat + 0.5),
                    lon=slice(min_lon - 0.5, max_lon + 0.5)
                )
                # Load to memory and close file immediately
                arrays.append(regional.load())
        
        # Combine all arrays
        if len(arrays) > 1:
            combined = xr.concat(arrays, dim='time')
        else:
            combined = arrays[0]
            
        # Calculate area-weighted mean
        weights = np.cos(np.deg2rad(combined.lat))
        county_mean = combined.weighted(weights).mean(dim=['lat', 'lon'])
        
        # Ensure units are preserved
        county_mean.attrs['units'] = arrays[0].attrs.get('units', 'unknown')
        
        return county_mean
    
    @lru_cache(maxsize=128)
    def get_files_for_period_cached(self, 
                                  variable: str, 
                                  scenario: str, 
                                  start_year: int, 
                                  end_year: int) -> Tuple[Path, ...]:
        """
        Cached version of get_files_for_period.
        Returns tuple for hashability.
        """
        files = self.get_files_for_period(variable, scenario, start_year, end_year)
        return tuple(files)
    
    def process_county_optimized(self,
                               county_info: Dict,
                               scenarios: List[str],
                               variables: List[str],
                               historical_period: Tuple[int, int],
                               future_period: Tuple[int, int]) -> List[Dict]:
        """
        Optimized county processing with caching and better I/O.
        """
        results = []
        bounds = county_info['bounds']
        
        # Calculate baseline percentiles once (with caching)
        print(f"  Processing {county_info['name']}...")
        thresholds = self.calculate_baseline_percentiles(bounds)
        
        # Process each scenario
        for scenario in scenarios:
            print(f"    Processing {scenario}...")
            
            # Determine time period
            if scenario == 'historical':
                start_year, end_year = historical_period
            else:
                start_year, end_year = future_period
                
            # Load data for all variables using optimized method
            scenario_data = {}
            
            for var in variables:
                files = list(self.get_files_for_period_cached(
                    var, scenario, start_year, end_year
                ))
                
                if files:
                    # Use optimized extraction
                    county_mean = self.extract_county_data_optimized(files, var, bounds)
                    scenario_data[var] = county_mean.values
                    scenario_data['time'] = county_mean.time.values
                    
            # Calculate indicators (same as before)
            if all(var in scenario_data for var in variables):
                indicators = self.calculate_indicators(scenario_data, thresholds)
                
                # Extract annual values
                time_values = indicators['tg_mean'].time.values
                if hasattr(time_values[0], 'year'):
                    years = [t.year for t in time_values]
                else:
                    years = pd.to_datetime(time_values).year.tolist()
                    
                # Create annual records
                for i, year in enumerate(years):
                    record = {
                        'GEOID': county_info['geoid'],
                        'NAME': county_info['name'],
                        'STATE': county_info['state'],
                        'scenario': scenario,
                        'year': year
                    }
                    
                    # Add indicator values
                    for ind_name, ind_data in indicators.items():
                        if ind_name == 'tg_mean':
                            record[ind_name + '_C'] = float(ind_data.isel(time=i).values) - 273.15
                        elif ind_name == 'precip_accumulation':
                            record[ind_name + '_mm'] = float(ind_data.isel(time=i).values)
                        elif ind_name in ['tx90p', 'tn10p']:
                            # Convert to percentage
                            count = float(ind_data.isel(time=i).values)
                            year_start = pd.Timestamp(f'{year}-01-01')
                            year_end = pd.Timestamp(f'{year}-12-31')
                            n_days = (year_end - year_start).days + 1
                            percentage = (count / n_days) * 100
                            record[ind_name + '_percent'] = percentage
                        else:
                            record[ind_name] = float(ind_data.isel(time=i).values)
                            
                    results.append(record)
                    
        return results
    
    def clear_cache(self, memory_only: bool = False) -> None:
        """
        Clear cache.
        
        Parameters
        ----------
        memory_only : bool
            If True, only clears memory cache. If False, also clears disk cache.
        """
        # Clear memory cache
        self._memory_cache.clear()
        print("Memory cache cleared")
        
        # Clear disk cache if requested
        if not memory_only and self.cache_dir.exists():
            cache_files = list(self.cache_dir.glob("*.pkl"))
            for f in cache_files:
                f.unlink()
            print(f"Removed {len(cache_files)} cache files from disk")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        stats = {
            'memory_cache_size': len(self._memory_cache),
            'memory_cache_keys': list(self._memory_cache.keys())[:5],  # First 5 keys
        }
        
        if self.cache_dir.exists():
            cache_files = list(self.cache_dir.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in cache_files) / 1024 / 1024  # MB
            stats['disk_cache_files'] = len(cache_files)
            stats['disk_cache_size_mb'] = round(total_size, 2)
            
        return stats


class BatchOptimizedCalculator(OptimizedClimateCalculator):
    """
    Further optimized calculator for batch processing multiple counties.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with batch processing optimizations."""
        super().__init__(*args, **kwargs)
        self._file_handle_cache = {}
        
    def process_county_batch(self, 
                           county_infos: List[Dict],
                           scenarios: List[str],
                           variables: List[str],
                           historical_period: Tuple[int, int],
                           future_period: Tuple[int, int]) -> List[Dict]:
        """
        Process multiple counties in a batch with shared file handles.
        """
        all_results = []
        
        # Pre-load all baseline percentiles
        print(f"Pre-calculating baselines for {len(county_infos)} counties...")
        baselines = {}
        for county_info in county_infos:
            baselines[county_info['geoid']] = self.calculate_baseline_percentiles(
                county_info['bounds']
            )
        
        # Process by scenario to minimize file opening/closing
        for scenario in scenarios:
            print(f"\nProcessing scenario: {scenario}")
            
            # Determine time period
            if scenario == 'historical':
                start_year, end_year = historical_period
            else:
                start_year, end_year = future_period
            
            # Pre-open all files for this scenario
            scenario_files = {}
            for var in variables:
                files = list(self.get_files_for_period_cached(
                    var, scenario, start_year, end_year
                ))
                if files:
                    # Open all files at once
                    scenario_files[var] = xr.open_mfdataset(
                        files, 
                        combine='by_coords',
                        chunks={'time': 365}  # Chunk by year
                    )
            
            # Process each county with open file handles
            for county_info in county_infos:
                county_scenario_data = {}
                bounds = county_info['bounds']
                min_lon, min_lat, max_lon, max_lat = bounds
                
                # Convert longitude if needed
                if min_lon < 0:
                    min_lon = min_lon % 360
                if max_lon < 0:
                    max_lon = max_lon % 360
                
                # Extract data for each variable
                for var, ds in scenario_files.items():
                    # Select region
                    regional = ds[var].sel(
                        lat=slice(min_lat - 0.5, max_lat + 0.5),
                        lon=slice(min_lon - 0.5, max_lon + 0.5)
                    )
                    
                    # Calculate weighted mean
                    weights = np.cos(np.deg2rad(regional.lat))
                    county_mean = regional.weighted(weights).mean(dim=['lat', 'lon'])
                    
                    # Load to memory
                    county_scenario_data[var] = county_mean.compute().values
                    county_scenario_data['time'] = county_mean.time.values
                
                # Calculate indicators using cached baseline
                if all(var in county_scenario_data for var in variables):
                    thresholds = baselines[county_info['geoid']]
                    indicators = self.calculate_indicators(county_scenario_data, thresholds)
                    
                    # Convert to records (same as before)
                    time_values = indicators['tg_mean'].time.values
                    if hasattr(time_values[0], 'year'):
                        years = [t.year for t in time_values]
                    else:
                        years = pd.to_datetime(time_values).year.tolist()
                    
                    for i, year in enumerate(years):
                        record = {
                            'GEOID': county_info['geoid'],
                            'NAME': county_info['name'],
                            'STATE': county_info['state'],
                            'scenario': scenario,
                            'year': year
                        }
                        
                        # Add indicator values
                        for ind_name, ind_data in indicators.items():
                            if ind_name == 'tg_mean':
                                record[ind_name + '_C'] = float(ind_data.isel(time=i).values) - 273.15
                            elif ind_name == 'precip_accumulation':
                                record[ind_name + '_mm'] = float(ind_data.isel(time=i).values)
                            elif ind_name in ['tx90p', 'tn10p']:
                                count = float(ind_data.isel(time=i).values)
                                year_start = pd.Timestamp(f'{year}-01-01')
                                year_end = pd.Timestamp(f'{year}-12-31')
                                n_days = (year_end - year_start).days + 1
                                percentage = (count / n_days) * 100
                                record[ind_name + '_percent'] = percentage
                            else:
                                record[ind_name] = float(ind_data.isel(time=i).values)
                        
                        all_results.append(record)
            
            # Close file handles for this scenario
            for ds in scenario_files.values():
                ds.close()
                
        return all_results


# Example usage
if __name__ == "__main__":
    # Test optimized calculator
    print("Testing optimized climate calculator...")
    
    calculator = OptimizedClimateCalculator(
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM",
        baseline_period=(1980, 2010),
        enable_caching=True
    )
    
    # Test county
    county_info = {
        'geoid': '31039',
        'name': 'Cuming',
        'state': '31',
        'bounds': (-96.7887, 41.7193, -96.1251, 42.2088)
    }
    
    # First run - calculates baseline
    print("\nFirst run (no cache):")
    start = time.time()
    results1 = calculator.process_county_optimized(
        county_info=county_info,
        scenarios=['historical'],
        variables=['tas', 'tasmax', 'tasmin', 'pr'],
        historical_period=(2009, 2010),
        future_period=(2040, 2041)
    )
    time1 = time.time() - start
    print(f"Time: {time1:.2f}s")
    
    # Second run - uses cache
    print("\nSecond run (with cache):")
    start = time.time()
    results2 = calculator.process_county_optimized(
        county_info=county_info,
        scenarios=['historical'],
        variables=['tas', 'tasmax', 'tasmin', 'pr'],
        historical_period=(2009, 2010),
        future_period=(2040, 2041)
    )
    time2 = time.time() - start
    print(f"Time: {time2:.2f}s")
    print(f"Speedup: {time1/time2:.1f}x")
    
    # Check cache stats
    print("\nCache statistics:")
    stats = calculator.get_cache_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")