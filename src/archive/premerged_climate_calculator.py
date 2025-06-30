"""
Climate calculator optimized to use pre-merged data files.

This calculator leverages pre-merged data structures:
1. Time-merged baseline files for fast percentile calculations
2. County-extracted data subsets for direct loading
3. Regional tiles for batch processing
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import time

from .optimized_climate_calculator import OptimizedClimateCalculator


class PreMergedClimateCalculator(OptimizedClimateCalculator):
    """Climate calculator optimized for pre-merged data."""
    
    def __init__(self, 
                 base_data_path: str,
                 premerged_data_path: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 enable_caching: bool = True):
        """
        Initialize calculator with pre-merged data support.
        
        Parameters
        ----------
        base_data_path : str
            Path to original climate data (fallback)
        premerged_data_path : str, optional
            Path to pre-merged data directory
        cache_dir : str, optional
            Directory for caching results
        enable_caching : bool
            Whether to use caching
        """
        super().__init__(base_data_path, cache_dir, enable_caching)
        
        self.premerged_path = Path(premerged_data_path) if premerged_data_path else None
        self.premerged_available = False
        self.county_extracts = {}
        self.tile_index = {}
        self.baseline_merged_files = {}
        
        if self.premerged_path and self.premerged_path.exists():
            self._load_premerged_metadata()
    
    def _load_premerged_metadata(self):
        """Load metadata about available pre-merged data."""
        
        # Check for baseline merged files
        baseline_dir = self.premerged_path / 'baseline_merged'
        if baseline_dir.exists():
            for nc_file in baseline_dir.glob("*.nc"):
                var_name = nc_file.name.split('_')[0]
                self.baseline_merged_files[var_name] = nc_file
            print(f"Found {len(self.baseline_merged_files)} merged baseline files")
        
        # Check for county extracts
        county_dir = self.premerged_path / 'county_extracts'
        if county_dir.exists():
            for geoid_dir in county_dir.iterdir():
                if geoid_dir.is_dir() and geoid_dir.name.isdigit():
                    metadata_file = geoid_dir / 'metadata.json'
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            self.county_extracts[geoid_dir.name] = json.load(f)
            print(f"Found {len(self.county_extracts)} county extracts")
        
        # Load tile index
        tiles_dir = self.premerged_path / 'regional_tiles'
        if tiles_dir.exists():
            tile_index_file = tiles_dir / 'tile_index.json'
            if tile_index_file.exists():
                with open(tile_index_file, 'r') as f:
                    self.tile_index = json.load(f)
                print(f"Found {len(self.tile_index)} regional tiles")
        
        self.premerged_available = bool(
            self.baseline_merged_files or self.county_extracts or self.tile_index
        )
    
    def calculate_baseline_percentiles(self,
                                     bounds: Tuple[float, float, float, float],
                                     variables: List[str] = ['tasmax', 'tasmin']) -> Dict:
        """
        Calculate baseline percentiles using pre-merged data when available.
        """
        # Check cache first
        cache_key = self._get_cache_key(bounds, 'baseline_percentiles')
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            print(f"  Loaded baseline percentiles from cache")
            return cached_data
        
        # Try to use pre-merged baseline files
        if self.baseline_merged_files:
            print(f"  Calculating baseline percentiles from pre-merged data...")
            start_time = time.time()
            
            thresholds = {}
            
            for var in variables:
                if var in self.baseline_merged_files:
                    # Load pre-merged baseline file
                    baseline_file = self.baseline_merged_files[var]
                    
                    # Extract region from pre-merged file
                    with xr.open_dataset(baseline_file) as ds:
                        min_lon, min_lat, max_lon, max_lat = bounds
                        
                        # Adjust longitude if needed
                        if min_lon < 0:
                            min_lon = min_lon % 360
                        if max_lon < 0:
                            max_lon = max_lon % 360
                        
                        # Extract region
                        regional = ds[var].sel(
                            lat=slice(min_lat - 0.5, max_lat + 0.5),
                            lon=slice(min_lon - 0.5, max_lon + 0.5)
                        )
                        
                        # Calculate area-weighted mean
                        weights = np.cos(np.deg2rad(regional.lat))
                        county_mean = regional.weighted(weights).mean(dim=['lat', 'lon'])
                        
                        # Group by day of year
                        county_mean['dayofyear'] = county_mean.time.dt.dayofyear
                        
                        # Calculate percentiles
                        if var == 'tasmax':
                            p90 = county_mean.groupby('dayofyear').quantile(0.9)
                            thresholds[f'{var}_p90_doy'] = p90
                        elif var == 'tasmin':
                            p10 = county_mean.groupby('dayofyear').quantile(0.1)
                            thresholds[f'{var}_p10_doy'] = p10
            
            calc_time = time.time() - start_time
            print(f"  Baseline calculation took {calc_time:.2f}s (pre-merged)")
            
            # Save to cache
            self._save_to_cache(cache_key, thresholds)
            
            return thresholds
        else:
            # Fallback to original method
            return super().calculate_baseline_percentiles(bounds, variables)
    
    def extract_county_data_optimized(self,
                                    files: List[Path],
                                    variable: str,
                                    bounds: Tuple[float, float, float, float],
                                    geoid: Optional[str] = None,
                                    scenario: Optional[str] = None,
                                    year_range: Optional[Tuple[int, int]] = None) -> xr.DataArray:
        """
        Extract county data using pre-extracted files when available.
        """
        # Check if we have pre-extracted data for this county
        if geoid and geoid in self.county_extracts:
            extract_info = self.county_extracts[geoid]
            county_dir = self.premerged_path / 'county_extracts' / geoid
            
            # Look for matching pre-extracted file
            if scenario and year_range:
                extract_file = county_dir / f"{variable}_{scenario}_{year_range[0]}-{year_range[1]}.nc"
                
                if extract_file.exists():
                    print(f"    Using pre-extracted data for {geoid}")
                    
                    with xr.open_dataset(extract_file) as ds:
                        county_data = ds[variable]
                        
                        # Calculate area-weighted mean (data is already regional)
                        weights = np.cos(np.deg2rad(county_data.lat))
                        county_mean = county_data.weighted(weights).mean(dim=['lat', 'lon'])
                        
                        return county_mean
        
        # Fallback to original method
        return super().extract_county_data_optimized(files, variable, bounds)
    
    def process_county_optimized(self,
                               county_info: Dict,
                               scenarios: List[str],
                               variables: List[str],
                               historical_period: Tuple[int, int],
                               future_period: Tuple[int, int]) -> List[Dict]:
        """
        Process county using pre-merged data when available.
        """
        results = []
        bounds = county_info['bounds']
        geoid = county_info.get('geoid')
        
        # Calculate baseline percentiles (uses pre-merged if available)
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
            
            # Load data for all variables
            scenario_data = {}
            
            for var in variables:
                # Try pre-extracted data first
                if geoid and self._has_preextracted_data(geoid, var, scenario, (start_year, end_year)):
                    county_mean = self._load_preextracted_data(
                        geoid, var, scenario, (start_year, end_year)
                    )
                    scenario_data[var] = county_mean.values
                    scenario_data['time'] = county_mean.time.values
                else:
                    # Fallback to file-based loading
                    files = list(self.get_files_for_period_cached(
                        var, scenario, start_year, end_year
                    ))
                    
                    if files:
                        county_mean = self.extract_county_data_optimized(
                            files, var, bounds, geoid, scenario, (start_year, end_year)
                        )
                        scenario_data[var] = county_mean.values
                        scenario_data['time'] = county_mean.time.values
            
            # Calculate indicators and create records (same as parent)
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
    
    def _has_preextracted_data(self, geoid: str, variable: str, scenario: str, 
                              year_range: Tuple[int, int]) -> bool:
        """Check if pre-extracted data exists for this request."""
        if geoid not in self.county_extracts:
            return False
        
        county_dir = self.premerged_path / 'county_extracts' / geoid
        extract_file = county_dir / f"{variable}_{scenario}_{year_range[0]}-{year_range[1]}.nc"
        
        return extract_file.exists()
    
    def _load_preextracted_data(self, geoid: str, variable: str, scenario: str,
                               year_range: Tuple[int, int]) -> xr.DataArray:
        """Load pre-extracted county data."""
        county_dir = self.premerged_path / 'county_extracts' / geoid
        extract_file = county_dir / f"{variable}_{scenario}_{year_range[0]}-{year_range[1]}.nc"
        
        with xr.open_dataset(extract_file) as ds:
            county_data = ds[variable]
            
            # Calculate area-weighted mean
            weights = np.cos(np.deg2rad(county_data.lat))
            county_mean = county_data.weighted(weights).mean(dim=['lat', 'lon'])
            
            return county_mean
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics including pre-merged data usage."""
        stats = super().get_cache_stats()
        
        stats['premerged'] = {
            'available': self.premerged_available,
            'baseline_files': len(self.baseline_merged_files),
            'county_extracts': len(self.county_extracts),
            'regional_tiles': len(self.tile_index)
        }
        
        return stats