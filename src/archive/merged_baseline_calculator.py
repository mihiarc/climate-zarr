"""
Simplified climate calculator using merged baseline files.

This calculator uses pre-merged baseline cache files for ultra-fast baseline lookups.
"""

import pickle
import json
import time
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

from .optimized_climate_calculator import OptimizedClimateCalculator as ClimateCalculator


class MergedBaselineCalculator(ClimateCalculator):
    """Climate calculator optimized with merged baseline cache."""
    
    def __init__(self, 
                 base_data_path: str,
                 merged_baseline_path: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize calculator with merged baseline support.
        
        Parameters
        ----------
        base_data_path : str
            Path to climate data
        merged_baseline_path : str, optional
            Path to merged baseline pickle file
        cache_dir : str, optional
            Directory for additional caching
        """
        super().__init__(base_data_path)
        
        self.merged_baseline_path = Path(merged_baseline_path) if merged_baseline_path else None
        self.merged_baselines = None
        self.baseline_lookup = {}  # GEOID -> baseline data
        self.baseline_by_bounds = {}  # bounds_key -> baseline data
        
        # Load merged baselines if available
        if self.merged_baseline_path and self.merged_baseline_path.exists():
            self._load_merged_baselines()
    
    def _load_merged_baselines(self):
        """Load the merged baseline file into memory."""
        print(f"Loading merged baselines from {self.merged_baseline_path}")
        
        start_time = time.time()
        
        try:
            with open(self.merged_baseline_path, 'rb') as f:
                self.merged_baselines = pickle.load(f)
            
            # Create lookup indices
            if 'county_info' in self.merged_baselines:
                for geoid, info in self.merged_baselines['county_info'].items():
                    # Store baseline data by GEOID
                    self.baseline_lookup[geoid] = {
                        'tasmax_p90_doy': self.merged_baselines['tasmax_p90'].get(geoid),
                        'tasmin_p10_doy': self.merged_baselines['tasmin_p10'].get(geoid),
                        'bounds': info['bounds']
                    }
                    
                    # Also create bounds-based lookup
                    bounds_key = self._get_bounds_key(info['bounds'])
                    self.baseline_by_bounds[bounds_key] = self.baseline_lookup[geoid]
            
            elapsed = time.time() - start_time
            print(f"  Loaded {len(self.baseline_lookup)} county baselines in {elapsed:.3f}s")
            print(f"  Memory index created for instant lookups")
            
        except Exception as e:
            print(f"Error loading merged baselines: {e}")
            self.merged_baselines = None
    
    def _get_bounds_key(self, bounds: List[float]) -> str:
        """Create a key from bounds for lookup."""
        # Round to 3 decimal places for consistency
        rounded = [round(b, 3) for b in bounds]
        return f"{rounded[0]}_{rounded[1]}_{rounded[2]}_{rounded[3]}"
    
    def calculate_baseline_percentiles(self,
                                     bounds: Tuple[float, float, float, float],
                                     variables: List[str] = ['tasmax', 'tasmin'],
                                     geoid: Optional[str] = None) -> Dict:
        """
        Calculate baseline percentiles using merged data when available.
        
        Parameters
        ----------
        bounds : tuple
            (min_lon, min_lat, max_lon, max_lat)
        variables : list
            Variables to calculate percentiles for
        geoid : str, optional
            County GEOID for direct lookup
        
        Returns
        -------
        dict
            Baseline percentiles by day of year
        """
        # Try fast lookup first
        if self.merged_baselines:
            baseline_data = None
            
            # Method 1: Direct GEOID lookup (fastest)
            if geoid and geoid in self.baseline_lookup:
                print(f"  Using merged baseline for GEOID {geoid} (instant lookup)")
                baseline_data = self.baseline_lookup[geoid]
            
            # Method 2: Bounds-based lookup  
            else:
                bounds_key = self._get_bounds_key(list(bounds))
                if bounds_key in self.baseline_by_bounds:
                    print(f"  Using merged baseline for bounds (instant lookup)")
                    baseline_data = self.baseline_by_bounds[bounds_key]
            
            # Return if found
            if baseline_data:
                result = {}
                if baseline_data.get('tasmax_p90_doy') is not None:
                    result['tasmax_p90_doy'] = baseline_data['tasmax_p90_doy']
                if baseline_data.get('tasmin_p10_doy') is not None:
                    result['tasmin_p10_doy'] = baseline_data['tasmin_p10_doy']
                
                if result:
                    return result
                
            print(f"  No merged baseline found for bounds/GEOID, falling back to calculation")
        
        # Fallback to original calculation
        return super().calculate_baseline_percentiles(bounds, variables)
    
    def process_county(self,
                      county_info: Dict,
                      scenarios: List[str],
                      variables: List[str],
                      historical_period: Tuple[int, int],
                      future_period: Tuple[int, int]) -> List[Dict]:
        """
        Process county with optimized baseline lookup.
        """
        results = []
        bounds = county_info['bounds']
        geoid = county_info.get('geoid')
        
        # Calculate baseline percentiles with fast lookup
        print(f"Processing {county_info['name']}...")
        start_time = time.time()
        
        thresholds = self.calculate_baseline_percentiles(bounds, variables, geoid)
        
        baseline_time = time.time() - start_time
        print(f"  Baseline calculation: {baseline_time:.3f}s")
        
        # Process scenarios (same as parent)
        for scenario in scenarios:
            print(f"  Processing {scenario}...")
            
            # Determine time period
            if scenario == 'historical':
                start_year, end_year = historical_period
            else:
                start_year, end_year = future_period
            
            # Load data for all variables
            scenario_data = {}
            
            for var in variables:
                files = self.get_files_for_period(var, scenario, start_year, end_year)
                
                if files:
                    # Extract county data
                    county_mean = self.extract_county_data(files, var, bounds)
                    scenario_data[var] = county_mean.values
                    scenario_data['time'] = county_mean.time.values
            
            # Calculate indicators
            if all(var in scenario_data for var in variables):
                indicators = self.calculate_indicators(scenario_data, thresholds)
                
                # Create annual records
                time_values = indicators['tg_mean'].time.values
                years = [t.year for t in time_values] if hasattr(time_values[0], 'year') else \
                       pd.to_datetime(time_values).year.tolist()
                
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
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        stats = {
            'merged_baseline_loaded': self.merged_baselines is not None,
            'counties_in_lookup': len(self.baseline_lookup),
            'baseline_method': 'merged' if self.merged_baselines else 'calculated',
            'expected_speedup': '100-1000x' if self.merged_baselines else '1x'
        }
        
        if self.merged_baselines:
            # Calculate memory usage
            import sys
            baseline_size = sys.getsizeof(self.merged_baselines) / (1024 * 1024)
            stats['merged_baseline_size_mb'] = round(baseline_size, 2)
        
        return stats


class MergedBaselineProcessor:
    """Processor that uses merged baseline calculator."""
    
    def __init__(self, shapefile_path: str, base_data_path: str, 
                 merged_baseline_path: str):
        """Initialize processor with merged baseline support."""
        from parallel_processor import ParallelClimateProcessor
        
        self.base_processor = ParallelClimateProcessor(shapefile_path, base_data_path)
        self.merged_baseline_path = merged_baseline_path
        
        # Replace calculator with merged version
        self.calculator = MergedBaselineCalculator(
            base_data_path=base_data_path,
            merged_baseline_path=merged_baseline_path
        )
    
    def process_counties(self, 
                        county_ids: List[str],
                        scenarios: List[str] = ['historical', 'ssp245', 'ssp585'],
                        variables: List[str] = ['tasmax', 'tasmin', 'pr'],
                        historical_period: Tuple[int, int] = (2010, 2014),
                        future_period: Tuple[int, int] = (2040, 2070)) -> pd.DataFrame:
        """Process counties using merged baselines."""
        import pandas as pd
        
        # Get county subset
        counties_df = self.base_processor.counties[
            self.base_processor.counties['GEOID'].isin(county_ids)
        ]
        
        all_results = []
        
        print(f"\nProcessing {len(counties_df)} counties with merged baselines")
        print("=" * 60)
        
        start_time = time.time()
        
        for idx, county in counties_df.iterrows():
            county_info = self.base_processor.prepare_county_info(county)
            
            # Process with merged baseline calculator
            results = self.calculator.process_county(
                county_info=county_info,
                scenarios=scenarios,
                variables=variables,
                historical_period=historical_period,
                future_period=future_period
            )
            
            all_results.extend(results)
        
        elapsed = time.time() - start_time
        
        print(f"\nProcessing complete:")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Per county: {elapsed/len(counties_df):.2f}s")
        print(f"  Records created: {len(all_results)}")
        
        # Get performance stats
        stats = self.calculator.get_performance_stats()
        print(f"\nPerformance stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return pd.DataFrame(all_results)