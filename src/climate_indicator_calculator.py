#!/usr/bin/env python3
"""
Core climate indicator calculator for NEX-GDDP data.
This module handles all climate calculations without any parallel processing logic.
"""

import xarray as xr
import numpy as np
import pandas as pd
from xclim import atmos
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class ClimateIndicatorCalculator:
    """
    Core calculator for climate indicators using xclim.
    
    This class handles all climate calculations including:
    - Baseline percentile calculations
    - Climate indicator calculations
    - Data extraction and processing
    
    No parallel processing logic is included here.
    """
    
    def __init__(self, base_data_path: str, baseline_period: Tuple[int, int] = (1980, 2010)):
        """
        Initialize the climate calculator.
        
        Parameters
        ----------
        base_data_path : str
            Base path to NEX-GDDP climate data
        baseline_period : tuple
            (start_year, end_year) for calculating percentile thresholds
        """
        self.base_path = Path(base_data_path)
        self.baseline_period = baseline_period
        
    def extract_county_data(self, 
                           files: List[Path], 
                           variable: str,
                           bounds: Tuple[float, float, float, float]) -> xr.DataArray:
        """
        Extract spatially averaged data for a county from NetCDF files.
        
        Parameters
        ----------
        files : list
            List of NetCDF file paths
        variable : str
            Variable name to extract
        bounds : tuple
            (min_lon, min_lat, max_lon, max_lat) in degrees
            
        Returns
        -------
        xr.DataArray
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
    
    def get_files_for_period(self, 
                            variable: str, 
                            scenario: str, 
                            start_year: int, 
                            end_year: int) -> List[Path]:
        """
        Get list of files for a specific variable, scenario, and time period.
        
        Parameters
        ----------
        variable : str
            Climate variable name
        scenario : str
            Climate scenario (e.g., 'historical', 'ssp245')
        start_year : int
            Start year
        end_year : int
            End year
            
        Returns
        -------
        list
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
    
    def calculate_baseline_percentiles(self, 
                                     bounds: Tuple[float, float, float, float],
                                     variables: List[str] = ['tasmax', 'tasmin']) -> Dict:
        """
        Calculate percentile thresholds from baseline period.
        
        Parameters
        ----------
        bounds : tuple
            County bounds (min_lon, min_lat, max_lon, max_lat)
        variables : list
            Variables to calculate percentiles for
            
        Returns
        -------
        dict
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
                print(f"WARNING: Only {len(baseline_files)} years available for {var} baseline")
                continue
                
            # Extract data
            county_mean = self.extract_county_data(baseline_files, var, bounds)
            
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
    
    def calculate_indicators(self, 
                           data: Dict[str, np.ndarray],
                           thresholds: Dict) -> Dict:
        """
        Calculate climate indicators from data using xclim.
        
        Parameters
        ----------
        data : dict
            Dictionary with arrays for each variable and time coordinate
        thresholds : dict
            Pre-calculated baseline thresholds
            
        Returns
        -------
        dict
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
    
    def process_county(self,
                      county_info: Dict,
                      scenarios: List[str],
                      variables: List[str],
                      historical_period: Tuple[int, int],
                      future_period: Tuple[int, int]) -> List[Dict]:
        """
        Process a single county for all scenarios and time periods.
        
        Parameters
        ----------
        county_info : dict
            Dictionary with county information including bounds, name, GEOID
        scenarios : list
            List of climate scenarios
        variables : list
            List of climate variables
        historical_period : tuple
            (start_year, end_year) for historical
        future_period : tuple
            (start_year, end_year) for future
            
        Returns
        -------
        list
            List of dictionaries with annual results
        """
        results = []
        bounds = county_info['bounds']
        
        # Calculate baseline percentiles once
        print(f"  Calculating baseline for {county_info['name']}...")
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
                files = self.get_files_for_period(var, scenario, start_year, end_year)
                if files:
                    county_mean = self.extract_county_data(files, var, bounds)
                    scenario_data[var] = county_mean.values
                    scenario_data['time'] = county_mean.time.values
                    
            # Calculate indicators
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