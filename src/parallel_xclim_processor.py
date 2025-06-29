#!/usr/bin/env python3
"""
Parallel processor with xclim indicators integration

DEPRECATED: This module is deprecated in favor of parallel_xclim_processor_unified.py
Please use the unified processor which supports both fixed and period-specific baselines.
"""

import warnings
warnings.warn(
    "parallel_xclim_processor.py is deprecated. "
    "Please use parallel_xclim_processor_unified.py instead. "
    "See MIGRATION_GUIDE.md for details.",
    DeprecationWarning,
    stacklevel=2
)

import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd
import regionmask
import xclim
from xclim import atmos
from pathlib import Path
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from functools import partial

warnings.filterwarnings('ignore')

# Import the parallel processor from archive
import sys
sys.path.append('./archive')
from parallel_nex_gddp_processor import ParallelNEXGDDP_CountyProcessor

class ParallelXclimProcessor(ParallelNEXGDDP_CountyProcessor):
    """
    Extended parallel processor with xclim indicators
    """
    
    def calculate_xclim_indicators_for_county(self, county_data, thresholds):
        """
        Calculate xclim indicators for a single county's data
        
        Parameters:
        -----------
        county_data : dict
            Dictionary with time series data for all variables
        thresholds : dict
            Pre-calculated thresholds (percentiles, etc.)
        
        Returns:
        --------
        dict
            Annual indicator values
        """
        # Extract data
        tas_mean = county_data['tas']
        tasmax_mean = county_data['tasmax']
        tasmin_mean = county_data['tasmin']
        pr_mean = county_data['pr']
        
        # Create xarray DataArrays
        # Use the time coordinate directly without converting (handles cftime)
        time_coord = county_data['time']
        
        tas_da = xr.DataArray(tas_mean, dims=['time'], coords={'time': time_coord})
        tasmax_da = xr.DataArray(tasmax_mean, dims=['time'], coords={'time': time_coord})
        tasmin_da = xr.DataArray(tasmin_mean, dims=['time'], coords={'time': time_coord})
        pr_da = xr.DataArray(pr_mean, dims=['time'], coords={'time': time_coord})
        
        # Add units attributes
        tas_da.attrs['units'] = 'K'
        tasmax_da.attrs['units'] = 'K' 
        tasmin_da.attrs['units'] = 'K'
        pr_da.attrs['units'] = 'kg m-2 s-1'
        
        # Calculate indicators
        indicators = {}
        
        # 1. tx90p - using pre-calculated threshold
        if 'tasmax_p90' in thresholds:
            indicators['tx90p'] = atmos.tx90p(tasmax_da, thresholds['tasmax_p90'], freq='YS')
        
        # 2. tx_days_above 90°F (305.37 K)
        indicators['tx_days_above_90F'] = atmos.tx_days_above(tasmax_da, thresh='305.37 K', freq='YS')
        
        # 3. tn10p - using pre-calculated threshold
        if 'tasmin_p10' in thresholds:
            indicators['tn10p'] = atmos.tn10p(tasmin_da, thresholds['tasmin_p10'], freq='YS')
        
        # 4. tn_days_below 32°F (273.15 K)
        indicators['tn_days_below_32F'] = atmos.tn_days_below(tasmin_da, thresh='273.15 K', freq='YS')
        
        # 5. tg_mean
        indicators['tg_mean'] = atmos.tg_mean(tas_da, freq='YS')
        
        # 6. wetdays - Days with precip > 25.4mm (0.000294 kg/m2/s)
        indicators['days_precip_over_25.4mm'] = atmos.wetdays(
            pr_da, thresh='0.000294 kg m-2 s-1', freq='YS'
        )
        
        # 7. precip_accumulation
        # Convert to mm/day first
        pr_mm_day = pr_da * 86400
        pr_mm_day.attrs['units'] = 'mm/day'
        indicators['precip_accumulation'] = atmos.precip_accumulation(pr_mm_day, freq='YS')
        
        return indicators
    
    def process_county_chunk_with_xclim(self, counties_chunk, scenarios, variables, 
                                       historical_period, future_period, chunk_id):
        """
        Process a chunk of counties with xclim indicators
        """
        print(f"Processing chunk {chunk_id} with {len(counties_chunk)} counties")
        
        results = []
        
        for idx, county in counties_chunk.iterrows():
            geoid = county['GEOID']
            name = county['NAME']
            
            print(f"  Processing {name} (GEOID: {geoid})")
            
            county_results = {
                'GEOID': geoid,
                'NAME': name,
                'STATE': county.get('STATEFP', 'Unknown')
            }
            
            # First, load historical data to calculate thresholds
            hist_data = {}
            hist_time = None
            for var in variables:
                file_pattern = self.base_path / var / 'historical' / f"{var}_*.nc"
                files = list(file_pattern.parent.glob(file_pattern.name))
                
                # Filter files by year range
                # Extract year from filename, handling version suffixes like _v1.1
                hist_files = []
                for f in files:
                    parts = f.stem.split('_')
                    # Find the year (4-digit number)
                    for part in parts:
                        if part.isdigit() and len(part) == 4:
                            year = int(part)
                            if historical_period[0] <= year <= historical_period[1]:
                                hist_files.append(f)
                            break
                
                if hist_files:
                    ds = xr.open_mfdataset(hist_files, combine='by_coords')
                    
                    # Extract county data using mask
                    county_bounds = county.geometry.bounds
                    lat_slice = slice(county_bounds[1] - 0.5, county_bounds[3] + 0.5)
                    
                    # Convert longitude from -180/180 to 0/360 if needed
                    lon_min = county_bounds[0]
                    lon_max = county_bounds[2]
                    if lon_min < 0:
                        lon_min = lon_min % 360
                    if lon_max < 0:
                        lon_max = lon_max % 360
                    lon_slice = slice(lon_min - 0.5, lon_max + 0.5)
                    
                    county_data = ds[var].sel(lat=lat_slice, lon=lon_slice)
                    
                    # Calculate spatial mean
                    weights = np.cos(np.deg2rad(county_data.lat))
                    county_mean = county_data.weighted(weights).mean(dim=['lat', 'lon'])
                    
                    hist_data[var] = county_mean.values
                    if hist_time is None:
                        hist_time = county_mean.time.values
                    ds.close()
            
            # Calculate thresholds from historical data
            # For percentile-based indicators, we need day-of-year thresholds
            thresholds = {}
            
            # Create DataArrays from historical data for threshold calculation
            if 'tasmax' in hist_data and len(hist_data['tasmax']) > 0 and hist_time is not None:
                tasmax_hist_da = xr.DataArray(
                    hist_data['tasmax'], 
                    dims=['time'], 
                    coords={'time': hist_time}
                )
                tasmax_hist_da.attrs['units'] = 'K'
                # Group by day of year and calculate 90th percentile
                tasmax_grouped = tasmax_hist_da.groupby('time.dayofyear')
                thresholds['tasmax_p90'] = tasmax_grouped.quantile(0.9, dim='time')
                thresholds['tasmax_p90'].attrs['units'] = 'K'
            
            if 'tasmin' in hist_data and len(hist_data['tasmin']) > 0 and hist_time is not None:
                tasmin_hist_da = xr.DataArray(
                    hist_data['tasmin'],
                    dims=['time'],
                    coords={'time': hist_time}
                )
                tasmin_hist_da.attrs['units'] = 'K'
                # Group by day of year and calculate 10th percentile
                tasmin_grouped = tasmin_hist_da.groupby('time.dayofyear')
                thresholds['tasmin_p10'] = tasmin_grouped.quantile(0.1, dim='time')
                thresholds['tasmin_p10'].attrs['units'] = 'K'
            
            # Process each scenario
            for scenario in scenarios:
                scenario_data = {}
                
                # Determine time period
                if scenario == 'historical':
                    start_year, end_year = historical_period
                else:
                    start_year, end_year = future_period
                
                # Load data for all variables
                for var in variables:
                    file_pattern = self.base_path / var / scenario / f"{var}_*.nc"
                    files = list(file_pattern.parent.glob(file_pattern.name))
                    
                    # Filter files by year range
                    # Extract year from filename, handling version suffixes like _v1.1
                    scenario_files = []
                    for f in files:
                        parts = f.stem.split('_')
                        # Find the year (4-digit number)
                        for part in parts:
                            if part.isdigit() and len(part) == 4:
                                year = int(part)
                                if start_year <= year <= end_year:
                                    scenario_files.append(f)
                                break
                    
                    if scenario_files:
                        ds = xr.open_mfdataset(scenario_files, combine='by_coords')
                        
                        # Extract county data using corrected bounds
                        county_bounds = county.geometry.bounds
                        lat_slice = slice(county_bounds[1] - 0.5, county_bounds[3] + 0.5)
                        
                        # Convert longitude from -180/180 to 0/360 if needed
                        lon_min = county_bounds[0]
                        lon_max = county_bounds[2]
                        if lon_min < 0:
                            lon_min = lon_min % 360
                        if lon_max < 0:
                            lon_max = lon_max % 360
                        lon_slice = slice(lon_min - 0.5, lon_max + 0.5)
                        
                        county_data = ds[var].sel(lat=lat_slice, lon=lon_slice)
                        weights = np.cos(np.deg2rad(county_data.lat))
                        county_mean = county_data.weighted(weights).mean(dim=['lat', 'lon'])
                        
                        scenario_data[var] = county_mean.values
                        scenario_data['time'] = county_mean.time.values
                        
                        ds.close()
                
                # Calculate xclim indicators
                if all(var in scenario_data for var in variables):
                    indicators = self.calculate_xclim_indicators_for_county(scenario_data, thresholds)
                    
                    # Extract annual values
                    # Handle cftime objects
                    time_values = indicators['tg_mean'].time.values
                    if hasattr(time_values[0], 'year'):
                        # cftime objects
                        years = [t.year for t in time_values]
                    else:
                        # Standard datetime
                        years = pd.to_datetime(time_values).year.tolist()
                    
                    for i, year in enumerate(years):
                        annual_result = {
                            'GEOID': geoid,
                            'NAME': name,
                            'STATE': county.get('STATEFP', 'Unknown'),
                            'scenario': scenario,
                            'year': year
                        }
                        
                        # Add indicator values
                        for ind_name, ind_data in indicators.items():
                            if ind_name == 'tg_mean':
                                # Convert temperature from K to C
                                annual_result[ind_name + '_C'] = float(ind_data.isel(time=i).values) - 273.15
                            elif ind_name == 'precip_accumulation':
                                # Already in mm from the calculation
                                annual_result[ind_name + '_mm'] = float(ind_data.isel(time=i).values)
                            elif ind_name in ['tx90p', 'tn10p']:
                                annual_result[ind_name + '_percent'] = float(ind_data.isel(time=i).values)
                            else:
                                annual_result[ind_name] = float(ind_data.isel(time=i).values)
                        
                        results.append(annual_result)
        
        return results
    
    def process_xclim_parallel(self, scenarios=['historical', 'ssp245'], 
                              variables=['tas', 'tasmax', 'tasmin', 'pr'],
                              historical_period=(1980, 2010),
                              future_period=(2040, 2070),
                              n_chunks=None):
        """
        Process all counties in parallel to calculate xclim indicators
        """
        if n_chunks is None:
            n_chunks = min(mp.cpu_count(), 16)
        
        print(f"Processing {len(self.counties)} counties using {n_chunks} parallel chunks")
        
        # Divide counties into chunks
        county_chunks = np.array_split(self.counties, n_chunks)
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=n_chunks) as executor:
            futures = []
            
            for i, chunk in enumerate(county_chunks):
                future = executor.submit(
                    self.process_county_chunk_with_xclim,
                    chunk, scenarios, variables, 
                    historical_period, future_period, i
                )
                futures.append(future)
            
            # Collect results
            all_results = []
            for future in futures:
                chunk_results = future.result()
                all_results.extend(chunk_results)
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        return df


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = ParallelXclimProcessor(
        counties_shapefile_path="/home/mihiarc/repos/claude_climate/tl_2024_us_county/tl_2024_us_county.shp",
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM"
    )
    
    # Process with xclim indicators
    df = processor.process_xclim_parallel(
        scenarios=['historical', 'ssp245'],
        variables=['tas', 'tasmax', 'tasmin', 'pr'],
        historical_period=(1980, 2010),
        future_period=(2040, 2070),
        n_chunks=16  # Use 16 cores
    )
    
    # Save results
    df.to_csv("parallel_xclim_indicators.csv", index=False)
    print(f"Saved {len(df)} records to parallel_xclim_indicators.csv")