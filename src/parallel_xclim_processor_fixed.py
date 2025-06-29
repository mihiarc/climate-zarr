#!/usr/bin/env python3
"""
Fixed parallel processor with proper percentile baseline calculation
"""

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
sys.path.append('../archive')
from parallel_nex_gddp_processor import ParallelNEXGDDP_CountyProcessor

class ParallelXclimProcessorFixed(ParallelNEXGDDP_CountyProcessor):
    """
    Fixed parallel processor with proper percentile baseline
    """
    
    def __init__(self, counties_shapefile_path, base_data_path, 
                 baseline_period=(1980, 2010)):
        """
        Initialize with baseline period for percentile calculation
        
        Parameters:
        -----------
        baseline_period : tuple
            (start_year, end_year) for calculating percentile thresholds
        """
        super().__init__(counties_shapefile_path, base_data_path)
        self.baseline_period = baseline_period
        print(f"Baseline period for percentiles: {baseline_period[0]}-{baseline_period[1]}")
    
    def calculate_baseline_percentiles(self, county, variables=['tasmax', 'tasmin']):
        """
        Calculate percentile thresholds from baseline historical period
        
        Returns:
        --------
        dict
            Dictionary with day-of-year percentile thresholds
        """
        print(f"  Calculating baseline percentiles for {county['NAME']}...")
        
        thresholds = {}
        baseline_start, baseline_end = self.baseline_period
        
        for var in variables:
            file_pattern = self.base_path / var / 'historical' / f"{var}_*.nc"
            files = list(file_pattern.parent.glob(file_pattern.name))
            
            # Filter files by baseline period
            baseline_files = []
            for f in files:
                parts = f.stem.split('_')
                for part in parts:
                    if part.isdigit() and len(part) == 4:
                        year = int(part)
                        if baseline_start <= year <= baseline_end:
                            baseline_files.append(f)
                        break
            
            if len(baseline_files) >= 10:  # Need at least 10 years for good percentiles
                # Load baseline data
                ds = xr.open_mfdataset(baseline_files, combine='by_coords')
                
                # Extract county data
                county_bounds = county.geometry.bounds
                lat_slice = slice(county_bounds[1] - 0.5, county_bounds[3] + 0.5)
                
                # Convert longitude if needed
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
                
                # Compute to memory to avoid dask chunking issues
                county_mean = county_mean.compute()
                
                # Add units
                county_mean.attrs['units'] = 'K'
                
                # Calculate day-of-year percentiles
                if var == 'tasmax':
                    # Group by day of year and calculate 90th percentile
                    tasmax_grouped = county_mean.groupby('time.dayofyear')
                    thresholds['tasmax_p90_doy'] = tasmax_grouped.quantile(0.9, dim='time')
                    thresholds['tasmax_p90_doy'].attrs['units'] = 'K'
                    print(f"    Calculated tasmax 90th percentile from {len(baseline_files)} years")
                
                elif var == 'tasmin':
                    # Group by day of year and calculate 10th percentile
                    tasmin_grouped = county_mean.groupby('time.dayofyear')
                    thresholds['tasmin_p10_doy'] = tasmin_grouped.quantile(0.1, dim='time')
                    thresholds['tasmin_p10_doy'].attrs['units'] = 'K'
                    print(f"    Calculated tasmin 10th percentile from {len(baseline_files)} years")
                
                ds.close()
            else:
                print(f"    WARNING: Only {len(baseline_files)} years available for {var} baseline")
        
        return thresholds
    
    def calculate_xclim_indicators_for_county(self, county_data, thresholds):
        """
        Calculate xclim indicators using baseline thresholds
        """
        # Extract data
        tas_mean = county_data['tas']
        tasmax_mean = county_data['tasmax']
        tasmin_mean = county_data['tasmin']
        pr_mean = county_data['pr']
        
        # Create xarray DataArrays
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
        
        # 1. tx90p - using baseline thresholds
        if 'tasmax_p90_doy' in thresholds:
            indicators['tx90p'] = atmos.tx90p(tasmax_da, thresholds['tasmax_p90_doy'], freq='YS')
        
        # 2. tx_days_above 90°F (305.37 K)
        indicators['tx_days_above_90F'] = atmos.tx_days_above(tasmax_da, thresh='305.37 K', freq='YS')
        
        # 3. tn10p - using baseline thresholds
        if 'tasmin_p10_doy' in thresholds:
            indicators['tn10p'] = atmos.tn10p(tasmin_da, thresholds['tasmin_p10_doy'], freq='YS')
        
        # 4. tn_days_below 32°F (273.15 K)
        indicators['tn_days_below_32F'] = atmos.tn_days_below(tasmin_da, thresh='273.15 K', freq='YS')
        
        # 5. tg_mean
        indicators['tg_mean'] = atmos.tg_mean(tas_da, freq='YS')
        
        # 6. wetdays - Days with precip > 25.4mm
        indicators['days_precip_over_25.4mm'] = atmos.wetdays(pr_da, thresh='0.000294 kg m-2 s-1', freq='YS')
        
        # 7. precip_accumulation
        pr_mm_day = pr_da * 86400
        pr_mm_day.attrs['units'] = 'mm/day'
        indicators['precip_accumulation'] = atmos.precip_accumulation(pr_mm_day, freq='YS')
        
        return indicators
    
    def process_county_chunk_with_xclim(self, counties_chunk, scenarios, variables, 
                                       historical_period, future_period, chunk_id):
        """
        Process a chunk of counties with fixed baseline percentiles
        """
        print(f"Processing chunk {chunk_id} with {len(counties_chunk)} counties")
        
        results = []
        
        for idx, county in counties_chunk.iterrows():
            geoid = county['GEOID']
            name = county['NAME']
            
            print(f"  Processing {name} (GEOID: {geoid})")
            
            # First, calculate baseline percentile thresholds
            thresholds = self.calculate_baseline_percentiles(county)
            
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
                    scenario_files = []
                    for f in files:
                        parts = f.stem.split('_')
                        for part in parts:
                            if part.isdigit() and len(part) == 4:
                                year = int(part)
                                if start_year <= year <= end_year:
                                    scenario_files.append(f)
                                break
                    
                    if scenario_files:
                        ds = xr.open_mfdataset(scenario_files, combine='by_coords')
                        
                        # Extract county data
                        county_bounds = county.geometry.bounds
                        lat_slice = slice(county_bounds[1] - 0.5, county_bounds[3] + 0.5)
                        
                        # Convert longitude if needed
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
                    time_values = indicators['tg_mean'].time.values
                    if hasattr(time_values[0], 'year'):
                        years = [t.year for t in time_values]
                    else:
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
                                annual_result[ind_name + '_C'] = float(ind_data.isel(time=i).values) - 273.15
                            elif ind_name == 'precip_accumulation':
                                annual_result[ind_name + '_mm'] = float(ind_data.isel(time=i).values)
                            elif ind_name in ['tx90p', 'tn10p']:
                                # Convert count to percentage (divide by 365 days * 100)
                                count = float(ind_data.isel(time=i).values)
                                # Get the actual number of days in this year
                                year_start = pd.Timestamp(f'{year}-01-01')
                                year_end = pd.Timestamp(f'{year}-12-31')
                                n_days = (year_end - year_start).days + 1
                                percentage = (count / n_days) * 100
                                annual_result[ind_name + '_percent'] = percentage
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
    # Initialize processor with baseline period
    processor = ParallelXclimProcessorFixed(
        counties_shapefile_path="/home/mihiarc/repos/claude_climate/tl_2024_us_county/tl_2024_us_county.shp",
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM",
        baseline_period=(1980, 2010)  # Standard 30-year baseline
    )
    
    # Process with proper baseline
    df = processor.process_xclim_parallel(
        scenarios=['historical', 'ssp245'],
        variables=['tas', 'tasmax', 'tasmin', 'pr'],
        historical_period=(2000, 2010),  # Can be different from baseline
        future_period=(2040, 2050),
        n_chunks=4
    )
    
    # Save results
    df.to_csv("parallel_xclim_fixed_baseline.csv", index=False)
    print(f"Saved {len(df)} records with fixed baseline percentiles")