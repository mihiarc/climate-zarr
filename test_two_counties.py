#!/usr/bin/env python
"""Test climate data processing for two counties to compare runtime."""

import pandas as pd
import xarray as xr
from pathlib import Path
import time

def process_county(data_path, county_info, years):
    """Process climate data for a single county."""
    results_list = []
    
    for year in years:
        tas_file = data_path / f"tas/historical/tas_day_NorESM2-LM_historical_r1i1p1f1_gn_{year}.nc"
        tasmax_file = data_path / f"tasmax/historical/tasmax_day_NorESM2-LM_historical_r1i1p1f1_gn_{year}.nc"
        tasmin_file = data_path / f"tasmin/historical/tasmin_day_NorESM2-LM_historical_r1i1p1f1_gn_{year}.nc"
        pr_file = data_path / f"pr/historical/pr_day_NorESM2-LM_historical_r1i1p1f1_gn_{year}_v1.1.nc"
        
        # Load data
        ds_tas = xr.open_dataset(tas_file)
        ds_tasmax = xr.open_dataset(tasmax_file)
        ds_tasmin = xr.open_dataset(tasmin_file)
        ds_pr = xr.open_dataset(pr_file)
        
        # Convert bounds to 0-360 longitude
        min_lon = county_info['bounds'][0] + 360
        max_lon = county_info['bounds'][2] + 360
        min_lat = county_info['bounds'][1]
        max_lat = county_info['bounds'][3]
        
        # Select the region for all datasets
        county_tas = ds_tas.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
        county_tasmax = ds_tasmax.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
        county_tasmin = ds_tasmin.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
        county_pr = ds_pr.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
        
        # Calculate area-weighted means
        weights = xr.ones_like(county_tas.tas[0])
        tas_mean = county_tas.tas.weighted(weights).mean(dim=['lat', 'lon'])
        tasmax_mean = county_tasmax.tasmax.weighted(weights).mean(dim=['lat', 'lon'])
        tasmin_mean = county_tasmin.tasmin.weighted(weights).mean(dim=['lat', 'lon'])
        pr_mean = county_pr.pr.weighted(weights).mean(dim=['lat', 'lon'])
        
        # Calculate indicators
        annual_mean_c = tas_mean.groupby('time.year').mean() - 273.15
        days_above_90f = (tasmax_mean > 305.37).groupby('time.year').sum()
        days_below_0f = (tasmin_mean < 255.37).groupby('time.year').sum()
        
        # Precipitation
        pr_mm_day = pr_mean * 86400
        precip_accumulation = pr_mm_day.groupby('time.year').sum()
        days_over_25mm = (pr_mm_day > 25.4).groupby('time.year').sum()
        
        # Store results - properly extract scalar values
        results_list.append({
            'GEOID': county_info['geoid'],
            'NAME': county_info['name'],
            'STATE': county_info['state'],
            'scenario': 'historical',
            'year': year,
            'tg_mean_C': float(annual_mean_c.item()),
            'tx_days_above_90F': int(days_above_90f.item()),
            'tn_days_below_0F': int(days_below_0f.item()),
            'precip_accumulation_mm': float(precip_accumulation.item()),
            'days_precip_over_25.4mm': int(days_over_25mm.item())
        })
        
        # Close datasets
        ds_tas.close()
        ds_tasmax.close()
        ds_tasmin.close()
        ds_pr.close()
    
    return results_list

def test_two_counties():
    """Test processing two counties."""
    
    data_path = Path("/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM")
    years = [2010, 2011, 2012, 2013, 2014]
    
    # Define two counties
    counties = [
        {
            'geoid': '31039',
            'name': 'Cuming',
            'state': 'Nebraska',
            'bounds': (-96.7887, 41.7193, -96.1251, 42.2088)
        },
        {
            'geoid': '53069',
            'name': 'Wahkiakum',
            'state': 'Washington',
            'bounds': (-123.7268, 46.0628, -122.9593, 46.3914)
        }
    ]
    
    print("Processing 2 counties with 5 years of data each...")
    print(f"Counties: {[c['name'] + ', ' + c['state'] for c in counties]}")
    print(f"Years: {years}")
    
    # Start timing
    start_time = time.time()
    
    # Process all counties
    all_results = []
    for i, county in enumerate(counties):
        county_start = time.time()
        print(f"\nProcessing county {i+1}/2: {county['name']}, {county['state']}...")
        
        county_results = process_county(data_path, county, years)
        all_results.extend(county_results)
        
        county_time = time.time() - county_start
        print(f"  County processed in {county_time:.2f} seconds")
    
    # Create DataFrame
    results = pd.DataFrame(all_results)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print(f"Average time per county: {total_time/len(counties):.2f} seconds")
    print(f"Average time per county-year: {total_time/(len(counties)*len(years)):.2f} seconds")
    
    print("\nResults DataFrame shape:", results.shape)
    print("\nFirst few rows:")
    print(results.head())
    
    print("\nLast few rows:")
    print(results.tail())
    
    # Save results
    output_file = Path("results/test_two_counties_results.csv")
    output_file.parent.mkdir(exist_ok=True)
    results.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")
    
    return True

if __name__ == "__main__":
    success = test_two_counties()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")