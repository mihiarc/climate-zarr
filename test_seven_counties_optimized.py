#!/usr/bin/env python
"""Optimized test that loads data once and processes multiple counties."""

import pandas as pd
import xarray as xr
from pathlib import Path
import time

def process_counties_optimized(data_path, counties, years):
    """Process multiple counties efficiently by loading data only once."""
    all_results = []
    
    for year in years:
        print(f"\nLoading {year} data once for all counties...")
        load_start = time.time()
        
        # Load data files once
        tas_file = data_path / f"tas/historical/tas_day_NorESM2-LM_historical_r1i1p1f1_gn_{year}.nc"
        tasmax_file = data_path / f"tasmax/historical/tasmax_day_NorESM2-LM_historical_r1i1p1f1_gn_{year}.nc"
        tasmin_file = data_path / f"tasmin/historical/tasmin_day_NorESM2-LM_historical_r1i1p1f1_gn_{year}.nc"
        pr_file = data_path / f"pr/historical/pr_day_NorESM2-LM_historical_r1i1p1f1_gn_{year}_v1.1.nc"
        
        ds_tas = xr.open_dataset(tas_file)
        ds_tasmax = xr.open_dataset(tasmax_file)
        ds_tasmin = xr.open_dataset(tasmin_file)
        ds_pr = xr.open_dataset(pr_file)
        
        load_time = time.time() - load_start
        print(f"  Data loaded in {load_time:.2f} seconds")
        
        # Process each county with the loaded data
        for county in counties:
            process_start = time.time()
            
            # Convert bounds to 0-360 longitude if needed
            min_lon_orig = county['bounds'][0]
            max_lon_orig = county['bounds'][2]
            min_lat = county['bounds'][1]
            max_lat = county['bounds'][3]
            
            # Convert to 0-360 if longitude is negative
            if min_lon_orig < 0:
                min_lon = min_lon_orig + 360
            else:
                min_lon = min_lon_orig
                
            if max_lon_orig < 0:
                max_lon = max_lon_orig + 360
            else:
                max_lon = max_lon_orig
            
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
            all_results.append({
                'GEOID': county['geoid'],
                'NAME': county['name'],
                'STATE': county['state'],
                'REGION': county.get('region', 'Unknown'),
                'scenario': 'historical',
                'year': year,
                'tg_mean_C': float(annual_mean_c.item()),
                'tx_days_above_90F': int(days_above_90f.item()),
                'tn_days_below_0F': int(days_below_0f.item()),
                'precip_accumulation_mm': float(precip_accumulation.item()),
                'days_precip_over_25.4mm': int(days_over_25mm.item())
            })
            
            process_time = time.time() - process_start
            print(f"  Processed {county['name']}, {county['state']} in {process_time:.2f} seconds")
        
        # Close datasets after processing all counties for this year
        ds_tas.close()
        ds_tasmax.close()
        ds_tasmin.close()
        ds_pr.close()
    
    return all_results

def test_two_counties_optimized():
    """Test optimized processing of two counties."""
    
    data_path = Path("/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM")
    years = [2010, 2011, 2012, 2013, 2014]
    
    # Define counties from each region
    counties = [
        # CONUS (Continental US) - 2 counties
        {
            'geoid': '31039',
            'name': 'Cuming',
            'state': 'Nebraska',
            'bounds': (-96.7887, 41.7193, -96.1251, 42.2088),
            'region': 'CONUS'
        },
        {
            'geoid': '53069',
            'name': 'Wahkiakum',
            'state': 'Washington',
            'bounds': (-123.7268, 46.0628, -122.9593, 46.3914),
            'region': 'CONUS'
        },
        # Alaska
        {
            'geoid': '02220',
            'name': 'Sitka',
            'state': 'Alaska',
            'bounds': (-135.7195, 56.3268, -134.3315, 57.7355),
            'region': 'Alaska'
        },
        # Hawaii
        {
            'geoid': '15003',
            'name': 'Honolulu',
            'state': 'Hawaii',
            'bounds': (-158.2810, 21.2543, -157.6470, 21.7124),
            'region': 'Hawaii'
        },
        # Puerto Rico
        {
            'geoid': '72115',
            'name': 'Quebradillas',
            'state': 'Puerto Rico',
            'bounds': (-66.9856, 18.3365, -66.8541, 18.5165),
            'region': 'Puerto Rico'
        },
        # US Virgin Islands
        {
            'geoid': '78030',
            'name': 'St. Thomas',
            'state': 'US Virgin Islands',
            'bounds': (-65.0854, 18.2747, -64.6652, 18.4642),
            'region': 'US Virgin Islands'
        },
        # Guam
        {
            'geoid': '66010',
            'name': 'Guam',
            'state': 'Guam',
            'bounds': (144.6181, 13.2343, 144.9569, 13.6541),
            'region': 'Guam'
        }
    ]
    
    print("Optimized processing: 7 counties (one from each US region) with 5 years of data each")
    print(f"Counties: {[c['name'] + ', ' + c['state'] for c in counties]}")
    print(f"Years: {years}")
    print("\nStrategy: Load data once per year, process all counties")
    
    # Start timing
    start_time = time.time()
    
    # Process all counties optimized
    all_results = process_counties_optimized(data_path, counties, years)
    
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
    output_file = Path("results/test_two_counties_optimized_results.csv")
    output_file.parent.mkdir(exist_ok=True)
    results.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")
    
    return True

if __name__ == "__main__":
    success = test_two_counties_optimized()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")