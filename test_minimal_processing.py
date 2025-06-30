#!/usr/bin/env python
"""Minimal test to verify the package works."""

import pandas as pd
import xarray as xr
from pathlib import Path

def test_minimal_processing():
    """Test the most basic climate data processing."""
    
    # Load two years of temperature data
    data_path = Path("/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM")
    
    years = [2010, 2011, 2012, 2013, 2014]
    results_list = []
    
    for year in years:
        tas_file = data_path / f"tas/historical/tas_day_NorESM2-LM_historical_r1i1p1f1_gn_{year}.nc"
        tasmax_file = data_path / f"tasmax/historical/tasmax_day_NorESM2-LM_historical_r1i1p1f1_gn_{year}.nc"
        tasmin_file = data_path / f"tasmin/historical/tasmin_day_NorESM2-LM_historical_r1i1p1f1_gn_{year}.nc"
        pr_file = data_path / f"pr/historical/pr_day_NorESM2-LM_historical_r1i1p1f1_gn_{year}_v1.1.nc"
        
        print(f"\nLoading {year} temperature and precipitation data...")
        ds_tas = xr.open_dataset(tas_file)
        ds_tasmax = xr.open_dataset(tasmax_file)
        ds_tasmin = xr.open_dataset(tasmin_file)
        ds_pr = xr.open_dataset(pr_file)
    
        # Nebraska bounds (approximately)
        # County 31039 (Cuming County, NE) bounds: -96.7887, 41.7193, -96.1251, 42.2088
        # Convert to 0-360 longitude
        min_lon = -96.7887 + 360  # 263.2113
        max_lon = -96.1251 + 360  # 263.8749
        min_lat = 41.7193
        max_lat = 42.2088
        
        print(f"Selecting region: lon=[{min_lon:.2f}, {max_lon:.2f}], lat=[{min_lat:.2f}, {max_lat:.2f}]")
        
        # Select the region for all datasets
        county_tas = ds_tas.sel(
            lon=slice(min_lon, max_lon),
            lat=slice(min_lat, max_lat)
        )
        county_tasmax = ds_tasmax.sel(
            lon=slice(min_lon, max_lon),
            lat=slice(min_lat, max_lat)
        )
        county_tasmin = ds_tasmin.sel(
            lon=slice(min_lon, max_lon),
            lat=slice(min_lat, max_lat)
        )
        county_pr = ds_pr.sel(
            lon=slice(min_lon, max_lon),
            lat=slice(min_lat, max_lat)
        )
        
        print(f"Selected shape: {county_tas.tas.shape}")
        
        # Calculate area-weighted mean temperature and precipitation
        weights = xr.ones_like(county_tas.tas[0])  # Simple equal weights for now
        tas_mean = county_tas.tas.weighted(weights).mean(dim=['lat', 'lon'])
        tasmax_mean = county_tasmax.tasmax.weighted(weights).mean(dim=['lat', 'lon'])
        tasmin_mean = county_tasmin.tasmin.weighted(weights).mean(dim=['lat', 'lon'])
        pr_mean = county_pr.pr.weighted(weights).mean(dim=['lat', 'lon'])
        
        # Calculate annual mean temperature
        annual_mean = tas_mean.groupby('time.year').mean()
        annual_mean_c = annual_mean - 273.15
        
        # Calculate days above 90°F (32.22°C or 305.37K)
        threshold_90f_k = 305.37  # 90°F in Kelvin
        days_above_90f = (tasmax_mean > threshold_90f_k).groupby('time.year').sum()
        
        # Calculate days below 0°F (-17.78°C or 255.37K)
        threshold_0f_k = 255.37  # 0°F in Kelvin
        days_below_0f = (tasmin_mean < threshold_0f_k).groupby('time.year').sum()
        
        # Convert precipitation from kg m-2 s-1 to mm/day
        pr_mm_day = pr_mean * 86400  # 86400 seconds in a day
        
        # Calculate annual precipitation accumulation
        precip_accumulation = pr_mm_day.groupby('time.year').sum()
        
        # Calculate days with precipitation over 25.4mm (1 inch)
        threshold_precip_mm = 25.4
        days_over_25mm = (pr_mm_day > threshold_precip_mm).groupby('time.year').sum()
        
        print(f"{year} Mean Temperature: {float(annual_mean_c.item()):.2f}°C")
        print(f"{year} Days above 90°F: {int(days_above_90f.item())} days")
        print(f"{year} Days below 0°F: {int(days_below_0f.item())} days")
        print(f"{year} Total Precipitation: {float(precip_accumulation.item()):.1f} mm")
        print(f"{year} Days with >25.4mm precip: {int(days_over_25mm.item())} days")
        
        # Store results for this year - properly extract scalar values
        results_list.append({
            'GEOID': '31039',
            'NAME': 'Cuming',
            'STATE': 'Nebraska',
            'scenario': 'historical',
            'year': year,
            'tg_mean_C': float(annual_mean_c.item()),
            'tx_days_above_90F': int(days_above_90f.item()),
            'tn_days_below_0F': int(days_below_0f.item()),
            'precip_accumulation_mm': float(precip_accumulation.item()),
            'days_precip_over_25.4mm': int(days_over_25mm.item())
        })
        
        ds_tas.close()
        ds_tasmax.close()
        ds_tasmin.close()
        ds_pr.close()
    
    # Create results dataframe from all years
    results = pd.DataFrame(results_list)
    
    print("\nResults DataFrame:")
    print(results)
    
    # Save results
    output_file = Path("results/test_minimal_results.csv")
    output_file.parent.mkdir(exist_ok=True)
    results.to_csv(output_file, index=False)
    print(f"\nSaved to: {output_file}")
    
    return True

if __name__ == "__main__":
    success = test_minimal_processing()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")