#!/usr/bin/env python3
"""
Test xclim indicators calculation with a few counties
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
warnings.filterwarnings('ignore')

def test_xclim_indicators():
    """
    Test xclim indicators with 2 counties and 2 years
    """
    print("Loading counties...")
    counties = gpd.read_file("/home/mihiarc/repos/claude_climate/tl_2024_us_county/tl_2024_us_county.shp")
    counties = counties.to_crs('EPSG:4326')
    
    # Get just 2 counties
    test_counties = counties.iloc[:2].copy()
    print(f"\nTest counties:")
    for idx, county in test_counties.iterrows():
        print(f"  - {county['NAME']}, {county['STATEFP']} (GEOID: {county['GEOID']})")
    
    # Load data
    print("\nLoading climate data...")
    base_path = Path("/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM")
    
    # Load 10 years of data (2001-2010) for all variables
    # Helper function to find files with or without version suffix
    def find_climate_file(base_path, variable, scenario, year):
        patterns = [
            f"{variable}_day_NorESM2-LM_{scenario}_r1i1p1f1_gn_{year}.nc",
            f"{variable}_day_NorESM2-LM_{scenario}_r1i1p1f1_gn_{year}_v1.1.nc",
            f"{variable}_day_NorESM2-LM_{scenario}_r1i1p1f1_gn_{year}_v1.0.nc"
        ]
        for pattern in patterns:
            file_path = base_path / variable / scenario / pattern
            if file_path.exists():
                return file_path
        raise FileNotFoundError(f"No file found for {variable} {scenario} {year}")
    
    years = list(range(2001, 2011))  # 10 years: 2001-2010
    print(f"Loading {len(years)} years of data: {years[0]}-{years[-1]}")
    
    print("\nLoading tas (mean temperature)...")
    tas_files = [find_climate_file(base_path, "tas", "historical", year) for year in years]
    print(f"  Found {len(tas_files)} files")
    tas_ds = xr.open_mfdataset(tas_files, combine='by_coords')
    
    print("Loading tasmax (maximum temperature)...")
    tasmax_files = [find_climate_file(base_path, "tasmax", "historical", year) for year in years]
    print(f"  Found {len(tasmax_files)} files")
    tasmax_ds = xr.open_mfdataset(tasmax_files, combine='by_coords')
    
    print("Loading tasmin (minimum temperature)...")
    tasmin_files = [find_climate_file(base_path, "tasmin", "historical", year) for year in years]
    print(f"  Found {len(tasmin_files)} files")
    tasmin_ds = xr.open_mfdataset(tasmin_files, combine='by_coords')
    
    print("Loading pr (precipitation)...")
    pr_files = [find_climate_file(base_path, "pr", "historical", year) for year in years]
    print(f"  Found {len(pr_files)} files")
    pr_ds = xr.open_mfdataset(pr_files, combine='by_coords')
    
    # Create county masks
    print("\nCreating county masks...")
    counties_mask = regionmask.from_geopandas(
        test_counties,
        names="GEOID",
        abbrevs="GEOID",
        name="Test_Counties"
    )
    
    mask_3D = counties_mask.mask_3D(tas_ds)
    
    # Process each county
    results = []
    
    for i, (idx, county) in enumerate(test_counties.iterrows()):
        print(f"\nProcessing {county['NAME']}...")
        
        # Extract county mask
        county_mask = mask_3D.isel(region=i)
        
        # Extract county data
        tas_county = tas_ds['tas'].where(county_mask)
        tasmax_county = tasmax_ds['tasmax'].where(county_mask)
        tasmin_county = tasmin_ds['tasmin'].where(county_mask)
        pr_county = pr_ds['pr'].where(county_mask)
        
        # Calculate spatial mean (weighted by latitude)
        weights = np.cos(np.deg2rad(tas_ds.lat))
        
        print("  Calculating spatial means...")
        tas_mean = tas_county.weighted(weights).mean(dim=['lat', 'lon']).compute()
        tasmax_mean = tasmax_county.weighted(weights).mean(dim=['lat', 'lon']).compute()
        tasmin_mean = tasmin_county.weighted(weights).mean(dim=['lat', 'lon']).compute()
        pr_mean = pr_county.weighted(weights).mean(dim=['lat', 'lon']).compute()
        
        # Add units metadata (required by xclim)
        tas_mean.attrs['units'] = 'K'
        tasmax_mean.attrs['units'] = 'K'
        tasmin_mean.attrs['units'] = 'K'
        pr_mean.attrs['units'] = 'kg m-2 s-1'
        
        # Calculate percentile thresholds (using the same data as baseline for testing)
        print("  Calculating thresholds...")
        tasmax_p90 = tasmax_mean.quantile(0.9, dim='time')
        tasmin_p10 = tasmin_mean.quantile(0.1, dim='time')
        
        # Add units to thresholds
        tasmax_p90.attrs['units'] = 'K'
        tasmin_p10.attrs['units'] = 'K'
        
        print(f"    90th percentile of tasmax: {float(tasmax_p90.values):.2f} K ({float(tasmax_p90.values) - 273.15:.2f}°C)")
        print(f"    10th percentile of tasmin: {float(tasmin_p10.values):.2f} K ({float(tasmin_p10.values) - 273.15:.2f}°C)")
        
        # Calculate xclim indicators
        print("  Calculating indicators...")
        
        # For percentile-based indicators, calculate day-of-year percentiles
        # Group by day of year for percentile calculation
        tasmax_grouped = tasmax_mean.groupby('time.dayofyear')
        tasmin_grouped = tasmin_mean.groupby('time.dayofyear')
        
        # Calculate percentiles for each day of year
        tasmax_p90_doy = tasmax_grouped.quantile(0.9, dim='time')
        tasmin_p10_doy = tasmin_grouped.quantile(0.1, dim='time')
        
        # Add units
        tasmax_p90_doy.attrs['units'] = 'K'
        tasmin_p10_doy.attrs['units'] = 'K'
        
        # 1. tx90p - Percentage of days with tasmax > 90th percentile
        tx90p = atmos.tx90p(tasmax_mean, tasmax_p90_doy, freq='YS')
        
        # 2. tx_days_above - Days with tasmax > 90°F (32.22°C, 305.37 K)
        tx_days_above = atmos.tx_days_above(tasmax_mean, thresh='305.37 K', freq='YS')
        
        # 3. tn10p - Percentage of days with tasmin < 10th percentile
        tn10p = atmos.tn10p(tasmin_mean, tasmin_p10_doy, freq='YS')
        
        # 4. tn_days_below - Days with tasmin < 32°F (0°C, 273.15 K)
        tn_days_below = atmos.tn_days_below(tasmin_mean, thresh='273.15 K', freq='YS')
        
        # 5. tg_mean - Mean temperature
        tg_mean = atmos.tg_mean(tas_mean, freq='YS')
        
        # 6. wetdays - Days with precip > 25.4mm (1 inch)
        # Convert 25.4mm/day to kg/m2/s: 25.4 / 86400 = 0.000294 kg/m2/s
        days_over_precip = atmos.wetdays(pr_mean, thresh='0.000294 kg m-2 s-1', freq='YS')
        
        # 7. precip_accumulation - Total annual precipitation
        # First convert units from kg/m2/s to mm/day by multiplying by 86400
        # (1 kg/m2/s = 86400 mm/day)
        pr_mean_mm_day = pr_mean * 86400
        pr_mean_mm_day.attrs['units'] = 'mm/day'
        precip_accumulation = atmos.precip_accumulation(pr_mean_mm_day, freq='YS')
        
        # Extract years from the data
        years = list(range(2001, 2011))  # We loaded 2001-2010
        
        # Print results
        print(f"\n  Results for {county['NAME']}:")
        for j, year in enumerate(years):
            print(f"\n    Year {year}:")
            print(f"      tx90p: {float(tx90p.isel(time=j).values):.1f}%")
            print(f"      Days > 90°F: {float(tx_days_above.isel(time=j).values):.0f} days")
            print(f"      tn10p: {float(tn10p.isel(time=j).values):.1f}%")
            print(f"      Days < 32°F: {float(tn_days_below.isel(time=j).values):.0f} days")
            print(f"      Mean temp: {float(tg_mean.isel(time=j).values) - 273.15:.1f}°C")
            print(f"      Days > 25.4mm precip: {float(days_over_precip.isel(time=j).values):.0f} days")
            print(f"      Total precip: {float(precip_accumulation.isel(time=j).values):.1f} mm")
            
            # Add to results
            result = {
                'GEOID': county['GEOID'],
                'NAME': county['NAME'],
                'STATE': county['STATEFP'],
                'year': year,
                'tx90p_percent': float(tx90p.isel(time=j).values),
                'tx_days_above_90F': float(tx_days_above.isel(time=j).values),
                'tn10p_percent': float(tn10p.isel(time=j).values),
                'tn_days_below_32F': float(tn_days_below.isel(time=j).values),
                'tg_mean_C': float(tg_mean.isel(time=j).values) - 273.15,
                'days_precip_over_25.4mm': float(days_over_precip.isel(time=j).values),
                'precip_accumulation_mm': float(precip_accumulation.isel(time=j).values)
            }
            results.append(result)
    
    # Create and save dataframe
    df = pd.DataFrame(results)
    output_file = "test_xclim_indicators_results.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n\nResults saved to: {output_file}")
    print("\nDataFrame preview:")
    print(df.to_string())
    
    # Close datasets
    tas_ds.close()
    tasmax_ds.close()
    tasmin_ds.close()
    pr_ds.close()

if __name__ == "__main__":
    test_xclim_indicators()