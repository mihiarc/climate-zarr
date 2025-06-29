#!/usr/bin/env python3
"""
Simple test to verify annual data output with minimal processing
"""

import xarray as xr
import geopandas as gpd
import numpy as np
import pandas as pd
import regionmask
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def test_annual_processing():
    """
    Test annual data processing with 2 counties and 2 years
    """
    print("Loading counties...")
    counties = gpd.read_file("/home/mihiarc/repos/claude_climate/tl_2024_us_county/tl_2024_us_county.shp")
    counties = counties.to_crs('EPSG:4326')
    
    # Get just 2 counties
    test_counties = counties.iloc[:2].copy()
    print(f"\nTest counties:")
    for idx, county in test_counties.iterrows():
        print(f"  - {county['NAME']}, {county['STATEFP']} (GEOID: {county['GEOID']})")
    
    # Load just 2 years of data
    print("\nLoading climate data...")
    base_path = Path("/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM")
    
    # Load 2009-2010 historical data
    files = []
    for year in [2009, 2010]:
        file_path = base_path / "tas" / "historical" / f"tas_day_NorESM2-LM_historical_r1i1p1f1_gn_{year}.nc"
        if file_path.exists():
            files.append(file_path)
    
    if len(files) < 2:
        print("Could not find both 2009 and 2010 files, using available files...")
        files = list((base_path / "tas" / "historical").glob("*.nc"))[:2]
    
    print(f"Using files: {[f.name for f in files]}")
    
    # Open dataset
    ds = xr.open_mfdataset(files, combine='by_coords')
    print(f"Data shape: {ds['tas'].shape}")
    print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    
    # Create masks
    print("\nCreating county masks...")
    counties_mask = regionmask.from_geopandas(
        test_counties,
        names="GEOID",
        abbrevs="GEOID",
        name="Test_Counties"
    )
    
    mask_3D = counties_mask.mask_3D(ds)
    
    # Process each county
    results = []
    
    for i, (idx, county) in enumerate(test_counties.iterrows()):
        print(f"\nProcessing {county['NAME']}...")
        
        # Extract county data
        county_mask = mask_3D.isel(region=i)
        county_data = ds['tas'].where(county_mask)
        
        # Calculate spatial mean
        weights = np.cos(np.deg2rad(ds.lat))
        county_mean = county_data.weighted(weights).mean(dim=['lat', 'lon'])
        
        # Resample to annual
        print("  Calculating annual means...")
        annual_mean = county_mean.resample(time='YE').mean()
        
        # Extract years - handle cftime objects
        time_values = annual_mean.time.values
        if hasattr(time_values[0], 'year'):
            # cftime objects
            years = np.array([t.year for t in time_values])
        else:
            # Standard datetime
            years = pd.to_datetime(time_values).year
        
        print(f"  Years: {years.tolist()}")
        print(f"  Annual values (K): {annual_mean.values}")
        print(f"  Annual values (C): {annual_mean.values - 273.15}")
        
        # Create rows for each year
        for j, year in enumerate(years):
            row = {
                'GEOID': county['GEOID'],
                'NAME': county['NAME'],
                'STATEFP': county['STATEFP'],
                'scenario': 'historical',
                'year': year,
                'variable': 'tas',
                'value_kelvin': annual_mean.values[j],
                'value_celsius': annual_mean.values[j] - 273.15
            }
            results.append(row)
    
    # Create dataframe
    annual_df = pd.DataFrame(results)
    
    # Save results
    output_file = "test_annual_simple_results.csv"
    annual_df.to_csv(output_file, index=False)
    
    # Display results
    print("\n" + "="*60)
    print("ANNUAL RESULTS")
    print("="*60)
    print(f"\nDataframe shape: {annual_df.shape}")
    print(f"Columns: {annual_df.columns.tolist()}")
    print("\nData:")
    print(annual_df.to_string())
    
    # Show summary by county
    print("\n" + "="*60)
    print("SUMMARY BY COUNTY")
    print("="*60)
    summary = annual_df.groupby('NAME')['value_celsius'].agg(['mean', 'min', 'max'])
    print(summary)
    
    print(f"\nResults saved to: {output_file}")
    
    ds.close()

if __name__ == "__main__":
    test_annual_processing()