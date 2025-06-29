#!/usr/bin/env python3
"""
Simplified parallel test - only fixed threshold indicators
"""

import time
import pandas as pd
import multiprocessing as mp
from parallel_xclim_processor import ParallelXclimProcessor

# Create a simplified version that only calculates fixed-threshold indicators
class SimpleParallelXclimProcessor(ParallelXclimProcessor):
    
    def calculate_xclim_indicators_for_county(self, county_data, thresholds):
        """
        Calculate only fixed-threshold xclim indicators (no percentiles)
        """
        import xarray as xr
        from xclim import atmos
        import numpy as np
        
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
        
        # Calculate only fixed-threshold indicators
        indicators = {}
        
        # Skip tx90p and tn10p for now
        
        # 2. tx_days_above 90°F (305.37 K)
        indicators['tx_days_above_90F'] = atmos.tx_days_above(tasmax_da, thresh='305.37 K', freq='YS')
        
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
        
        # Add dummy values for percentile indicators
        indicators['tx90p'] = indicators['tg_mean'] * 0 + 10  # Placeholder 10%
        indicators['tn10p'] = indicators['tg_mean'] * 0 + 10  # Placeholder 10%
        
        return indicators


def test_simple_parallel():
    """
    Test with simplified indicators
    """
    start_time = time.time()
    
    print("SIMPLIFIED PARALLEL XCLIM TEST")
    print("="*40)
    print("(Only fixed-threshold indicators)")
    print()
    
    # Initialize simplified processor
    processor = SimpleParallelXclimProcessor(
        counties_shapefile_path="/home/mihiarc/repos/claude_climate/tl_2024_us_county/tl_2024_us_county.shp",
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM"
    )
    
    # Just 4 counties
    test_geoids = ['31039', '53069', '06037', '48201']
    processor.counties = processor.counties[processor.counties['GEOID'].isin(test_geoids)].copy()
    
    if len(processor.counties) == 0:
        processor.counties = processor.all_counties.iloc[:4].copy()
    
    print(f"\nTesting with {len(processor.counties)} counties:")
    for idx, county in processor.counties.iterrows():
        print(f"  - {county['NAME']}, {county['STATEFP']}")
    
    try:
        # Process with minimal data
        print("\nProcessing...")
        df = processor.process_xclim_parallel(
            scenarios=['historical'],
            variables=['tas', 'tasmax', 'tasmin', 'pr'],
            historical_period=(2009, 2010),
            future_period=(2040, 2041),
            n_chunks=2
        )
        
        # Save and display results
        output_file = "test_parallel_simple_results.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\nResults shape: {df.shape}")
        print(f"Counties: {df['GEOID'].nunique()}")
        print(f"Years: {sorted(df['year'].unique())}")
        
        print("\nSample results:")
        print(df.head(8).to_string())
        
        # Show key indicators
        print("\nIndicator summary:")
        for county in df['NAME'].unique():
            county_data = df[df['NAME'] == county]
            print(f"\n{county}:")
            print(f"  Days > 90°F: {county_data['tx_days_above_90F'].mean():.0f} average")
            print(f"  Days < 32°F: {county_data['tn_days_below_32F'].mean():.0f} average")
            print(f"  Mean temp: {county_data['tg_mean_C'].mean():.1f}°C")
            print(f"  Annual precip: {county_data['precip_accumulation_mm'].mean():.0f} mm")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    execution_time = time.time() - start_time
    print(f"\nExecution time: {execution_time:.1f} seconds")

if __name__ == "__main__":
    test_simple_parallel()