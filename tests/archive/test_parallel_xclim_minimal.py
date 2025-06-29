#!/usr/bin/env python3
"""
Minimal test for parallel xclim processor - just 4 counties, 2 years
"""

import time
import pandas as pd
import multiprocessing as mp
from parallel_xclim_processor import ParallelXclimProcessor

def test_parallel_minimal():
    """
    Minimal test with 4 counties and 2 years per scenario
    """
    start_time = time.time()
    
    print("MINIMAL PARALLEL XCLIM TEST")
    print("="*40)
    
    # Initialize processor
    processor = ParallelXclimProcessor(
        counties_shapefile_path="/home/mihiarc/repos/claude_climate/tl_2024_us_county/tl_2024_us_county.shp",
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM"
    )
    
    # Just 4 counties - the same ones we tested before
    test_geoids = ['31039', '53069', '06037', '48201']  # Cuming NE, Wahkiakum WA, LA County CA, Harris TX
    processor.counties = processor.counties[processor.counties['GEOID'].isin(test_geoids)].copy()
    
    if len(processor.counties) == 0:
        # If exact GEOIDs not found, just take first 4
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
            historical_period=(2009, 2010),  # Just 2 years
            future_period=(2040, 2041),      # Not used for historical
            n_chunks=2  # Use 2 chunks for 4 counties
        )
        
        # Save and display results
        output_file = "test_parallel_minimal_results.csv"
        df.to_csv(output_file, index=False)
        
        print(f"\nResults shape: {df.shape}")
        print(f"Counties: {df['GEOID'].nunique()}")
        print(f"Years: {sorted(df['year'].unique())}")
        
        print("\nSample results:")
        print(df.head(8).to_string())
        
        # Quick validation
        print("\nValidation:")
        print(f"  tx90p range: {df['tx90p_percent'].min():.1f} - {df['tx90p_percent'].max():.1f}%")
        print(f"  Precip range: {df['precip_accumulation_mm'].min():.0f} - {df['precip_accumulation_mm'].max():.0f} mm")
        print(f"  Mean temp range: {df['tg_mean_C'].min():.1f} - {df['tg_mean_C'].max():.1f}°C")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    execution_time = time.time() - start_time
    print(f"\nExecution time: {execution_time:.1f} seconds")

if __name__ == "__main__":
    test_parallel_minimal()