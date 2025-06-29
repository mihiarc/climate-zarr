#!/usr/bin/env python3
"""
Test script for parallel xclim processor with a subset of counties
"""

import time
import pandas as pd
import multiprocessing as mp
from parallel_xclim_processor import ParallelXclimProcessor

def test_parallel_xclim():
    """
    Test parallel xclim processing with 20 counties across different states
    """
    start_time = time.time()
    
    print("="*60)
    print("TESTING PARALLEL XCLIM PROCESSOR")
    print("="*60)
    
    # Initialize processor
    print("\nInitializing processor...")
    processor = ParallelXclimProcessor(
        counties_shapefile_path="/home/mihiarc/repos/claude_climate/tl_2024_us_county/tl_2024_us_county.shp",
        base_data_path="/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM"
    )
    
    # Limit to 20 counties for testing (diverse selection)
    # Select counties from different states to test geographic diversity
    test_counties = processor.counties[
        processor.counties['STATEFP'].isin(['06', '31', '53', '48', '32'])  # CA, NE, WA, TX, NV
    ].iloc[:20].copy()
    
    processor.counties = test_counties
    
    print(f"\nTest configuration:")
    print(f"  Counties: {len(processor.counties)}")
    print(f"  Available CPUs: {mp.cpu_count()}")
    print(f"  Using chunks: 4")
    
    print("\nSelected counties:")
    for idx, county in processor.counties.iterrows():
        print(f"  - {county['NAME']}, {county['STATEFP']} (GEOID: {county['GEOID']})")
    
    # Test with shorter time periods
    print("\n" + "="*60)
    print("PROCESSING CLIMATE INDICATORS")
    print("="*60)
    
    try:
        # Process with parallel xclim
        df = processor.process_xclim_parallel(
            scenarios=['historical', 'ssp245'],
            variables=['tas', 'tasmax', 'tasmin', 'pr'],
            historical_period=(2005, 2010),  # 6 years for faster testing
            future_period=(2040, 2045),      # 6 years
            n_chunks=4  # Use 4 parallel chunks
        )
        
        # Save results
        output_file = "test_parallel_xclim_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        # Analyze results
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        
        print(f"\nTotal records: {len(df)}")
        print(f"Counties processed: {df['GEOID'].nunique()}")
        print(f"Scenarios: {df['scenario'].unique().tolist()}")
        print(f"Years: {sorted(df['year'].unique())}")
        
        # Show sample data
        print("\nSample data (first 5 rows):")
        print(df.head().to_string())
        
        # Summary statistics by scenario
        print("\n" + "-"*40)
        print("INDICATOR AVERAGES BY SCENARIO")
        print("-"*40)
        
        scenario_summary = df.groupby('scenario').agg({
            'tx90p_percent': 'mean',
            'tx_days_above_90F': 'mean',
            'tn10p_percent': 'mean',
            'tn_days_below_32F': 'mean',
            'tg_mean_C': 'mean',
            'days_precip_over_25.4mm': 'mean',
            'precip_accumulation_mm': 'mean'
        }).round(1)
        
        print(scenario_summary.to_string())
        
        # Check for data quality
        print("\n" + "-"*40)
        print("DATA QUALITY CHECKS")
        print("-"*40)
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            print("Missing values found:")
            print(missing[missing > 0])
        else:
            print("✓ No missing values")
        
        # Check percentile ranges
        tx90p_check = (df['tx90p_percent'] >= 0) & (df['tx90p_percent'] <= 100)
        tn10p_check = (df['tn10p_percent'] >= 0) & (df['tn10p_percent'] <= 100)
        
        if tx90p_check.all() and tn10p_check.all():
            print("✓ Percentile values within valid range (0-100%)")
        else:
            print("✗ Invalid percentile values found!")
        
        # Check precipitation values
        precip_check = (df['precip_accumulation_mm'] > 0) & (df['precip_accumulation_mm'] < 10000)
        if precip_check.all():
            print("✓ Precipitation values within reasonable range")
        else:
            print("✗ Unreasonable precipitation values found!")
        
        # Show example county comparison
        print("\n" + "-"*40)
        print("EXAMPLE COUNTY COMPARISON")
        print("-"*40)
        
        # Pick one county to show historical vs future
        example_geoid = df['GEOID'].iloc[0]
        example_name = df[df['GEOID'] == example_geoid]['NAME'].iloc[0]
        
        county_data = df[df['GEOID'] == example_geoid].copy()
        
        print(f"\nCounty: {example_name} (GEOID: {example_geoid})")
        
        # Calculate period averages
        hist_avg = county_data[county_data['scenario'] == 'historical'].mean(numeric_only=True)
        fut_avg = county_data[county_data['scenario'] == 'ssp245'].mean(numeric_only=True)
        
        print("\nAverage values by period:")
        print(f"{'Indicator':<30} {'Historical':>12} {'SSP2-4.5':>12} {'Change':>12}")
        print("-" * 70)
        
        indicators = [
            ('Mean Temperature (°C)', 'tg_mean_C', '{:.1f}'),
            ('Days > 90°F', 'tx_days_above_90F', '{:.0f}'),
            ('Days < 32°F', 'tn_days_below_32F', '{:.0f}'),
            ('Annual Precipitation (mm)', 'precip_accumulation_mm', '{:.0f}'),
            ('Days > 25.4mm rain', 'days_precip_over_25.4mm', '{:.0f}')
        ]
        
        for label, col, fmt in indicators:
            hist_val = hist_avg[col]
            fut_val = fut_avg[col]
            change = fut_val - hist_val
            print(f"{label:<30} {fmt.format(hist_val):>12} {fmt.format(fut_val):>12} {fmt.format(change):>12}")
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
    
    # Print execution time
    execution_time = time.time() - start_time
    print(f"\n" + "="*60)
    print(f"Total execution time: {execution_time:.1f} seconds ({execution_time/60:.1f} minutes)")
    print(f"Time per county: {execution_time/len(processor.counties):.1f} seconds")
    print("="*60)

if __name__ == "__main__":
    test_parallel_xclim()