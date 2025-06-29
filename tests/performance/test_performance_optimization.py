#!/usr/bin/env python3
"""
Test and demonstrate performance optimizations.
"""

import sys
import time
import pandas as pd
sys.path.append('../src')

from parallel_processor import ParallelClimateProcessor
from optimized_parallel_processor import OptimizedParallelProcessor


def test_performance_improvements():
    """Test and compare performance of original vs optimized processors."""
    
    print("PERFORMANCE OPTIMIZATION TEST")
    print("="*60)
    
    # Configuration
    shapefile_path = "../data/shapefiles/tl_2024_us_county.shp"
    base_data_path = "/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM"
    test_counties = ['31039', '53069', '48453', '06037', '17031']  # 5 diverse counties
    test_period = (2008, 2010)  # 3 years for quick test
    
    # Load test counties
    print(f"\nTest configuration:")
    print(f"  Counties: {len(test_counties)}")
    print(f"  Period: {test_period[0]}-{test_period[1]}")
    print(f"  Scenarios: historical")
    
    # 1. Original processor
    print("\n1. Testing ORIGINAL processor...")
    processor_orig = ParallelClimateProcessor(shapefile_path, base_data_path)
    test_counties_df = processor_orig.counties[
        processor_orig.counties['GEOID'].isin(test_counties)
    ].copy()
    
    start = time.time()
    df_orig = processor_orig.process_parallel(
        counties_subset=test_counties_df,
        scenarios=['historical'],
        historical_period=test_period,
        future_period=(2040, 2042),
        n_workers=2
    )
    time_orig = time.time() - start
    
    print(f"   Time: {time_orig:.2f}s")
    print(f"   Records: {len(df_orig)}")
    print(f"   Per county: {time_orig/len(test_counties):.2f}s")
    
    # 2. Optimized processor (first run - no cache)
    print("\n2. Testing OPTIMIZED processor (first run - building cache)...")
    processor_opt = OptimizedParallelProcessor(
        shapefile_path, 
        base_data_path,
        enable_caching=True
    )
    test_counties_df_opt = processor_opt.counties[
        processor_opt.counties['GEOID'].isin(test_counties)
    ].copy()
    
    start = time.time()
    df_opt1 = processor_opt.process_parallel_optimized(
        counties_subset=test_counties_df_opt,
        scenarios=['historical'],
        historical_period=test_period,
        future_period=(2040, 2042),
        n_workers=2,
        counties_per_batch=3
    )
    time_opt1 = time.time() - start
    
    print(f"   Time: {time_opt1:.2f}s")
    print(f"   Records: {len(df_opt1)}")
    print(f"   Per county: {time_opt1/len(test_counties):.2f}s")
    print(f"   Speedup vs original: {time_orig/time_opt1:.1f}x")
    
    # 3. Optimized processor (second run - with cache)
    print("\n3. Testing OPTIMIZED processor (second run - using cache)...")
    
    start = time.time()
    df_opt2 = processor_opt.process_parallel_optimized(
        counties_subset=test_counties_df_opt,
        scenarios=['historical'],
        historical_period=test_period,
        future_period=(2040, 2042),
        n_workers=2,
        counties_per_batch=3
    )
    time_opt2 = time.time() - start
    
    print(f"   Time: {time_opt2:.2f}s")
    print(f"   Records: {len(df_opt2)}")
    print(f"   Per county: {time_opt2/len(test_counties):.2f}s")
    print(f"   Speedup vs original: {time_orig/time_opt2:.1f}x")
    print(f"   Speedup vs first run: {time_opt1/time_opt2:.1f}x")
    
    # Results comparison
    print("\n" + "-"*60)
    print("PERFORMANCE SUMMARY")
    print("-"*60)
    print(f"Original processor:        {time_orig:6.2f}s (1.0x)")
    print(f"Optimized (cold cache):    {time_opt1:6.2f}s ({time_orig/time_opt1:.1f}x)")
    print(f"Optimized (warm cache):    {time_opt2:6.2f}s ({time_orig/time_opt2:.1f}x)")
    
    # Verify results are the same
    print("\nVerifying results consistency...")
    
    # Sort both dataframes for comparison
    df_orig_sorted = df_orig.sort_values(['GEOID', 'year']).reset_index(drop=True)
    df_opt2_sorted = df_opt2.sort_values(['GEOID', 'year']).reset_index(drop=True)
    
    # Compare key columns
    cols_to_compare = ['GEOID', 'year', 'tg_mean_C', 'tx_days_above_90F']
    if all(col in df_orig_sorted.columns for col in cols_to_compare):
        differences = []
        for col in cols_to_compare:
            if col in ['tg_mean_C']:
                # Allow small floating point differences
                diff = (df_orig_sorted[col] - df_opt2_sorted[col]).abs().max()
                if diff > 0.01:
                    differences.append(f"{col}: max diff = {diff}")
            else:
                if not df_orig_sorted[col].equals(df_opt2_sorted[col]):
                    differences.append(col)
        
        if differences:
            print(f"   ⚠ Differences found: {differences}")
        else:
            print("   ✓ Results match between original and optimized processors")
    
    # Scaling estimates
    print("\n" + "-"*60)
    print("SCALING ESTIMATES")
    print("-"*60)
    
    total_counties = 3235
    counties_per_second_orig = len(test_counties) / time_orig
    counties_per_second_opt = len(test_counties) / time_opt2
    
    print(f"\nProcessing rate:")
    print(f"  Original:  {counties_per_second_orig:.3f} counties/second")
    print(f"  Optimized: {counties_per_second_opt:.3f} counties/second")
    
    print(f"\nEstimated time for all {total_counties} US counties:")
    
    for n_workers in [4, 8, 16, 32]:
        # Assume 80% parallel efficiency
        efficiency = 0.8
        
        time_orig_total = total_counties / (counties_per_second_orig * n_workers * efficiency)
        time_opt_total = total_counties / (counties_per_second_opt * n_workers * efficiency)
        
        print(f"\n  With {n_workers} workers:")
        print(f"    Original:  {time_orig_total/3600:6.1f} hours ({time_orig_total/86400:.1f} days)")
        print(f"    Optimized: {time_opt_total/3600:6.1f} hours ({time_opt_total/86400:.1f} days)")
        print(f"    Speedup:   {time_orig_total/time_opt_total:.1f}x")
    
    # Save test results for verification
    output_file = "../results/performance_test_results.csv"
    df_opt2.to_csv(output_file, index=False)
    print(f"\n✓ Test results saved to: {output_file}")


if __name__ == "__main__":
    test_performance_improvements()