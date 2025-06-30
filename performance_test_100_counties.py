#!/usr/bin/env python
"""Performance test with 100 counties to estimate full dataset runtime."""

import time
import pandas as pd
from pathlib import Path
from multiprocessing import cpu_count
from process_all_counties_parallel import test_seven_counties_parallel
import geopandas as gpd

def run_performance_test():
    """Run performance test with 100 counties and calculate estimates."""
    
    print("="*60)
    print("CLIMATE DATA PROCESSING PERFORMANCE TEST")
    print("="*60)
    
    # Get total county count
    shapefile_path = Path("data/shapefiles/tl_2024_us_county.shp")
    counties_gdf = gpd.read_file(shapefile_path)
    total_counties = len(counties_gdf)
    
    # Test parameters
    test_counties = 100
    test_years = 4  # 2 historical + 2 future
    total_years = 91  # 5 historical + 86 future
    
    print(f"\nDataset Information:")
    print(f"- Total counties in shapefile: {total_counties:,}")
    print(f"- Total years to process: {total_years} (2010-2014 historical + 2015-2100 ssp245)")
    print(f"- Total county-years for full dataset: {total_counties * total_years:,}")
    
    print(f"\nTest Configuration:")
    print(f"- Test counties: {test_counties}")
    print(f"- Test years: {test_years} (2010-2011 historical + 2050-2051 ssp245)")
    print(f"- Test county-years: {test_counties * test_years}")
    print(f"- Available CPUs: {cpu_count()}")
    
    # Run the test
    print(f"\nStarting performance test...")
    print("-"*60)
    
    start_time = time.time()
    
    # Run test with 100 counties and 4 years
    success = test_seven_counties_parallel(max_counties=test_counties, test_years_only=True)
    
    test_duration = time.time() - start_time
    
    if not success:
        print("ERROR: Test failed!")
        return
    
    print("-"*60)
    print(f"\nTest Results:")
    print(f"- Test duration: {test_duration:.2f} seconds ({test_duration/60:.1f} minutes)")
    print(f"- Counties processed: {test_counties}")
    print(f"- Years processed: {test_years}")
    print(f"- County-years processed: {test_counties * test_years}")
    
    # Calculate rates
    county_years_per_second = (test_counties * test_years) / test_duration
    seconds_per_county_year = test_duration / (test_counties * test_years)
    
    print(f"\nProcessing Rates:")
    print(f"- County-years per second: {county_years_per_second:.2f}")
    print(f"- Seconds per county-year: {seconds_per_county_year:.3f}")
    
    # Extrapolate to full dataset
    print(f"\n{'='*60}")
    print("EXTRAPOLATION TO FULL DATASET")
    print(f"{'='*60}")
    
    # Method 1: Linear extrapolation
    full_county_years = total_counties * total_years
    estimated_time_linear = full_county_years * seconds_per_county_year
    
    print(f"\nMethod 1: Linear Extrapolation")
    print(f"- Total county-years: {full_county_years:,}")
    print(f"- Estimated time: {estimated_time_linear:.0f} seconds")
    print(f"  = {estimated_time_linear/60:.1f} minutes")
    print(f"  = {estimated_time_linear/3600:.1f} hours")
    print(f"  = {estimated_time_linear/86400:.1f} days")
    
    # Method 2: Consider I/O overhead (more realistic)
    # Assume some fixed overhead per year for file loading
    avg_counties_per_year = total_counties
    io_overhead_factor = 1.1  # 10% overhead for I/O at scale
    
    estimated_time_with_overhead = estimated_time_linear * io_overhead_factor
    
    print(f"\nMethod 2: With I/O Overhead (10%)")
    print(f"- Estimated time: {estimated_time_with_overhead:.0f} seconds")
    print(f"  = {estimated_time_with_overhead/60:.1f} minutes")
    print(f"  = {estimated_time_with_overhead/3600:.1f} hours")
    print(f"  = {estimated_time_with_overhead/86400:.1f} days")
    
    # Method 3: Consider efficiency at different scales
    # Processing all counties per year is more efficient than our test
    efficiency_gain = 0.8  # Assume 20% efficiency gain from processing all counties together
    
    estimated_time_optimistic = estimated_time_linear * efficiency_gain
    
    print(f"\nMethod 3: Optimistic (with efficiency gains)")
    print(f"- Estimated time: {estimated_time_optimistic:.0f} seconds")
    print(f"  = {estimated_time_optimistic/60:.1f} minutes")
    print(f"  = {estimated_time_optimistic/3600:.1f} hours")
    print(f"  = {estimated_time_optimistic/86400:.1f} days")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY OF ESTIMATES")
    print(f"{'='*60}")
    print(f"\nFor processing {total_counties:,} counties × {total_years} years = {full_county_years:,} county-years:")
    print(f"\n- Optimistic estimate: {estimated_time_optimistic/3600:.1f} hours ({estimated_time_optimistic/86400:.2f} days)")
    print(f"- Realistic estimate: {estimated_time_linear/3600:.1f} hours ({estimated_time_linear/86400:.2f} days)")
    print(f"- Conservative estimate: {estimated_time_with_overhead/3600:.1f} hours ({estimated_time_with_overhead/86400:.2f} days)")
    
    # Resource usage
    print(f"\nResource Usage:")
    print(f"- CPUs utilized: {min(cpu_count(), total_years)} of {cpu_count()} available")
    print(f"- Parallelization efficiency: {min(cpu_count(), total_years)/total_years*100:.1f}%")
    
    # Save performance results
    results = {
        'test_counties': test_counties,
        'test_years': test_years,
        'test_duration_seconds': test_duration,
        'total_counties': total_counties,
        'total_years': total_years,
        'county_years_per_second': county_years_per_second,
        'estimated_hours_linear': estimated_time_linear/3600,
        'estimated_hours_with_overhead': estimated_time_with_overhead/3600,
        'estimated_hours_optimistic': estimated_time_optimistic/3600,
        'cpus_available': cpu_count(),
        'cpus_utilized': min(cpu_count(), total_years)
    }
    
    output_file = Path("results/performance_test_results.json")
    output_file.parent.mkdir(exist_ok=True)
    
    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nPerformance test results saved to: {output_file}")

if __name__ == "__main__":
    run_performance_test()