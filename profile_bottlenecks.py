#!/usr/bin/env python
"""Profile the climate data processing to identify bottlenecks."""

import time
import xarray as xr
import numpy as np
from pathlib import Path
import cProfile
import pstats
from io import StringIO
import pandas as pd

def profile_single_county_year():
    """Profile processing a single county for one year to identify bottlenecks."""
    
    print("="*60)
    print("CLIMATE DATA PROCESSING BOTTLENECK ANALYSIS")
    print("="*60)
    
    # Test parameters
    data_path = Path("/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM")
    year = 2010
    scenario = 'historical'
    
    # Single county bounds (Cuming County, NE)
    county_bounds = (-96.7887, 41.7193, -96.1251, 42.2088)
    min_lon = county_bounds[0] + 360
    max_lon = county_bounds[2] + 360
    min_lat = county_bounds[1]
    max_lat = county_bounds[3]
    
    timings = {}
    
    # 1. File path construction
    start = time.time()
    tas_file = data_path / f"tas/{scenario}/tas_day_NorESM2-LM_{scenario}_r1i1p1f1_gn_{year}.nc"
    tasmax_file = data_path / f"tasmax/{scenario}/tasmax_day_NorESM2-LM_{scenario}_r1i1p1f1_gn_{year}.nc"
    tasmin_file = data_path / f"tasmin/{scenario}/tasmin_day_NorESM2-LM_{scenario}_r1i1p1f1_gn_{year}.nc"
    pr_file = data_path / f"pr/{scenario}/pr_day_NorESM2-LM_{scenario}_r1i1p1f1_gn_{year}_v1.1.nc"
    timings['file_path_construction'] = time.time() - start
    
    # 2. File opening (metadata reading)
    start = time.time()
    ds_tas = xr.open_dataset(tas_file)
    ds_tasmax = xr.open_dataset(tasmax_file)
    ds_tasmin = xr.open_dataset(tasmin_file)
    ds_pr = xr.open_dataset(pr_file)
    timings['file_opening'] = time.time() - start
    
    print(f"\nDataset shapes:")
    print(f"- TAS: {ds_tas.tas.shape} ({ds_tas.tas.nbytes / 1e9:.2f} GB)")
    print(f"- Dimensions: time={ds_tas.dims['time']}, lat={ds_tas.dims['lat']}, lon={ds_tas.dims['lon']}")
    
    # 3. Spatial selection (lazy operation)
    start = time.time()
    county_tas = ds_tas.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
    county_tasmax = ds_tasmax.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
    county_tasmin = ds_tasmin.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
    county_pr = ds_pr.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
    timings['spatial_selection'] = time.time() - start
    
    print(f"\nCounty subset shape: {county_tas.tas.shape}")
    
    # 4. Weight creation
    start = time.time()
    weights = xr.ones_like(county_tas.tas[0])
    timings['weight_creation'] = time.time() - start
    
    # 5. Weighted mean calculation (triggers actual data loading)
    start = time.time()
    tas_mean = county_tas.tas.weighted(weights).mean(dim=['lat', 'lon'])
    tas_values = tas_mean.values  # Force computation
    timings['tas_weighted_mean'] = time.time() - start
    
    start = time.time()
    tasmax_mean = county_tasmax.tasmax.weighted(weights).mean(dim=['lat', 'lon'])
    tasmax_values = tasmax_mean.values
    timings['tasmax_weighted_mean'] = time.time() - start
    
    start = time.time()
    tasmin_mean = county_tasmin.tasmin.weighted(weights).mean(dim=['lat', 'lon'])
    tasmin_values = tasmin_mean.values
    timings['tasmin_weighted_mean'] = time.time() - start
    
    start = time.time()
    pr_mean = county_pr.pr.weighted(weights).mean(dim=['lat', 'lon'])
    pr_values = pr_mean.values
    timings['pr_weighted_mean'] = time.time() - start
    
    # 6. Annual aggregations
    start = time.time()
    annual_mean = tas_mean.groupby('time.year').mean()
    annual_mean_c = annual_mean - 273.15
    annual_mean_value = float(annual_mean_c.item())
    timings['annual_mean_calc'] = time.time() - start
    
    start = time.time()
    days_above_90f = (tasmax_mean > 305.37).groupby('time.year').sum()
    days_above_value = int(days_above_90f.item())
    timings['days_above_90f_calc'] = time.time() - start
    
    start = time.time()
    days_below_0f = (tasmin_mean < 255.37).groupby('time.year').sum()
    days_below_value = int(days_below_0f.item())
    timings['days_below_0f_calc'] = time.time() - start
    
    start = time.time()
    pr_mm_day = pr_mean * 86400
    precip_accumulation = pr_mm_day.groupby('time.year').sum()
    precip_value = float(precip_accumulation.item())
    days_over_25mm = (pr_mm_day > 25.4).groupby('time.year').sum()
    days_over_value = int(days_over_25mm.item())
    timings['precipitation_calcs'] = time.time() - start
    
    # 7. File closing
    start = time.time()
    ds_tas.close()
    ds_tasmax.close()
    ds_tasmin.close()
    ds_pr.close()
    timings['file_closing'] = time.time() - start
    
    # Total time
    total_time = sum(timings.values())
    
    # Print results
    print("\n" + "="*60)
    print("TIMING BREAKDOWN")
    print("="*60)
    
    # Sort by time taken
    sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)
    
    for operation, duration in sorted_timings:
        percentage = (duration / total_time) * 100
        print(f"{operation:<30} {duration:>8.3f}s  ({percentage:>5.1f}%)")
    
    print("-"*60)
    print(f"{'TOTAL':<30} {total_time:>8.3f}s  (100.0%)")
    
    # Analysis
    print("\n" + "="*60)
    print("BOTTLENECK ANALYSIS")
    print("="*60)
    
    io_time = timings['file_opening'] + timings['tas_weighted_mean'] + timings['tasmax_weighted_mean'] + \
               timings['tasmin_weighted_mean'] + timings['pr_weighted_mean']
    compute_time = timings['annual_mean_calc'] + timings['days_above_90f_calc'] + \
                   timings['days_below_0f_calc'] + timings['precipitation_calcs']
    overhead_time = timings['file_path_construction'] + timings['spatial_selection'] + \
                    timings['weight_creation'] + timings['file_closing']
    
    print(f"I/O Operations:     {io_time:>8.3f}s  ({io_time/total_time*100:>5.1f}%)")
    print(f"Computations:       {compute_time:>8.3f}s  ({compute_time/total_time*100:>5.1f}%)")
    print(f"Overhead:           {overhead_time:>8.3f}s  ({overhead_time/total_time*100:>5.1f}%)")
    
    # Data volume analysis
    print("\n" + "="*60)
    print("DATA VOLUME ANALYSIS")
    print("="*60)
    
    # Estimate data read
    county_pixels = county_tas.tas.shape[1] * county_tas.tas.shape[2]  # lat * lon
    time_steps = county_tas.tas.shape[0]  # 365 days
    bytes_per_value = 4  # float32
    
    data_per_variable = county_pixels * time_steps * bytes_per_value
    total_data_read = data_per_variable * 4  # 4 variables
    
    print(f"County spatial extent: {county_pixels} pixels")
    print(f"Time steps: {time_steps}")
    print(f"Data read per variable: {data_per_variable / 1e6:.2f} MB")
    print(f"Total data read: {total_data_read / 1e6:.2f} MB")
    print(f"Read bandwidth: {total_data_read / io_time / 1e6:.1f} MB/s")
    
    return timings

def profile_parallel_overhead():
    """Test parallel processing overhead."""
    print("\n" + "="*60)
    print("PARALLEL PROCESSING OVERHEAD TEST")
    print("="*60)
    
    from multiprocessing import Pool
    import os
    
    def dummy_task(x):
        """Minimal task to measure overhead."""
        time.sleep(0.01)  # Simulate minimal work
        return x * 2
    
    # Test different pool sizes
    test_sizes = [1, 4, 8, 16, 32, 56]
    n_tasks = 100
    
    print(f"\nTesting with {n_tasks} tasks...")
    
    for n_workers in test_sizes:
        if n_workers > os.cpu_count():
            continue
            
        start = time.time()
        with Pool(n_workers) as pool:
            results = pool.map(dummy_task, range(n_tasks))
        duration = time.time() - start
        
        per_task = duration / n_tasks * 1000  # milliseconds
        print(f"Workers: {n_workers:>3} | Total: {duration:>6.3f}s | Per task: {per_task:>6.2f}ms")

if __name__ == "__main__":
    # Profile single county-year processing
    timings = profile_single_county_year()
    
    # Test parallel overhead
    profile_parallel_overhead()
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("\n1. Main bottleneck is I/O - reading NetCDF files")
    print("2. Each file open/read takes significant time")
    print("3. Parallel processing helps by overlapping I/O across CPUs")
    print("4. Consider:")
    print("   - Pre-loading data into memory if possible")
    print("   - Using Dask for lazy loading and chunked processing")
    print("   - Caching frequently accessed data")
    print("   - Using SSD storage for climate data if not already")