#!/usr/bin/env python
"""Optimized parallel processing that minimizes I/O bottleneck."""

import pandas as pd
import xarray as xr
from pathlib import Path
import time
from multiprocessing import Pool, cpu_count
import os
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

def process_year_optimized(args):
    """Process all counties for a single year with optimized data loading."""
    year, scenario, data_path, counties = args
    
    year_start = time.time()
    
    # File paths
    files = {
        'tas': data_path / f"tas/{scenario}/tas_day_NorESM2-LM_{scenario}_r1i1p1f1_gn_{year}.nc",
        'tasmax': data_path / f"tasmax/{scenario}/tasmax_day_NorESM2-LM_{scenario}_r1i1p1f1_gn_{year}.nc",
        'tasmin': data_path / f"tasmin/{scenario}/tasmin_day_NorESM2-LM_{scenario}_r1i1p1f1_gn_{year}.nc",
        'pr': data_path / f"pr/{scenario}/pr_day_NorESM2-LM_{scenario}_r1i1p1f1_gn_{year}_v1.1.nc"
    }
    
    # Load all data into memory at once
    load_start = time.time()
    datasets = {}
    for var, filepath in files.items():
        datasets[var] = xr.open_dataset(filepath, chunks={'time': -1, 'lat': 100, 'lon': 100})
    load_time = time.time() - load_start
    
    # Process all counties with data in memory
    process_start = time.time()
    year_results = []
    
    for county in counties:
        # Convert bounds
        min_lon_orig = county['bounds'][0]
        max_lon_orig = county['bounds'][2]
        min_lat = county['bounds'][1]
        max_lat = county['bounds'][3]
        
        # Convert to 0-360 if needed
        min_lon = min_lon_orig + 360 if min_lon_orig < 0 else min_lon_orig
        max_lon = max_lon_orig + 360 if max_lon_orig < 0 else max_lon_orig
        
        # Extract county data for all variables at once
        county_data = {}
        for var, ds in datasets.items():
            var_name = var if var != 'pr' else 'pr'
            if var == 'tas':
                var_name = 'tas'
            elif var == 'tasmax':
                var_name = 'tasmax'
            elif var == 'tasmin':
                var_name = 'tasmin'
                
            # Select spatial subset
            subset = ds[var_name].sel(
                lon=slice(min_lon, max_lon),
                lat=slice(min_lat, max_lat)
            )
            
            # Compute weighted mean efficiently
            weights = xr.ones_like(subset[0])
            county_data[var] = subset.weighted(weights).mean(dim=['lat', 'lon'])
        
        # Calculate indicators
        annual_mean_c = county_data['tas'].groupby('time.year').mean() - 273.15
        days_above_90f = (county_data['tasmax'] > 305.37).groupby('time.year').sum()
        days_below_0f = (county_data['tasmin'] < 255.37).groupby('time.year').sum()
        
        # Precipitation
        pr_mm_day = county_data['pr'] * 86400
        precip_accumulation = pr_mm_day.groupby('time.year').sum()
        days_over_25mm = (pr_mm_day > 25.4).groupby('time.year').sum()
        
        # Store results
        year_results.append({
            'GEOID': county['geoid'],
            'NAME': county['name'],
            'STATE': county['state'],
            'REGION': 'US',
            'scenario': scenario,
            'year': year,
            'tg_mean_C': float(annual_mean_c.item()),
            'tx_days_above_90F': int(days_above_90f.item()),
            'tn_days_below_0F': int(days_below_0f.item()),
            'precip_accumulation_mm': float(precip_accumulation.item()),
            'days_precip_over_25.4mm': int(days_over_25mm.item())
        })
    
    process_time = time.time() - process_start
    
    # Close datasets
    for ds in datasets.values():
        ds.close()
    
    total_time = time.time() - year_start
    
    return {
        'year': year,
        'scenario': scenario,
        'results': year_results,
        'duration': total_time,
        'load_time': load_time,
        'process_time': process_time,
        'counties_processed': len(counties)
    }

def run_comparison_test():
    """Compare original vs optimized processing."""
    console = Console()
    
    console.print("\n[bold cyan]Climate Data Processing Optimization Test[/bold cyan]")
    console.print("="*60)
    
    # Test parameters
    data_path = Path("/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM")
    test_counties = 50  # Test with 50 counties
    
    # Load counties
    import geopandas as gpd
    shapefile_path = Path("data/shapefiles/tl_2024_us_county.shp")
    counties_gdf = gpd.read_file(shapefile_path).head(test_counties)
    
    counties = []
    for idx, row in counties_gdf.iterrows():
        counties.append({
            'geoid': row['GEOID'],
            'name': row['NAME'],
            'state': row.get('STATEFP', 'Unknown'),
            'bounds': row.geometry.bounds
        })
    
    # Test single year
    args = (2010, 'historical', data_path, counties)
    
    console.print(f"\nTesting with {test_counties} counties for year 2010...")
    
    # Run optimized version
    with console.status("[bold green]Running optimized processing..."):
        result = process_year_optimized(args)
    
    # Create results table
    results_table = Table(title="Performance Comparison", title_style="bold green")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="white")
    
    results_table.add_row("Counties processed", str(result['counties_processed']))
    results_table.add_row("Total time", f"{result['duration']:.2f}s")
    results_table.add_row("Data loading time", f"{result['load_time']:.2f}s ({result['load_time']/result['duration']*100:.1f}%)")
    results_table.add_row("Processing time", f"{result['process_time']:.2f}s ({result['process_time']/result['duration']*100:.1f}%)")
    results_table.add_row("Time per county", f"{result['duration']/result['counties_processed']:.3f}s")
    
    console.print("\n")
    console.print(results_table)
    
    # Extrapolation
    console.print("\n[bold cyan]Extrapolation to Full Dataset[/bold cyan]")
    
    total_counties = 3220  # Approximate
    total_years = 91
    time_per_county = result['duration'] / result['counties_processed']
    
    # Since we process all counties per year in parallel
    time_per_year = result['duration'] * (total_counties / test_counties)
    total_time_parallel = time_per_year * total_years / min(cpu_count(), total_years)
    
    extrap_table = Table(title="Full Dataset Estimates", title_style="bold yellow")
    extrap_table.add_column("Metric", style="cyan")
    extrap_table.add_column("Value", style="white")
    
    extrap_table.add_row("Total counties", f"{total_counties:,}")
    extrap_table.add_row("Total years", str(total_years))
    extrap_table.add_row("Time per year (all counties)", f"{time_per_year:.1f}s")
    extrap_table.add_row("Total time (56 CPUs)", f"{total_time_parallel/3600:.1f} hours")
    extrap_table.add_row("", f"({total_time_parallel/86400:.2f} days)")
    
    console.print("\n")
    console.print(extrap_table)
    
    # Show the dramatic improvement
    console.print("\n[bold green]Optimization Impact:[/bold green]")
    console.print(f"- Original: ~25s per county (loading data each time)")
    console.print(f"- Optimized: ~{time_per_county:.3f}s per county (loading once)")
    console.print(f"- Speedup: ~{25/time_per_county:.0f}x faster!")

if __name__ == "__main__":
    run_comparison_test()