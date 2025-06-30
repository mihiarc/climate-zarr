#!/usr/bin/env python
"""Parallel test that processes years on different CPUs."""

import pandas as pd
import xarray as xr
from pathlib import Path
import time
from multiprocessing import Pool, cpu_count
import os
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout

def process_year(args):
    """Process all counties for a single year."""
    year, scenario, data_path, counties = args
    
    year_start = time.time()
    
    # Load data files for this year
    tas_file = data_path / f"tas/{scenario}/tas_day_NorESM2-LM_{scenario}_r1i1p1f1_gn_{year}.nc"
    tasmax_file = data_path / f"tasmax/{scenario}/tasmax_day_NorESM2-LM_{scenario}_r1i1p1f1_gn_{year}.nc"
    tasmin_file = data_path / f"tasmin/{scenario}/tasmin_day_NorESM2-LM_{scenario}_r1i1p1f1_gn_{year}.nc"
    
    # Precipitation files have _v1.1 suffix
    pr_file = data_path / f"pr/{scenario}/pr_day_NorESM2-LM_{scenario}_r1i1p1f1_gn_{year}_v1.1.nc"
    
    ds_tas = xr.open_dataset(tas_file)
    ds_tasmax = xr.open_dataset(tasmax_file)
    ds_tasmin = xr.open_dataset(tasmin_file)
    ds_pr = xr.open_dataset(pr_file)
    
    year_results = []
    
    # Process each county
    for county in counties:
        # Convert bounds to 0-360 longitude if needed
        min_lon_orig = county['bounds'][0]
        max_lon_orig = county['bounds'][2]
        min_lat = county['bounds'][1]
        max_lat = county['bounds'][3]
        
        # Convert to 0-360 if longitude is negative
        if min_lon_orig < 0:
            min_lon = min_lon_orig + 360
        else:
            min_lon = min_lon_orig
            
        if max_lon_orig < 0:
            max_lon = max_lon_orig + 360
        else:
            max_lon = max_lon_orig
        
        # Select the region for all datasets
        county_tas = ds_tas.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
        county_tasmax = ds_tasmax.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
        county_tasmin = ds_tasmin.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
        county_pr = ds_pr.sel(lon=slice(min_lon, max_lon), lat=slice(min_lat, max_lat))
        
        # Calculate area-weighted means
        weights = xr.ones_like(county_tas.tas[0])
        tas_mean = county_tas.tas.weighted(weights).mean(dim=['lat', 'lon'])
        tasmax_mean = county_tasmax.tasmax.weighted(weights).mean(dim=['lat', 'lon'])
        tasmin_mean = county_tasmin.tasmin.weighted(weights).mean(dim=['lat', 'lon'])
        pr_mean = county_pr.pr.weighted(weights).mean(dim=['lat', 'lon'])
        
        # Calculate indicators
        annual_mean_c = tas_mean.groupby('time.year').mean() - 273.15
        days_above_90f = (tasmax_mean > 305.37).groupby('time.year').sum()
        days_below_0f = (tasmin_mean < 255.37).groupby('time.year').sum()
        
        # Precipitation
        pr_mm_day = pr_mean * 86400
        precip_accumulation = pr_mm_day.groupby('time.year').sum()
        days_over_25mm = (pr_mm_day > 25.4).groupby('time.year').sum()
        
        # Store results
        year_results.append({
            'GEOID': county['geoid'],
            'NAME': county['name'],
            'STATE': county['state'],
            'REGION': 'US',  # All US counties
            'scenario': scenario,
            'year': year,
            'tg_mean_C': float(annual_mean_c.item()),
            'tx_days_above_90F': int(days_above_90f.item()),
            'tn_days_below_0F': int(days_below_0f.item()),
            'precip_accumulation_mm': float(precip_accumulation.item()),
            'days_precip_over_25.4mm': int(days_over_25mm.item())
        })
    
    # Close datasets
    ds_tas.close()
    ds_tasmax.close()
    ds_tasmin.close()
    ds_pr.close()
    
    year_time = time.time() - year_start
    
    return {'year': year, 'scenario': scenario, 'results': year_results, 'duration': year_time}

def test_seven_counties_parallel(max_counties=None, test_years_only=False):
    """Test parallel processing of counties across multiple years.
    
    Args:
        max_counties: Limit to first N counties (None for all)
        test_years_only: Use only 2010-2011 and 2050-2051 for testing
    """
    
    console = Console()
    data_path = Path("/media/mihiarc/RPA1TB/CLIMATE_DATA/NorESM2-LM")
    
    # Process historical (2010-2014) and future ssp245 (2015-2100)
    if test_years_only:
        historical_years = [2010, 2011]
        ssp245_years = [2050, 2051]
    else:
        historical_years = [2010, 2011, 2012, 2013, 2014]
        ssp245_years = list(range(2015, 2101))  # 2015 to 2100
    
    # Load all counties from shapefile
    import geopandas as gpd
    
    shapefile_path = Path("data/shapefiles/tl_2024_us_county.shp")
    
    with console.status("[bold green]Loading counties from shapefile...") as status:
        counties_gdf = gpd.read_file(shapefile_path)
        
        # Convert to list of dictionaries with bounds
        counties = []
        for idx, row in counties_gdf.iterrows():
            bounds = row.geometry.bounds  # (minx, miny, maxx, maxy)
            counties.append({
                'geoid': row['GEOID'],
                'name': row['NAME'],
                'state': row.get('STATEFP', 'Unknown'),
                'bounds': bounds
            })
        
        console.print(f"[green]✓[/green] Loaded {len(counties)} counties from shapefile")
    
    # Limit counties if requested
    if max_counties:
        counties = counties[:max_counties]
        console.print(f"[yellow]Limited to first {max_counties} counties for testing[/yellow]")
    
    # Create summary table
    summary_table = Table(title="Climate Data Processing Configuration", title_style="bold cyan")
    summary_table.add_column("Parameter", style="cyan", width=30)
    summary_table.add_column("Value", style="white")
    
    summary_table.add_row("Counties to process", f"{len(counties):,}")
    summary_table.add_row("Sample counties", f"{', '.join([c['name'] for c in counties[:3]])}...")
    summary_table.add_row("Historical years", f"{historical_years[0]}-{historical_years[-1]} ({len(historical_years)} years)")
    summary_table.add_row("SSP245 years", f"{ssp245_years[0]}-{ssp245_years[-1]} ({len(ssp245_years)} years)")
    summary_table.add_row("Total years", f"{len(historical_years) + len(ssp245_years)}")
    summary_table.add_row("Total county-years", f"{len(counties) * (len(historical_years) + len(ssp245_years)):,}")
    
    # Determine number of workers
    n_cpus = cpu_count()
    total_years = len(historical_years) + len(ssp245_years)
    n_workers = min(total_years, n_cpus)
    
    summary_table.add_row("Available CPUs", str(n_cpus))
    summary_table.add_row("Workers to use", str(n_workers))
    summary_table.add_row("Processing strategy", "One year per CPU")
    
    console.print(summary_table)
    
    # Start timing
    start_time = time.time()
    
    # Create arguments for each year and scenario
    args_list = []
    for year in historical_years:
        args_list.append((year, 'historical', data_path, counties))
    for year in ssp245_years:
        args_list.append((year, 'ssp245', data_path, counties))
    
    console.print(f"\n[bold]Processing {len(args_list)} year-scenario combinations...[/bold]\n")
    
    # Process years in parallel with progress tracking
    all_results = []
    completed_years = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[cyan]{task.completed}/{task.total} years"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("[cyan]Processing climate data...", total=len(args_list))
        
        # Use imap_unordered for better progress tracking
        with Pool(processes=n_workers) as pool:
            # Submit all tasks
            results_iter = pool.imap_unordered(process_year, args_list)
            
            # Process results as they complete
            for result in results_iter:
                year = result['year']
                scenario = result['scenario']
                year_results = result['results']
                duration = result['duration']
                
                all_results.extend(year_results)
                completed_years.append(year)
                
                progress.update(task, advance=1)
                progress.console.print(
                    f"[green]✓[/green] Year {year} ({scenario}) completed in {duration:.1f}s - "
                    f"{len(year_results)} county records"
                )
    
    # Create DataFrame
    results = pd.DataFrame(all_results)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Create results summary table
    results_table = Table(title="Processing Results", title_style="bold green")
    results_table.add_column("Metric", style="cyan", width=35)
    results_table.add_column("Value", style="white")
    
    results_table.add_row("Total processing time", f"{total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    results_table.add_row("Average time per year", f"{total_time/total_years:.2f} seconds")
    results_table.add_row("Average time per county-year", f"{total_time/(len(counties)*total_years):.3f} seconds")
    results_table.add_row("Parallel speedup", f"~{min(n_workers, total_years):.1f}x")
    results_table.add_row("", "")  # Empty row for spacing
    results_table.add_row("Total records", f"{len(results):,}")
    results_table.add_row("Counties processed", f"{results['GEOID'].nunique():,}")
    results_table.add_row("Years processed", f"{results['year'].nunique()}")
    results_table.add_row("Data shape", f"{results.shape[0]:,} rows × {results.shape[1]} columns")
    
    console.print("\n")
    console.print(results_table)
    
    # Show sample results
    if 'ssp245' in results['scenario'].unique() and 2050 in results['year'].unique():
        console.print("\n[bold cyan]Sample Results - Year 2050 (SSP245):[/bold cyan]")
        sample_2050 = results[(results['year'] == 2050) & (results['scenario'] == 'ssp245')].head()
        
        sample_table = Table(show_header=True, header_style="bold magenta")
        sample_table.add_column("County", width=20)
        sample_table.add_column("State", width=5)
        sample_table.add_column("Mean Temp (°C)", justify="right")
        sample_table.add_column("Days >90°F", justify="right")
        sample_table.add_column("Days <0°F", justify="right")
        sample_table.add_column("Precip (mm)", justify="right")
        
        for _, row in sample_2050.iterrows():
            sample_table.add_row(
                row['NAME'],
                row['STATE'],
                f"{row['tg_mean_C']:.1f}",
                str(row['tx_days_above_90F']),
                str(row['tn_days_below_0F']),
                f"{row['precip_accumulation_mm']:.0f}"
            )
        
        console.print(sample_table)
    
    # Save results with progress
    output_file = Path("results/all_counties_climate_projections_2010_2100.csv")
    output_file.parent.mkdir(exist_ok=True)
    
    with console.status("[bold green]Saving results...") as status:
        results.to_csv(output_file, index=False)
        console.print(f"\n[green]✓[/green] Results saved to: [bold]{output_file}[/bold]")
    
    return True

if __name__ == "__main__":
    import sys
    
    # Check for test mode
    if '--test' in sys.argv:
        print("Running in test mode (10 counties, 4 years)")
        success = test_seven_counties_parallel(max_counties=10, test_years_only=True)
    else:
        print("Running full processing (all counties, all years)")
        success = test_seven_counties_parallel()
    
    print(f"\nProcessing {'PASSED' if success else 'FAILED'}")