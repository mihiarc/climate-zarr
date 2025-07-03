#!/usr/bin/env python
"""Calculate county statistics with parallel processing."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

try:
    from dask.distributed import Client, LocalCluster
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False
    warnings.warn("Dask distributed not available. Parallel processing will use multiprocessing instead.")

warnings.filterwarnings('ignore', category=RuntimeWarning)

console = Console()


class ModernCountyProcessor:
    """County statistics processor."""
    
    def __init__(
        self,
        n_workers: int = 4,
        memory_limit: str = "4GB",
        use_distributed: bool = False
    ):
        self.n_workers = n_workers
        self.memory_limit = memory_limit
        self.use_distributed = use_distributed
        self.client = None
        
        if use_distributed:
            self._setup_dask_client()
    
    def _setup_dask_client(self):
        """Setup Dask distributed client for parallel processing."""
        if not DISTRIBUTED_AVAILABLE:
            console.print("[yellow]Dask distributed not available, using multiprocessing instead[/yellow]")
            self.client = None
            return
        
        try:
            cluster = LocalCluster(
                n_workers=self.n_workers,
                threads_per_worker=2,
                memory_limit=self.memory_limit,
                silence_logs=False
            )
            self.client = Client(cluster)
            console.print(f"[green]Dask client started: {self.client.dashboard_link}[/green]")
        except Exception as e:
            console.print(f"[yellow]Failed to start Dask client: {e}[/yellow]")
            console.print("[yellow]Falling back to multiprocessing[/yellow]")
            self.client = None
    
    def prepare_shapefile(
        self, 
        shapefile_path: Path, 
        target_crs: str = 'EPSG:4326'
    ) -> gpd.GeoDataFrame:
        """Load and prepare shapefile."""
        console.print(f"[blue]Loading shapefile:[/blue] {shapefile_path}")
        
        # Load with optimizations
        gdf = gpd.read_file(shapefile_path)
        
        # Optimize geometry column
        gdf.geometry = gdf.geometry.simplify(0.001)  # Slight simplification for speed
        
        # Convert to target CRS if needed
        if gdf.crs.to_string() != target_crs:
            console.print(f"[yellow]Converting CRS from {gdf.crs} to {target_crs}[/yellow]")
            gdf = gdf.to_crs(target_crs)
        
        # Standardize column names
        gdf = self._standardize_columns(gdf)
        
        # Add spatial index for faster operations
        gdf.sindex  # Force creation of spatial index
        
        return gdf
    
    def _standardize_columns(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Standardize column names and add required fields."""
        # County identifier
        if 'GEOID' in gdf.columns:
            gdf['county_id'] = gdf['GEOID']
        elif 'FIPS' in gdf.columns:
            gdf['county_id'] = gdf['FIPS']
        else:
            gdf['county_id'] = gdf.index.astype(str)
        
        # County name
        if 'NAME' in gdf.columns:
            gdf['county_name'] = gdf['NAME']
        elif 'NAMELSAD' in gdf.columns:
            gdf['county_name'] = gdf['NAMELSAD']
        else:
            gdf['county_name'] = gdf['county_id']
        
        # State
        if 'STUSPS' in gdf.columns:
            gdf['state'] = gdf['STUSPS']
        elif 'STATE_NAME' in gdf.columns:
            gdf['state'] = gdf['STATE_NAME']
        else:
            gdf['state'] = ''
        
        # Add numeric index for vectorized operations
        gdf['raster_id'] = range(1, len(gdf) + 1)
        
        return gdf[['county_id', 'county_name', 'state', 'raster_id', 'geometry']]
    
    def process_zarr_data(
        self,
        zarr_path: Path,
        gdf: gpd.GeoDataFrame,
        scenario: str = 'historical', # sets default scenario
        variable: str = 'pr', # sets default variable
        threshold_mm: float = 25.4, # sets default threshold
        chunk_by_county: bool = True # sets default chunking by county
    ) -> pd.DataFrame:
        """Process Zarr data."""
        
        console.print(f"[blue]Opening Zarr dataset:[/blue] {zarr_path}")
        
        # Open with optimizations
        ds = xr.open_zarr(zarr_path, chunks={'time': 365})
        
        # Determine processing type based on variable
        if variable == 'pr':
            return self._process_precipitation_data(ds, gdf, scenario, threshold_mm, chunk_by_county)
        elif variable == 'tas':
            return self._process_temperature_data(ds, gdf, scenario, chunk_by_county)
        elif variable == 'tasmax':
            return self._process_tasmax_data(ds, gdf, scenario, threshold_mm, chunk_by_county)
        elif variable == 'tasmin':
            return self._process_tasmin_data(ds, gdf, scenario, chunk_by_county)
        else:
            raise ValueError(f"Unsupported variable: {variable}")
    
    def _process_precipitation_data(
        self,
        ds: xr.Dataset,
        gdf: gpd.GeoDataFrame,
        scenario: str,
        threshold_mm: float,
        chunk_by_county: bool
    ) -> pd.DataFrame:
        """Process precipitation data."""
        # Get precipitation data
        pr_data = ds['pr']
        
        # Convert from kg/m²/s to mm/day
        # 1 kg/m²/s = 86400 mm/day (86400 seconds per day, 1 kg/m² = 1 mm)
        console.print("[yellow]Converting precipitation units from kg/m²/s to mm/day[/yellow]")
        pr_data = pr_data * 86400  # Convert to mm/day
        
        # Handle coordinate systems
        pr_data = self._standardize_coordinates(pr_data)
        
        # Determine processing approach
        if chunk_by_county and len(gdf) > 50:
            console.print("[cyan]Using county-chunked processing for large datasets[/cyan]")
            return self._process_chunked_counties_precip(pr_data, gdf, scenario, threshold_mm)
        else:
            console.print("[cyan]Using vectorized processing[/cyan]")
            return self._process_vectorized_precip(pr_data, gdf, scenario, threshold_mm)
    
    def _process_temperature_data(
        self,
        ds: xr.Dataset,
        gdf: gpd.GeoDataFrame,
        scenario: str,
        chunk_by_county: bool
    ) -> pd.DataFrame:
        """Process temperature data (tas variable)."""
        # Get temperature data
        tas_data = ds['tas']
        
        # Convert from Kelvin to Celsius
        console.print("[yellow]Converting temperature units from Kelvin to Celsius[/yellow]")
        tas_data = tas_data - 273.15  # Convert to Celsius
        
        # Handle coordinate systems
        tas_data = self._standardize_coordinates(tas_data)
        
        # Determine processing approach
        if chunk_by_county and len(gdf) > 50:
            console.print("[cyan]Using county-chunked processing for large datasets[/cyan]")
            return self._process_chunked_counties_temp(tas_data, gdf, scenario)
        else:
            console.print("[cyan]Using vectorized processing[/cyan]")
            return self._process_vectorized_temp(tas_data, gdf, scenario)
    
    def _process_tasmax_data(
        self,
        ds: xr.Dataset,
        gdf: gpd.GeoDataFrame,
        scenario: str,
        threshold_temp_c: float,
        chunk_by_county: bool
    ) -> pd.DataFrame:
        """Process daily maximum temperature data (tasmax variable)."""
        # Get daily maximum temperature data
        tasmax_data = ds['tasmax']
        
        # Convert from Kelvin to Celsius
        console.print("[yellow]Converting daily maximum temperature units from Kelvin to Celsius[/yellow]")
        tasmax_data = tasmax_data - 273.15  # Convert to Celsius
        
        # Convert threshold from Fahrenheit to Celsius if it's 90 (90°F = 32.2°C)
        if abs(threshold_temp_c - 90.0) < 0.1:  # Check if threshold is likely in Fahrenheit
            threshold_temp_c = (threshold_temp_c - 32) * 5.0/9.0
            console.print(f"[yellow]Converting threshold from 90°F to {threshold_temp_c:.1f}°C[/yellow]")
        
        # Handle coordinate systems
        tasmax_data = self._standardize_coordinates(tasmax_data)
        
        # Determine processing approach
        if chunk_by_county and len(gdf) > 50:
            console.print("[cyan]Using county-chunked processing for large datasets[/cyan]")
            return self._process_chunked_counties_tasmax(tasmax_data, gdf, scenario, threshold_temp_c)
        else:
            console.print("[cyan]Using vectorized processing[/cyan]")
            return self._process_vectorized_tasmax(tasmax_data, gdf, scenario, threshold_temp_c)
    
    def _process_tasmin_data(
        self,
        ds: xr.Dataset,
        gdf: gpd.GeoDataFrame,
        scenario: str,
        chunk_by_county: bool
    ) -> pd.DataFrame:
        """Process daily minimum temperature data (tasmin variable)."""
        # Get daily minimum temperature data
        tasmin_data = ds['tasmin']
        
        # Convert from Kelvin to Celsius
        console.print("[yellow]Converting daily minimum temperature units from Kelvin to Celsius[/yellow]")
        tasmin_data = tasmin_data - 273.15  # Convert to Celsius
        
        # Handle coordinate systems
        tasmin_data = self._standardize_coordinates(tasmin_data)
        
        # Determine processing approach
        if chunk_by_county and len(gdf) > 50:
            console.print("[cyan]Using county-chunked processing for large datasets[/cyan]")
            return self._process_chunked_counties_tasmin(tasmin_data, gdf, scenario)
        else:
            console.print("[cyan]Using vectorized processing[/cyan]")
            return self._process_vectorized_tasmin(tasmin_data, gdf, scenario)
    
    def _standardize_coordinates(self, data: xr.DataArray) -> xr.DataArray:
        """Standardize coordinate system and spatial reference."""
        # Rename dimensions for rioxarray compatibility
        if 'lon' in data.dims and 'lat' in data.dims:
            data = data.rename({'lon': 'x', 'lat': 'y'})
        
        # Add spatial reference
        data = data.rio.write_crs('EPSG:4326')
        
        # Handle longitude wrapping if needed
        if 'x' in data.coords and float(data.x.max()) > 180:
            data = data.assign_coords(x=(data.x + 180) % 360 - 180)
            data = data.sortby('x')
            data = data.rio.write_crs('EPSG:4326')
        
        return data
    
    def _process_vectorized_temp(
        self,
        tas_data: xr.DataArray,
        gdf: gpd.GeoDataFrame,
        scenario: str
    ) -> pd.DataFrame:
        """Process all counties using vectorized operations for temperature."""
        
        console.print("[yellow]Processing counties with temperature data using rioxarray clipping...[/yellow]")
        
        results = []
        
        # Get time information
        time_values = tas_data.time.values
        if hasattr(time_values[0], 'year'):
            years = np.array([t.year for t in time_values])
        else:
            years = pd.to_datetime(time_values).year
        
        unique_years = np.unique(years)
        
        console.print(f"[cyan]Processing {len(gdf)} counties over {len(unique_years)} years[/cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("Processing counties...", total=len(gdf))
            
            for idx, county in gdf.iterrows():
                try:
                    # Clip data to county using rioxarray
                    clipped = tas_data.rio.clip([county.geometry], all_touched=True)
                    
                    if clipped.size > 0:
                        for year in unique_years:
                            year_mask = years == year
                            year_data = clipped.isel(time=year_mask)
                            
                            # Calculate daily means
                            daily_means = year_data.mean(dim=['y', 'x']).values
                            valid_days = daily_means[~np.isnan(daily_means)]
                            
                            if len(valid_days) > 0:
                                # Temperature-specific statistics
                                results.append({
                                    'year': year,
                                    'scenario': scenario,
                                    'county_id': county['county_id'],
                                    'county_name': county['county_name'],
                                    'state': county['state'],
                                    'mean_annual_temp_c': float(np.mean(valid_days)),
                                    'min_temp_c': float(np.min(valid_days)),
                                    'max_temp_c': float(np.max(valid_days)),
                                    'temp_range_c': float(np.max(valid_days) - np.min(valid_days)),
                                    'temp_std_c': float(np.std(valid_days)),
                                    'days_below_freezing': int(np.sum(valid_days < 0)),
                                    'days_above_30c': int(np.sum(valid_days > 30)),
                                    'growing_degree_days': float(np.sum(np.maximum(valid_days - 10, 0))),  # Base 10°C
                                })
                        
                except Exception as e:
                    console.print(f"[red]Error processing {county['county_name']}: {e}[/red]")
                
                progress.advance(task)
        
        return pd.DataFrame(results)
    
    def _process_chunked_counties_temp(
        self,
        tas_data: xr.DataArray,
        gdf: gpd.GeoDataFrame,
        scenario: str
    ) -> pd.DataFrame:
        """Process counties in chunks for memory efficiency (temperature version)."""
        
        # Split counties into chunks
        chunk_size = max(10, len(gdf) // self.n_workers)
        county_chunks = [
            gdf.iloc[i:i+chunk_size] 
            for i in range(0, len(gdf), chunk_size)
        ]
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            if self.client and DISTRIBUTED_AVAILABLE:
                # Use Dask distributed processing
                task = progress.add_task("Processing county chunks (distributed)...", total=len(county_chunks))
                
                futures = []
                for chunk in county_chunks:
                    future = self.client.submit(
                        self._process_county_chunk_temp,
                        tas_data,
                        chunk,
                        scenario
                    )
                    futures.append(future)
                
                for future in as_completed(futures):
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    progress.advance(task)
            
            else:
                # Use multiprocessing
                task = progress.add_task("Processing county chunks (multiprocessing)...", total=len(county_chunks))
                
                with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                    futures = [
                        executor.submit(
                            self._process_county_chunk_temp,
                            tas_data,
                            chunk,
                            scenario
                        )
                        for chunk in county_chunks
                    ]
                    
                    for future in as_completed(futures):
                        chunk_results = future.result()
                        results.extend(chunk_results)
                        progress.advance(task)
        
        return pd.DataFrame(results)

    def _process_chunked_counties(
        self,
        pr_data: xr.DataArray,
        gdf: gpd.GeoDataFrame,
        scenario: str,
        threshold_mm: float
    ) -> pd.DataFrame:
        """Process counties in chunks for memory efficiency."""
        
        # Split counties into chunks
        chunk_size = max(10, len(gdf) // self.n_workers)
        county_chunks = [
            gdf.iloc[i:i+chunk_size] 
            for i in range(0, len(gdf), chunk_size)
        ]
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            if self.client and DISTRIBUTED_AVAILABLE:
                # Use Dask distributed processing
                task = progress.add_task("Processing county chunks (distributed)...", total=len(county_chunks))
                
                futures = []
                for chunk in county_chunks:
                    future = self.client.submit(
                        self._process_county_chunk,
                        pr_data,
                        chunk,
                        scenario,
                        threshold_mm
                    )
                    futures.append(future)
                
                for future in as_completed(futures):
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    progress.advance(task)
            
            else:
                # Use multiprocessing
                task = progress.add_task("Processing county chunks (multiprocessing)...", total=len(county_chunks))
                
                with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                    futures = [
                        executor.submit(
                            self._process_county_chunk,
                            pr_data,
                            chunk,
                            scenario,
                            threshold_mm
                        )
                        for chunk in county_chunks
                    ]
                    
                    for future in as_completed(futures):
                        chunk_results = future.result()
                        results.extend(chunk_results)
                        progress.advance(task)
        
        return pd.DataFrame(results)
    
    def _process_vectorized(
        self,
        pr_data: xr.DataArray,
        gdf: gpd.GeoDataFrame,
        scenario: str,
        threshold_mm: float
    ) -> pd.DataFrame:
        """Process all counties using vectorized operations."""
        
        # Use rioxarray clipping instead of manual rasterization for better compatibility
        console.print("[yellow]Processing counties with rioxarray clipping...[/yellow]")
        
        results = []
        
        # Get time information
        time_values = pr_data.time.values
        if hasattr(time_values[0], 'year'):
            years = np.array([t.year for t in time_values])
        else:
            years = pd.to_datetime(time_values).year
        
        unique_years = np.unique(years)
        
        console.print(f"[cyan]Processing {len(gdf)} counties over {len(unique_years)} years[/cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("Processing counties...", total=len(gdf))
            
            for idx, county in gdf.iterrows():
                try:
                    # Clip data to county using rioxarray
                    clipped = pr_data.rio.clip([county.geometry], all_touched=True)
                    
                    if clipped.size > 0:
                        for year in unique_years:
                            year_mask = years == year
                            year_data = clipped.isel(time=year_mask)
                            
                            # Calculate daily means
                            daily_means = year_data.mean(dim=['y', 'x']).values
                            valid_days = daily_means[~np.isnan(daily_means)]
                            
                            if len(valid_days) > 0:
                                results.append({
                                    'year': year,
                                    'scenario': scenario,
                                    'county_id': county['county_id'],
                                    'county_name': county['county_name'],
                                    'state': county['state'],
                                    'total_annual_precip_mm': float(np.sum(valid_days)),
                                    'days_above_25.4mm': int(np.sum(valid_days > threshold_mm)),
                                    'mean_daily_precip_mm': float(np.mean(valid_days)),
                                    'max_daily_precip_mm': float(np.max(valid_days)),
                                    'precip_std_mm': float(np.std(valid_days)),
                                    'dry_days': int(np.sum(valid_days < 0.1)),
                                })
                        
                except Exception as e:
                    console.print(f"[red]Error processing {county['county_name']}: {e}[/red]")
                
                progress.advance(task)
        
        return pd.DataFrame(results)

    # Rename the existing methods for clarity
    def _process_chunked_counties_precip(
        self,
        pr_data: xr.DataArray,
        gdf: gpd.GeoDataFrame,
        scenario: str,
        threshold_mm: float
    ) -> pd.DataFrame:
        """Process counties in chunks for memory efficiency (precipitation version)."""
        return self._process_chunked_counties(pr_data, gdf, scenario, threshold_mm)
    
    def _process_vectorized_precip(
        self,
        pr_data: xr.DataArray,
        gdf: gpd.GeoDataFrame,
        scenario: str,
        threshold_mm: float
    ) -> pd.DataFrame:
        """Process all counties using vectorized operations (precipitation version)."""
        return self._process_vectorized(pr_data, gdf, scenario, threshold_mm)
    
    def _calculate_stats_vectorized(
        self,
        pr_data: xr.DataArray,
        county_raster: np.ndarray,
        gdf: gpd.GeoDataFrame,
        scenario: str,
        threshold_mm: float
    ) -> pd.DataFrame:
        """Calculate statistics using vectorized operations."""
        
        # Get time information
        time_values = pr_data.time.values
        if hasattr(time_values[0], 'year'):
            years = np.array([t.year for t in time_values])
        else:
            years = pd.to_datetime(time_values).year
        
        unique_years = np.unique(years)
        results = []
        
        console.print(f"[cyan]Processing {len(unique_years)} years with vectorized operations[/cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("Processing years...", total=len(unique_years))
            
            for year in unique_years:
                year_mask = years == year
                pr_year = pr_data.isel(time=year_mask).values
                
                # Vectorized calculation for all counties
                unique_ids = np.unique(county_raster[county_raster > 0])
                
                for county_id in unique_ids:
                    county_mask = county_raster == county_id
                    county_info = gdf[gdf.raster_id == county_id].iloc[0]
                    
                    # Calculate daily means efficiently
                    daily_means = np.array([
                        np.mean(day_data[county_mask]) 
                        for day_data in pr_year
                        if np.any(county_mask)
                    ])
                    
                    if len(daily_means) > 0:
                        results.append({
                            'year': year,
                            'scenario': scenario,
                            'county_id': county_info['county_id'],
                            'county_name': county_info['county_name'],
                            'state': county_info['state'],
                            'total_annual_precip_mm': float(np.sum(daily_means)),
                            'days_above_25.4mm': int(np.sum(daily_means > threshold_mm)),
                            'mean_daily_precip_mm': float(np.mean(daily_means)),
                            'max_daily_precip_mm': float(np.max(daily_means)),
                            'precip_std_mm': float(np.std(daily_means)),
                            'dry_days': int(np.sum(daily_means < 0.1)),
                        })
                
                progress.advance(task)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def _process_county_chunk(
        pr_data: xr.DataArray,
        county_chunk: gpd.GeoDataFrame,
        scenario: str,
        threshold_mm: float
    ) -> List[Dict]:
        """Process a chunk of counties (for parallel execution)."""
        results = []
        
        for idx, county in county_chunk.iterrows():
            try:
                # Clip data to county
                clipped = pr_data.rio.clip([county.geometry], all_touched=True)
                
                if clipped.size > 0:
                    # Get time info
                    time_values = clipped.time.values
                    if hasattr(time_values[0], 'year'):
                        years = np.array([t.year for t in time_values])
                    else:
                        years = pd.to_datetime(time_values).year
                    
                    for year in np.unique(years):
                        year_mask = years == year
                        year_data = clipped.isel(time=year_mask)
                        
                        # Calculate daily means
                        daily_means = year_data.mean(dim=['y', 'x']).values
                        
                        if len(daily_means) > 0:
                            results.append({
                                'year': year,
                                'scenario': scenario,
                                'county_id': county['county_id'],
                                'county_name': county['county_name'],
                                'state': county['state'],
                                'total_annual_precip_mm': float(np.sum(daily_means)),
                                'days_above_25.4mm': int(np.sum(daily_means > threshold_mm)),
                                'mean_daily_precip_mm': float(np.mean(daily_means)),
                                'max_daily_precip_mm': float(np.max(daily_means)),
                                'precip_std_mm': float(np.std(daily_means)),
                                'dry_days': int(np.sum(daily_means < 0.1)),
                            })
            except Exception as e:
                console.print(f"[red]Error processing {county['county_name']}: {e}[/red]")
        
        return results
    
    @staticmethod
    def _process_county_chunk_temp(
        tas_data: xr.DataArray,
        county_chunk: gpd.GeoDataFrame,
        scenario: str
    ) -> List[Dict]:
        """Process a chunk of counties for temperature data (for parallel execution)."""
        results = []
        
        for idx, county in county_chunk.iterrows():
            try:
                # Clip data to county
                clipped = tas_data.rio.clip([county.geometry], all_touched=True)
                
                if clipped.size > 0:
                    # Get time info
                    time_values = clipped.time.values
                    if hasattr(time_values[0], 'year'):
                        years = np.array([t.year for t in time_values])
                    else:
                        years = pd.to_datetime(time_values).year
                    
                    for year in np.unique(years):
                        year_mask = years == year
                        year_data = clipped.isel(time=year_mask)
                        
                        # Calculate daily means
                        daily_means = year_data.mean(dim=['y', 'x']).values
                        valid_days = daily_means[~np.isnan(daily_means)]
                        
                        if len(valid_days) > 0:
                            results.append({
                                'year': year,
                                'scenario': scenario,
                                'county_id': county['county_id'],
                                'county_name': county['county_name'],
                                'state': county['state'],
                                'mean_annual_temp_c': float(np.mean(valid_days)),
                                'min_temp_c': float(np.min(valid_days)),
                                'max_temp_c': float(np.max(valid_days)),
                                'temp_range_c': float(np.max(valid_days) - np.min(valid_days)),
                                'temp_std_c': float(np.std(valid_days)),
                                'days_below_freezing': int(np.sum(valid_days < 0)),
                                'days_above_30c': int(np.sum(valid_days > 30)),
                                'growing_degree_days': float(np.sum(np.maximum(valid_days - 10, 0))),
                            })
            except Exception as e:
                console.print(f"[red]Error processing {county['county_name']}: {e}[/red]")
        
        return results
    
    def _process_vectorized_tasmax(
        self,
        tasmax_data: xr.DataArray,
        gdf: gpd.GeoDataFrame,
        scenario: str,
        threshold_temp_c: float
    ) -> pd.DataFrame:
        """Process all counties using vectorized operations for daily maximum temperature."""
        
        console.print("[yellow]Processing counties with daily maximum temperature data using rioxarray clipping...[/yellow]")
        
        results = []
        
        # Get time information
        time_values = tasmax_data.time.values
        if hasattr(time_values[0], 'year'):
            years = np.array([t.year for t in time_values])
        else:
            years = pd.to_datetime(time_values).year
        
        unique_years = np.unique(years)
        
        console.print(f"[cyan]Processing {len(gdf)} counties over {len(unique_years)} years[/cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("Processing counties...", total=len(gdf))
            
            for idx, county in gdf.iterrows():
                try:
                    # Clip data to county using rioxarray
                    clipped = tasmax_data.rio.clip([county.geometry], all_touched=True)
                    
                    if clipped.size > 0:
                        for year in unique_years:
                            year_mask = years == year
                            year_data = clipped.isel(time=year_mask)
                            
                            # Calculate daily means
                            daily_means = year_data.mean(dim=['y', 'x']).values
                            valid_days = daily_means[~np.isnan(daily_means)]
                            
                            if len(valid_days) > 0:
                                # Daily maximum temperature statistics
                                results.append({
                                    'year': year,
                                    'scenario': scenario,
                                    'county_id': county['county_id'],
                                    'county_name': county['county_name'],
                                    'state': county['state'],
                                    'mean_annual_tasmax_c': float(np.mean(valid_days)),
                                    'min_tasmax_c': float(np.min(valid_days)),
                                    'max_tasmax_c': float(np.max(valid_days)),
                                    'tasmax_range_c': float(np.max(valid_days) - np.min(valid_days)),
                                    'tasmax_std_c': float(np.std(valid_days)),
                                    'days_below_freezing_max': int(np.sum(valid_days < 0)),
                                    'days_above_30c_max': int(np.sum(valid_days > 30)),
                                    'days_above_threshold_c': int(np.sum(valid_days > threshold_temp_c)),
                                    'threshold_temp_c': float(threshold_temp_c),
                                    'growing_degree_days_max': float(np.sum(np.maximum(valid_days - 10, 0))),  # Base 10°C
                                })
                        
                except Exception as e:
                    console.print(f"[red]Error processing {county['county_name']}: {e}[/red]")
                
                progress.advance(task)
        
        return pd.DataFrame(results)
    
    def _process_vectorized_tasmin(
        self,
        tasmin_data: xr.DataArray,
        gdf: gpd.GeoDataFrame,
        scenario: str
    ) -> pd.DataFrame:
        """Process all counties using vectorized operations for daily minimum temperature."""
        
        console.print("[yellow]Processing counties with daily minimum temperature data using rioxarray clipping...[/yellow]")
        
        results = []
        
        # Get time information
        time_values = tasmin_data.time.values
        if hasattr(time_values[0], 'year'):
            years = np.array([t.year for t in time_values])
        else:
            years = pd.to_datetime(time_values).year
        
        unique_years = np.unique(years)
        
        console.print(f"[cyan]Processing {len(gdf)} counties over {len(unique_years)} years[/cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("Processing counties...", total=len(gdf))
            
            for idx, county in gdf.iterrows():
                try:
                    # Clip data to county using rioxarray
                    clipped = tasmin_data.rio.clip([county.geometry], all_touched=True)
                    
                    if clipped.size > 0:
                        for year in unique_years:
                            year_mask = years == year
                            year_data = clipped.isel(time=year_mask)
                            
                            # Calculate daily means
                            daily_means = year_data.mean(dim=['y', 'x']).values
                            valid_days = daily_means[~np.isnan(daily_means)]
                            
                            if len(valid_days) > 0:
                                # Daily minimum temperature statistics with cold day analysis
                                results.append({
                                    'year': year,
                                    'scenario': scenario,
                                    'county_id': county['county_id'],
                                    'county_name': county['county_name'],
                                    'state': county['state'],
                                    'mean_annual_tasmin_c': float(np.mean(valid_days)),
                                    'min_tasmin_c': float(np.min(valid_days)),
                                    'max_tasmin_c': float(np.max(valid_days)),
                                    'tasmin_range_c': float(np.max(valid_days) - np.min(valid_days)),
                                    'tasmin_std_c': float(np.std(valid_days)),
                                    'cold_days': int(np.sum(valid_days < 0)),  # Days below 0°C (freezing)
                                    'extreme_cold_days': int(np.sum(valid_days < -10)),  # Days below -10°C
                                    'very_extreme_cold_days': int(np.sum(valid_days < -20)),  # Days below -20°C
                                    'days_above_freezing': int(np.sum(valid_days >= 0)),  # Days at or above 0°C
                                    'frost_free_period_start': float(np.where(valid_days >= 0)[0][0] if np.any(valid_days >= 0) else -1),
                                    'frost_free_period_end': float(np.where(valid_days >= 0)[0][-1] if np.any(valid_days >= 0) else -1),
                                    'growing_degree_days_min': float(np.sum(np.maximum(valid_days - 0, 0))),  # Base 0°C for cold regions
                                    'heating_degree_days': float(np.sum(np.maximum(18 - valid_days, 0))),  # Base 18°C for heating
                                })
                        
                except Exception as e:
                    console.print(f"[red]Error processing {county['county_name']}: {e}[/red]")
                
                progress.advance(task)
        
        return pd.DataFrame(results)
    
    def _process_chunked_counties_tasmax(
        self,
        tasmax_data: xr.DataArray,
        gdf: gpd.GeoDataFrame,
        scenario: str,
        threshold_temp_c: float
    ) -> pd.DataFrame:
        """Process counties in chunks for memory efficiency (daily maximum temperature version)."""
        
        # Split counties into chunks
        chunk_size = max(10, len(gdf) // self.n_workers)
        county_chunks = [
            gdf.iloc[i:i+chunk_size] 
            for i in range(0, len(gdf), chunk_size)
        ]
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            if self.client and DISTRIBUTED_AVAILABLE:
                # Use Dask distributed processing
                task = progress.add_task("Processing county chunks (distributed)...", total=len(county_chunks))
                
                futures = []
                for chunk in county_chunks:
                    future = self.client.submit(
                        self._process_county_chunk_tasmax,
                        tasmax_data,
                        chunk,
                        scenario,
                        threshold_temp_c
                    )
                    futures.append(future)
                
                for future in as_completed(futures):
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    progress.advance(task)
            
            else:
                # Use multiprocessing
                task = progress.add_task("Processing county chunks (multiprocessing)...", total=len(county_chunks))
                
                with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                    futures = [
                        executor.submit(
                            self._process_county_chunk_tasmax,
                            tasmax_data,
                            chunk,
                            scenario,
                            threshold_temp_c
                        )
                        for chunk in county_chunks
                    ]
                    
                    for future in as_completed(futures):
                        chunk_results = future.result()
                        results.extend(chunk_results)
                        progress.advance(task)
        
        return pd.DataFrame(results)

    def _process_chunked_counties_tasmin(
        self,
        tasmin_data: xr.DataArray,
        gdf: gpd.GeoDataFrame,
        scenario: str
    ) -> pd.DataFrame:
        """Process counties in chunks for memory efficiency (daily minimum temperature version)."""
        
        # Split counties into chunks
        chunk_size = max(10, len(gdf) // self.n_workers)
        county_chunks = [
            gdf.iloc[i:i+chunk_size] 
            for i in range(0, len(gdf), chunk_size)
        ]
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            if self.client and DISTRIBUTED_AVAILABLE:
                # Use Dask distributed processing
                task = progress.add_task("Processing county chunks (distributed)...", total=len(county_chunks))
                
                futures = []
                for chunk in county_chunks:
                    future = self.client.submit(
                        self._process_county_chunk_tasmin,
                        tasmin_data,
                        chunk,
                        scenario
                    )
                    futures.append(future)
                
                for future in as_completed(futures):
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    progress.advance(task)
            
            else:
                # Use multiprocessing
                task = progress.add_task("Processing county chunks (multiprocessing)...", total=len(county_chunks))
                
                with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                    futures = [
                        executor.submit(
                            self._process_county_chunk_tasmin,
                            tasmin_data,
                            chunk,
                            scenario
                        )
                        for chunk in county_chunks
                    ]
                    
                    for future in as_completed(futures):
                        chunk_results = future.result()
                        results.extend(chunk_results)
                        progress.advance(task)
        
        return pd.DataFrame(results)

    @staticmethod
    def _process_county_chunk_tasmax(
        tasmax_data: xr.DataArray,
        county_chunk: gpd.GeoDataFrame,
        scenario: str,
        threshold_temp_c: float
    ) -> List[Dict]:
        """Process a chunk of counties for daily maximum temperature data (for parallel execution)."""
        results = []
        
        for idx, county in county_chunk.iterrows():
            try:
                # Clip data to county
                clipped = tasmax_data.rio.clip([county.geometry], all_touched=True)
                
                if clipped.size > 0:
                    # Get time info
                    time_values = clipped.time.values
                    if hasattr(time_values[0], 'year'):
                        years = np.array([t.year for t in time_values])
                    else:
                        years = pd.to_datetime(time_values).year
                    
                    for year in np.unique(years):
                        year_mask = years == year
                        year_data = clipped.isel(time=year_mask)
                        
                        # Calculate daily means
                        daily_means = year_data.mean(dim=['y', 'x']).values
                        valid_days = daily_means[~np.isnan(daily_means)]
                        
                        if len(valid_days) > 0:
                            results.append({
                                'year': year,
                                'scenario': scenario,
                                'county_id': county['county_id'],
                                'county_name': county['county_name'],
                                'state': county['state'],
                                'mean_annual_tasmax_c': float(np.mean(valid_days)),
                                'min_tasmax_c': float(np.min(valid_days)),
                                'max_tasmax_c': float(np.max(valid_days)),
                                'tasmax_range_c': float(np.max(valid_days) - np.min(valid_days)),
                                'tasmax_std_c': float(np.std(valid_days)),
                                'days_below_freezing_max': int(np.sum(valid_days < 0)),
                                'days_above_30c_max': int(np.sum(valid_days > 30)),
                                'days_above_threshold_c': int(np.sum(valid_days > threshold_temp_c)),
                                'threshold_temp_c': float(threshold_temp_c),
                                'growing_degree_days_max': float(np.sum(np.maximum(valid_days - 10, 0))),
                            })
            except Exception as e:
                console.print(f"[red]Error processing {county['county_name']}: {e}[/red]")
        
        return results

    @staticmethod
    def _process_county_chunk_tasmin(
        tasmin_data: xr.DataArray,
        county_chunk: gpd.GeoDataFrame,
        scenario: str
    ) -> List[Dict]:
        """Process a chunk of counties for daily minimum temperature data (for parallel execution)."""
        results = []
        
        for idx, county in county_chunk.iterrows():
            try:
                # Clip data to county
                clipped = tasmin_data.rio.clip([county.geometry], all_touched=True)
                
                if clipped.size > 0:
                    # Get time info
                    time_values = clipped.time.values
                    if hasattr(time_values[0], 'year'):
                        years = np.array([t.year for t in time_values])
                    else:
                        years = pd.to_datetime(time_values).year
                    
                    for year in np.unique(years):
                        year_mask = years == year
                        year_data = clipped.isel(time=year_mask)
                        
                        # Calculate daily means
                        daily_means = year_data.mean(dim=['y', 'x']).values
                        valid_days = daily_means[~np.isnan(daily_means)]
                        
                        if len(valid_days) > 0:
                            results.append({
                                'year': year,
                                'scenario': scenario,
                                'county_id': county['county_id'],
                                'county_name': county['county_name'],
                                'state': county['state'],
                                'mean_annual_tasmin_c': float(np.mean(valid_days)),
                                'min_tasmin_c': float(np.min(valid_days)),
                                'max_tasmin_c': float(np.max(valid_days)),
                                'tasmin_range_c': float(np.max(valid_days) - np.min(valid_days)),
                                'tasmin_std_c': float(np.std(valid_days)),
                                'cold_days': int(np.sum(valid_days < 0)),  # Days below 0°C (freezing)
                                'extreme_cold_days': int(np.sum(valid_days < -10)),  # Days below -10°C
                                'very_extreme_cold_days': int(np.sum(valid_days < -20)),  # Days below -20°C
                                'days_above_freezing': int(np.sum(valid_days >= 0)),  # Days at or above 0°C
                                'frost_free_period_start': float(np.where(valid_days >= 0)[0][0] if np.any(valid_days >= 0) else -1),
                                'frost_free_period_end': float(np.where(valid_days >= 0)[0][-1] if np.any(valid_days >= 0) else -1),
                                'growing_degree_days_min': float(np.sum(np.maximum(valid_days - 0, 0))),  # Base 0°C for cold regions
                                'heating_degree_days': float(np.sum(np.maximum(18 - valid_days, 0))),  # Base 18°C for heating
                            })
            except Exception as e:
                console.print(f"[red]Error processing {county['county_name']}: {e}[/red]")
        
        return results

    def close(self):
        """Clean up resources."""
        if self.client:
            self.client.close()


def main():
    """Main function with modern CLI."""
    parser = argparse.ArgumentParser(
        description="Calculate county statistics using modern 2025 techniques"
    )
    parser.add_argument(
        "zarr_path",
        type=Path,
        help="Path to Zarr dataset"
    )
    parser.add_argument(
        "shapefile_path", 
        type=Path,
        help="Path to county shapefile"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default="county_stats_modern.csv",
        help="Output CSV file (default: county_stats_modern.csv)"
    )
    parser.add_argument(
        "-s", "--scenario",
        type=str,
        default="historical",
        help="Scenario name (default: historical)"
    )
    parser.add_argument(
        "-v", "--variable",
        type=str,
        default="pr",
        choices=["pr", "tas", "tasmax", "tasmin"],
        help="Variable to process: 'pr' for precipitation, 'tas' for temperature, 'tasmax' for daily maximum temperature, 'tasmin' for daily minimum temperature (default: pr)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=25.4,
        help="Threshold value: for precipitation in mm (default: 25.4), for temperature in °F or °C (e.g., 90 for 90°F/32.2°C)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=4,
        help="Number of worker processes (default: 4)"
    )
    parser.add_argument(
        "--memory-limit",
        type=str,
        default="4GB",
        help="Memory limit per worker (default: 4GB)"
    )
    parser.add_argument(
        "--use-distributed",
        action="store_true",
        help="Use Dask distributed processing"
    )
    parser.add_argument(
        "--chunk-by-county",
        action="store_true",
        help="Process counties in chunks (for large datasets)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.zarr_path.exists():
        console.print(f"[red]Zarr path does not exist: {args.zarr_path}[/red]")
        return
    
    if not args.shapefile_path.exists():
        console.print(f"[red]Shapefile does not exist: {args.shapefile_path}[/red]")
        return
    
    # Create processor
    processor = ModernCountyProcessor(
        n_workers=args.workers,
        memory_limit=args.memory_limit,
        use_distributed=args.use_distributed
    )
    
    try:
        # Load shapefile
        gdf = processor.prepare_shapefile(args.shapefile_path)
        
        # Process data
        results_df = processor.process_zarr_data(
            args.zarr_path,
            gdf,
            args.scenario,
            args.variable,
            args.threshold,
            args.chunk_by_county
        )
        
        # Save results
        results_df.to_csv(args.output, index=False)
        
        # Show summary
        table = Table(title="Processing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Counties Processed", str(len(results_df['county_id'].unique())))
        table.add_row("Years Processed", str(len(results_df['year'].unique())))
        table.add_row("Total Records", str(len(results_df)))
        table.add_row("Variable", args.variable.upper())
        table.add_row("Output File", str(args.output))
        
        console.print(table)
        console.print("[green]✅ Processing completed successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Error during processing: {e}[/red]")
        raise
    
    finally:
        processor.close()


if __name__ == "__main__":
    main() 