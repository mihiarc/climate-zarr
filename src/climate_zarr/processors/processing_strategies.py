#!/usr/bin/env python
"""Streamlined processing strategy for county-level climate data analysis."""

from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path

import pandas as pd
import geopandas as gpd
import xarray as xr
from rich.console import Console

from ..utils.spatial_utils import (
    get_time_information, 
    get_coordinate_arrays,
    clip_county_data,
    get_spatial_dims,
    standardize_for_clipping
)
from ..utils.data_utils import calculate_statistics

console = Console()


class ProcessingStrategy(ABC):
    """Abstract base class for processing strategies."""
    
    @abstractmethod
    def process(
        self,
        data: xr.DataArray,
        gdf: gpd.GeoDataFrame,
        variable: str,
        scenario: str,
        threshold: float,
        n_workers: int = 4,
        zarr_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """Process climate data using this strategy.
        
        Args:
            data: Climate data array
            gdf: County geometries
            variable: Climate variable name
            scenario: Scenario name
            threshold: Threshold value
            n_workers: Number of workers
            zarr_path: Path to zarr file for multiprocessing
            
        Returns:
            DataFrame with processed results
        """
        pass


class UltraFastStrategy(ProcessingStrategy):
    """Ultra-fast strategy using Zarr streaming for memory-efficient multiprocessing."""
    
    @staticmethod
    def _process_county_chunk(args):
        """Static method for processing county chunks in worker processes."""
        try:
            (county_records, zarr_path_local, variable, scenario, threshold, years_arr, unique_years_arr, lon_range) = args
            
            # Open DataArray within the process to avoid pickling issues
            ds = xr.open_zarr(zarr_path_local) if zarr_path_local else None
            data_local = ds[variable] if ds is not None else None
            
            # Apply coordinate normalization in worker process (same as main process)
            if data_local is not None and lon_range is not None:
                from climate_zarr.utils.spatial_utils import normalize_longitude_coordinates
                import geopandas as gpd
                
                # Create a proper temporary GeoDataFrame from county records
                temp_gdf = gpd.GeoDataFrame(county_records, crs='EPSG:4326')
                temp_gdf = normalize_longitude_coordinates(temp_gdf, lon_range)
                
                # Update county_records with normalized geometries - ensure we maintain Series objects
                county_records = [temp_gdf.iloc[i] for i in range(len(county_records))]

            # Standardize data for clipping using utility function
            if data_local is not None:
                data_local = standardize_for_clipping(data_local)
            local_results = []
            
            for county in county_records:
                try:
                    try:
                        from climate_zarr.utils.spatial_utils import clip_county_data, get_spatial_dims
                        clipped = clip_county_data(data_local, county.geometry) if data_local is not None else None
                        use_clipped_data = True
                    except Exception:
                        use_clipped_data = False
                    for year in unique_years_arr:
                        year_mask = years_arr == year
                        if use_clipped_data and clipped.size > 0:
                            year_data = clipped.isel(time=year_mask)
                            if year_data.size > 0:
                                spatial_dims = get_spatial_dims(year_data)
                                if not spatial_dims:
                                    continue
                                daily_means = year_data.mean(dim=spatial_dims, skipna=True).values
                            else:
                                continue
                        else:
                            year_data = data_local.isel(time=year_mask) if data_local is not None else None
                            try:
                                year_clipped = clip_county_data(year_data, county.geometry) if year_data is not None else None
                                if year_clipped.size > 0:
                                    spatial_dims = get_spatial_dims(year_clipped)
                                    if not spatial_dims:
                                        continue
                                    daily_means = year_clipped.mean(dim=spatial_dims, skipna=True).values
                                else:
                                    continue
                            except Exception:
                                continue
                        county_info = {
                            'county_id': county['county_id'],
                            'county_name': county['county_name'],
                            'state': county['state']
                        }
                        from climate_zarr.utils.data_utils import calculate_statistics
                        stats = calculate_statistics(daily_means, variable, threshold, year, scenario, county_info)
                        if stats:
                            local_results.append(stats)
                except Exception:
                    continue
            return local_results
        except Exception as e:
            # Return error info for debugging
            return [{'error': str(e), 'chunk_size': len(county_records)}]
    
    def process(
        self,
        data: xr.DataArray,
        gdf: gpd.GeoDataFrame,
        variable: str,
        scenario: str,
        threshold: float,
        n_workers: int = 4,
        zarr_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """Process using ultra-fast vectorized operations."""
        # Validate variable early to propagate a clear error
        if variable not in {"pr", "tas", "tasmax", "tasmin"}:
            raise ValueError(f"Unsupported variable: {variable}")
        
        console.print("[yellow]🚀 Zarr streaming processing (county-level multiprocessing capable)...[/yellow]")
        
        # Get coordinate arrays and time information
        lats, lons = get_coordinate_arrays(data)
        years, unique_years = get_time_information(data)
        console.print(f"[cyan]Zarr shape: {data.shape} (time, lat, lon) - streaming access[/cyan]")
        console.print(f"[cyan]Processing {len(gdf)} counties × {len(unique_years)} years = {len(gdf) * len(unique_years)} records[/cyan]")
        
        # County-level multiprocessing implementation
        import math
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Decide execution mode
        small_job = len(gdf) < 8
        if n_workers is None or n_workers <= 1 or small_job or not zarr_path:
            # Single-process fallback
            results = []
            console.print(f"[blue]🔍 Starting single-process processing for {len(gdf)} counties...[/blue]")
            
            counties_processed = 0
            counties_with_data = 0
            
            for _, county in gdf.iterrows():
                counties_processed += 1
                try:
                    try:
                        clipped = clip_county_data(data, county.geometry)
                        use_clipped_data = True
                        console.print(f"[green]✅ County {counties_processed}/{len(gdf)}: {county.get('county_name', 'Unknown')} - clipping successful (size: {clipped.size})[/green]")
                    except Exception as e:
                        use_clipped_data = False
                        console.print(f"[yellow]⚠️ County {counties_processed}/{len(gdf)}: {county.get('county_name', 'Unknown')} - clipping failed: {e}[/yellow]")
                    for year in unique_years:
                        year_mask = years == year
                        if use_clipped_data and clipped.size > 0:
                            year_data = clipped.isel(time=year_mask)
                            if year_data.size == 0:
                                continue
                            spatial_dims = get_spatial_dims(year_data)
                            if not spatial_dims:
                                continue
                            daily_means = year_data.mean(dim=spatial_dims, skipna=True).values
                        else:
                            year_data = data.isel(time=year_mask)
                            try:
                                year_clipped = clip_county_data(year_data, county.geometry)
                                if year_clipped.size == 0:
                                    continue
                                spatial_dims = get_spatial_dims(year_clipped)
                                if not spatial_dims:
                                    continue
                                daily_means = year_clipped.mean(dim=spatial_dims, skipna=True).values
                            except Exception:
                                continue
                        county_info = {
                            'county_id': county['county_id'],
                            'county_name': county['county_name'],
                            'state': county['state']
                        }
                        stats = calculate_statistics(daily_means, variable, threshold, year, scenario, county_info)
                        if stats:
                            results.append(stats)
                            counties_with_data += 1
                            if counties_with_data <= 3:  # Show first few successful counties
                                console.print(f"[green]✅ Successfully processed year {year} for {county.get('county_name', 'Unknown')}[/green]")
                except Exception as e:
                    console.print(f"[red]❌ Error processing county {counties_processed}: {e}[/red]")
                    continue
            
            console.print(f"[blue]📊 Processing summary: {counties_with_data} counties with data out of {counties_processed} processed[/blue]")
            return pd.DataFrame(results)

        # Multiprocessing path
        console.print(f"[blue]🚀 Starting multiprocessing with {n_workers} workers...[/blue]")
        counties_list = [row[1] for row in gdf.iterrows()]
        chunk_size = max(1, math.ceil(len(counties_list) / n_workers))
        chunks = [counties_list[i:i+chunk_size] for i in range(0, len(counties_list), chunk_size)]
        console.print(f"[blue]Created {len(chunks)} chunks of ~{chunk_size} counties each[/blue]")

        # Get longitude range for coordinate normalization in worker processes
        lon_range = None
        if 'lon' in data.coords:
            lon_range = (float(data.lon.min()), float(data.lon.max()))
        elif 'longitude' in data.coords:
            lon_range = (float(data.longitude.min()), float(data.longitude.max()))
        elif 'x' in data.coords:  # Check for standardized coordinate name
            lon_range = (float(data.x.min()), float(data.x.max()))

        results: list = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(UltraFastStrategy._process_county_chunk, (chunk, zarr_path, variable, scenario, threshold, years, unique_years, lon_range)) for chunk in chunks]
            for fut in as_completed(futures):
                try:
                    chunk_results = fut.result()
                    results.extend(chunk_results)
                    console.print(f"[green]✅ Worker completed: {len(chunk_results)} results[/green]")
                except Exception as e:
                    console.print(f"[red]❌ Worker failed: {e}[/red]")
                    import traceback
                    console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
                    continue
        
        # Ensure DataFrame has expected columns even if empty
        if len(results) == 0:
            empty_cols = ['year','scenario','county_id','county_name','state']
            # Add variable-specific columns commonly used in tests
            if variable == 'pr':
                empty_cols += ['total_annual_precip_mm','days_above_threshold','dry_days','mean_daily_precip_mm','max_daily_precip_mm']
            elif variable == 'tas':
                empty_cols += ['mean_annual_temp_c','days_below_freezing','growing_degree_days','min_temp_c','max_temp_c','days_above_30c']
            elif variable == 'tasmax':
                empty_cols += ['days_above_threshold','max_temp_c','mean_annual_temp_c']
            elif variable == 'tasmin':
                empty_cols += ['days_below_freezing','min_temp_c','mean_annual_temp_c']
            return pd.DataFrame(columns=empty_cols)
        return pd.DataFrame(results)