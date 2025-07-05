#!/usr/bin/env python
"""Simplified processing strategies for county-level climate data analysis."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from ..utils.spatial_utils import (
    create_county_raster, 
    get_time_information, 
    get_coordinate_arrays,
    clip_county_data
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
        n_workers: int = 4
    ) -> pd.DataFrame:
        """Process climate data using this strategy.
        
        Args:
            data: Climate data array
            gdf: County geometries
            variable: Climate variable name
            scenario: Scenario name
            threshold: Threshold value
            n_workers: Number of workers
            
        Returns:
            DataFrame with processed results
        """
        pass


class VectorizedStrategy(ProcessingStrategy):
    """Vectorized processing strategy using rioxarray clipping."""
    
    def process(
        self,
        data: xr.DataArray,
        gdf: gpd.GeoDataFrame,
        variable: str,
        scenario: str,
        threshold: float,
        n_workers: int = 4
    ) -> pd.DataFrame:
        """Process using vectorized operations with rioxarray clipping."""
        
        console.print("[yellow]Processing counties with vectorized rioxarray clipping...[/yellow]")
        
        results = []
        years, unique_years = get_time_information(data)
        
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
                    clipped = clip_county_data(data, county.geometry)
                    
                    if clipped.size > 0:
                        for year in unique_years:
                            year_mask = years == year
                            year_data = clipped.isel(time=year_mask)
                            
                            # Calculate daily means
                            daily_means = year_data.mean(dim=['y', 'x']).values
                            
                            # Calculate statistics
                            county_info = {
                                'county_id': county['county_id'],
                                'county_name': county['county_name'],
                                'state': county['state']
                            }
                            
                            stats = calculate_statistics(
                                daily_means, variable, threshold, year, scenario, county_info
                            )
                            
                            if stats:
                                results.append(stats)
                        
                except Exception as e:
                    console.print(f"[red]Error processing {county['county_name']}: {e}[/red]")
                
                progress.advance(task)
        
        return pd.DataFrame(results)


class UltraFastStrategy(ProcessingStrategy):
    """Ultra-fast strategy that loads all data into memory for maximum speed."""
    
    def process(
        self,
        data: xr.DataArray,
        gdf: gpd.GeoDataFrame,
        variable: str,
        scenario: str,
        threshold: float,
        n_workers: int = 4
    ) -> pd.DataFrame:
        """Process using ultra-fast vectorized operations."""
        
        console.print("[yellow]🚀 Ultra-fast vectorized processing (all counties + years simultaneously)...[/yellow]")
        
        # Get coordinate arrays
        lats, lons = get_coordinate_arrays(data)
        console.print(f"[cyan]Zarr shape: {data.shape} (time, lat, lon)[/cyan]")
        
        # Create county raster mask
        county_raster = create_county_raster(gdf, lats, lons)
        unique_county_ids = np.unique(county_raster[county_raster > 0])
        
        # Load ALL data into memory (key optimization)
        console.print("[cyan]Loading all zarr data into memory...[/cyan]")
        all_data = data.values  # Shape: (time, lat, lon)
        
        # Get time information
        years, unique_years = get_time_information(data)
        console.print(f"[cyan]Processing {len(unique_county_ids)} counties × {len(unique_years)} years = {len(unique_county_ids) * len(unique_years)} records[/cyan]")
        
        # Ultra-fast vectorized computation
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            task = progress.add_task("Ultra-fast processing...", total=len(unique_years))
            
            for year in unique_years:
                year_mask = years == year
                year_data = all_data[year_mask]  # Shape: (days_in_year, lat, lon)
                
                # Vectorized processing for ALL counties at once
                for county_id in unique_county_ids:
                    county_mask = county_raster == county_id
                    county_info_row = gdf[gdf.raster_id == county_id].iloc[0]
                    
                    county_info = {
                        'county_id': county_info_row['county_id'],
                        'county_name': county_info_row['county_name'],
                        'state': county_info_row['state']
                    }
                    
                    if np.any(county_mask):
                        # Ultra-fast vectorized calculation
                        county_data = year_data[:, county_mask]  # Shape: (days, n_pixels)
                        daily_means = np.mean(county_data, axis=1)  # Shape: (days,)
                        
                        # Calculate statistics
                        stats = calculate_statistics(
                            daily_means, variable, threshold, year, scenario, county_info
                        )
                        
                        if stats:
                            results.append(stats)
                
                progress.advance(task)
        
        return pd.DataFrame(results) 