#!/usr/bin/env python
"""Daily maximum temperature data processor for county-level statistics."""

from pathlib import Path
import pandas as pd
import geopandas as gpd
import xarray as xr
from rich.console import Console

from .base_processor import BaseCountyProcessor
from .processing_strategies import VectorizedStrategy, UltraFastStrategy
from ..utils.data_utils import convert_units

console = Console()


class TasMaxProcessor(BaseCountyProcessor):
    """Processor for daily maximum temperature data (tasmax variable)."""
    
    def process_variable_data(
        self,
        data: xr.DataArray,
        gdf: gpd.GeoDataFrame,
        scenario: str,
        threshold_temp_c: float = 32.2,  # Default: 90°F converted to Celsius
        chunk_by_county: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """Process daily maximum temperature data for all counties.
        
        Args:
            data: Daily maximum temperature data array
            gdf: County geometries
            scenario: Scenario name
            threshold_temp_c: Temperature threshold in Celsius for hot days
            chunk_by_county: Whether to use chunked processing
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with daily maximum temperature statistics
        """
        console.print("[blue]Processing daily maximum temperature data...[/blue]")
        
        # Convert units from Kelvin to Celsius
        tasmax_data = convert_units(data, "K", "C")
        
        # Handle threshold conversion if it looks like Fahrenheit
        if abs(threshold_temp_c - 90.0) < 0.1:  # Check if threshold is likely in Fahrenheit
            threshold_temp_c = convert_units(threshold_temp_c, "F", "C")
            console.print(f"[yellow]Converting threshold from 90°F to {threshold_temp_c:.1f}°C[/yellow]")
        
        # Standardize coordinates
        tasmax_data = self._standardize_coordinates(tasmax_data)
        
        # Choose processing strategy
        strategy = self._select_processing_strategy(gdf, chunk_by_county)
        
        # Process the data
        return strategy.process(
            data=tasmax_data,
            gdf=gdf,
            variable='tasmax',
            scenario=scenario,
            threshold=threshold_temp_c,
            n_workers=self.n_workers
        )
    
    def _select_processing_strategy(self, gdf: gpd.GeoDataFrame, chunk_by_county: bool):
        """Select processing strategy for daily maximum temperature data.
        
        Args:
            gdf: County geometries
            chunk_by_county: Whether chunked processing is requested
            
        Returns:
            Selected processing strategy instance
        """
        if len(gdf) > 50 or not chunk_by_county:
            console.print("[cyan]🚀 Using ultra-fast processing (best performance)[/cyan]")
            return UltraFastStrategy()
        else:
            console.print("[cyan]Using vectorized processing[/cyan]")
            return VectorizedStrategy()
    
    def process_zarr_file(
        self,
        zarr_path: Path,
        gdf: gpd.GeoDataFrame,
        scenario: str = 'historical',
        threshold_temp_c: float = 32.2,
        chunk_by_county: bool = True
    ) -> pd.DataFrame:
        """Process a Zarr file containing daily maximum temperature data.
        
        Args:
            zarr_path: Path to Zarr dataset
            gdf: County geometries
            scenario: Scenario name
            threshold_temp_c: Temperature threshold in Celsius
            chunk_by_county: Whether to use chunked processing
            
        Returns:
            DataFrame with daily maximum temperature statistics
        """
        console.print(f"[blue]Opening daily maximum temperature Zarr dataset:[/blue] {zarr_path}")
        
        # Open with optimizations
        ds = xr.open_zarr(zarr_path, chunks={'time': 365})
        
        # Get daily maximum temperature data
        if 'tasmax' not in ds.data_vars:
            raise ValueError("Daily maximum temperature variable 'tasmax' not found in dataset")
        
        tasmax_data = ds['tasmax']
        
        return self.process_variable_data(
            data=tasmax_data,
            gdf=gdf,
            scenario=scenario,
            threshold_temp_c=threshold_temp_c,
            chunk_by_county=chunk_by_county
        ) 